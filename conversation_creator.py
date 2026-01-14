from utils.eval_other_utils import chunk_text_into_sentences
from utils.eval_data_utils import load_eval_data
from utils.templates import get_template

import json
import logging
import tiktoken
import os
import random
import yaml
from glob import glob

from openai import OpenAI

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConversationCreator:
    """
    A class responsible for creating conversation data structures from various datasets.
    
    This class handles:
    - Loading dataset configurations
    - Processing contexts and questions/answers
    - Converting data into appropriate formats for agent consumption
    - Chunking text data for memory agents
    """

    def __init__(self, agent_config, dataset_config):
        """
        Initialize the ConversationCreator with agent and dataset configurations.
        
        Args:
            agent_config: Configuration dictionary for the agent
            dataset_config: Configuration dictionary for the dataset
        """
        # Store core configuration parameters
        self.dataset = dataset_config['dataset']
        self.max_test_samples = dataset_config['max_test_samples']
        self.context_max_length = dataset_config['context_max_length']
        self.agent_name = agent_config['agent_name']
        self.sub_dataset = dataset_config['sub_dataset']
        
        #cut the context
        self.dataset_config = dataset_config
        self.chunk_keep_ratio = float(dataset_config.get('chunk_keep_ratio', 1.0))
        self.chunk_keep_seed = dataset_config.get('chunk_keep_seed', 2026)
        self._validate_chunk_keep_ratio()
        self.retrieve_num = agent_config.get('retrieve_num')
        self._validate_chunk_keep_ratio()
        # Determine chunk size based on agent type
        self.chunk_size = self._determine_chunk_size(agent_config, dataset_config)
        
        #新增部分
        #keep full dataset config for augmentation metadata
        self.dataset_config = dataset_config
        self.context_markers = []
        self.context_sections = []
        self.all_dataset_items = []
        self.unused_context_pool = []
        self.external_oot_items = []
        self.external_oot_docs = None
        self.num_samples_to_process = 0
        
        # Process the dataset and create conversation structures
        self._load_and_process_dataset(dataset_config)

    def _determine_chunk_size(self, agent_config, dataset_config):
        """
        Determine the appropriate chunk size based on agent configuration.
        
        Args:
            agent_config: Agent configuration dictionary
            dataset_config: Dataset configuration dictionary
            
        Returns:
            int: The chunk size to use for text processing
        """
        # Memory agents (mem0, letta, cognee) use agent-specific chunk size
        if agent_config.get('agent_chunk_size') is not None:
            assert any(agent_name in agent_config['agent_name'] 
                      for agent_name in ["mem0", "letta", "cognee", "zep"]), \
                   "agent_chunk_size should only be set for memory agents"
            
            chunk_size = agent_config['agent_chunk_size']
            print(f"\n\nUsing agent-specific chunk_size: {chunk_size}\n\n")
            return chunk_size
        else:
            # Default to dataset chunk size
            return dataset_config['chunk_size']
    #1.加载数据
    def _load_and_process_dataset(self, dataset_config):
        """
        Load the dataset and process it into contexts and query-answer pairs.
        
        Args:
            dataset_config: Dataset configuration dictionary
        """
        logger.info(f"Running test on {self.sub_dataset}")
      #2转换成列表格式
        # Load and convert dataset to processable format
        loaded_dataset = load_eval_data(dataset_config)
        dataset_items = self._convert_dataset_format(loaded_dataset)
        
        # Determine how many samples to process
        #num_samples_to_process = min(len(dataset_items), self.max_test_samples) #决定测试数量
        
        #增加部分
        # Track all dataset items with their indices so out-of-task memories can reuse
        # unused documents form the same split
        self.all_dataset_items = list(enumerate(dataset_items))
        
        #detemine how many samples to process and precompute the unused pool for
        #out-of-task memory injection
        self.num_samples_to_process = min(len(dataset_items), self.max_test_samples)
        self.unused_context_pool = self.all_dataset_items[self.num_samples_to_process:]
        ###
        

        # # Process each dataset item using list comprehension for better performance #转换为列表的目的
        # processed_items = [
        #     self._process_dataset_item(dataset_items[i]) 
        #     for i in range(num_samples_to_process)
        # ]
        
        #修改部分Process each dataset item using list comprehension for better performance
        processed_items = [
            self._process_dataset_item(dataset_items[i], i)
            for i in range(self.num_samples_to_process)
        ]
        
        # Unpack contexts and query-answer pairs
        self.contexts, self.query_and_answers = zip(*processed_items) if processed_items else ([], [])
        self.contexts, self.query_and_answers = list(self.contexts), list(self.query_and_answers)

        #新增部分 store marker metadata back to config for downstream metrics
        self.dataset_config["context_markers"] = self.context_markers
        
        if self.contexts:
            print(f"Dataset length: {len(self.contexts)}, each sample has {len(self.query_and_answers[0])} qa pairs")
        else:
            print("Dataset is empty - no samples found matching the filter criteria")
            raise ValueError(f"No samples found for sub_dataset: {self.sub_dataset}. Please check the dataset configuration.")
     #转化为列表函数
    def _convert_dataset_format(self, loaded_dataset):
        """
        Convert dataset from various formats to a consistent list format.
        
        Args:
            loaded_dataset: Raw dataset loaded from load_data()
            
        Returns:
            list: Dataset items in list format for consistent processing
        """
        # Handle both old format (direct list) and new HuggingFace format (dict with 'data' key)
        return (list(loaded_dataset['data']) 
                if isinstance(loaded_dataset, dict) and 'data' in loaded_dataset 
                else loaded_dataset)
   #3.这部分是最重要的，它将一个原始数据集条目（Context + 一组 Q/A 数据）转换成 MAB 框架所需的格式。
    def _process_dataset_item(self, dataset_item, dataset_index):
        """
        Process a single dataset item to extract context and create query-answer pairs.
        
        Args:
            dataset_item: Single item from the dataset
            
        Returns:
            tuple: (context_text, list_of_qa_pairs)
        """
        # Extract and validate context
        context_text = dataset_item["context"]
        assert len(context_text) > 2000, f"Context too short: {len(context_text)} characters"
        
        #新增部分
        # Apply optional context augmentation (repeat / biased / out-of-task / attack)
        context_text, marker, sections = self._augment_context(context_text, dataset_index)
        self.context_markers.append(marker)
        self.context_sections.append(sections)
        
        # Extract all non-context fields for question generation
        question_data = {key: value for key, value in dataset_item.items() if key != "context"}
        
        # Create query-answer pairs from the question data
        qa_pairs = self._create_query_answer_pairs(question_data)
        
        return context_text, qa_pairs

    def _augment_context(self, context_text, dataset_index):
        """Augment context with repeated, biased, out-of-task, and attack memory blocks."""
        repeat_cfg = self.dataset_config.get("repeat_memory", {}) or {}
        biased_cfg = self.dataset_config.get("biased_memory", {}) or {}
        oot_cfg = self.dataset_config.get("out_of_task_memory", {}) or {}
        attack_cfg = self.dataset_config.get("attack_chunk", {}) or {}

        sections = []
        context_marker = {
            "repeat_labels": [],
            "biased_labels": [],
            "out_of_task_labels": [],
            "attack_label": None,
            "section_labels": []
        }

        # Always keep the original document as the first section
        original_label = repeat_cfg.get("original_label", "[Original]")
        sections.append({
            "label": original_label,
            "type": "original",
            "text": context_text
        })
        context_marker["original_label"] = original_label
        context_marker["section_labels"].append(original_label)

        sections, context_marker = self._apply_repeat_memory(context_text, repeat_cfg, sections, context_marker)
        sections, context_marker = self._apply_biased_memory(context_text, biased_cfg, sections, context_marker, dataset_index)
        sections, context_marker = self._apply_out_of_task_memory(
            oot_cfg, sections, context_marker, dataset_index, context_text
        )

        augmented_context = "\n\n".join(f"{section['label']}\n{section['text']}" for section in sections)
        sections, context_marker, augmented_context = self._apply_attack_chunk(attack_cfg, sections, context_marker, augmented_context)

        self._log_augmentation_summary(dataset_index, repeat_cfg, biased_cfg, oot_cfg, attack_cfg, sections)

        return augmented_context, context_marker, sections

    def _apply_repeat_memory(self, context_text, repeat_cfg, sections, context_marker):
        """Handle repeat_memory augmentation independently."""
        repeat_count = int(repeat_cfg.get("copies", 0) or 0)
        if repeat_count <= 0:
            return sections, context_marker

        repeat_label_template = repeat_cfg.get("label_template", "[Repeat {index}]")
        for idx in range(repeat_count):
            label = repeat_label_template.format(index=idx + 1)
            sections.append({
                "label": label,
                "type": "repeat",
                "text": context_text,
                "copy_index": idx + 1
            })
            context_marker["repeat_labels"].append(label)
            context_marker["section_labels"].append(label)

        return sections, context_marker

    def _apply_biased_memory(self, context_text, biased_cfg, sections, context_marker, dataset_index):
        """Handle biased_memory augmentation independently."""
        biased_sections = self._create_biased_memories(context_text, biased_cfg, dataset_index)
        for section in biased_sections:
            sections.append(section)
            context_marker["biased_labels"].append(section["label"])
            context_marker["section_labels"].append(section["label"])

        return sections, context_marker

    def _apply_out_of_task_memory(self, oot_cfg, sections, context_marker, dataset_index, context_text):
        """Handle out_of_task_memory augmentation independently."""
        oot_sections = self._create_out_of_task_memories(oot_cfg, dataset_index, context_text)
        for section in oot_sections:
            sections.append(section)
            context_marker["out_of_task_labels"].append(section["label"])
            context_marker["section_labels"].append(section["label"])

        return sections, context_marker

    def _apply_attack_chunk(self, attack_cfg, sections, context_marker, augmented_context):
        """Optionally append an attack chunk without coupling to other augmentations."""
        if not attack_cfg.get("enabled"):
            return sections, context_marker, augmented_context

        attack_content = attack_cfg.get("content", "")
        attack_label = attack_cfg.get("label", "[Attack Chunk]")
        attack_section = {
            "label": attack_label,
            "type": "attack",
            "text": attack_content
        }
        sections.append(attack_section)
        augmented_context = f"{augmented_context}\n\n{attack_label}\n{attack_content}"
        context_marker["attack_label"] = attack_label
        context_marker["section_labels"].append(attack_label)
        return sections, context_marker, augmented_context

    def _log_augmentation_summary(self, dataset_index, repeat_cfg, biased_cfg, oot_cfg, attack_cfg, sections):
        """Print a concise summary showing which augmentations are active for this context."""
        summary_parts = [
            f"原文:1",
            f"repeat:{int(repeat_cfg.get('copies', 0) or 0)}",
            f"biased:{int(biased_cfg.get('count', 0) or 0)}",
            f"oot:{int(oot_cfg.get('count', 0) or 0)}",
            f"attack:{'on' if attack_cfg.get('enabled') else 'off'}",
        ]

        labels_preview = [section.get("label", "") for section in sections]
        logger.info(
            "Context %s augmentation -> %s | labels: %s",
            dataset_index,
            ", ".join(summary_parts),
            labels_preview,
        )

    def _create_biased_memories(self, context_text, biased_cfg, dataset_index):
        """
        Create biased memories as *new documents* by rewriting larger source spans before chunking.

        Instead of在检索后的 chunk 上做改写，这里先对原文的大段文本进行 LLM 改写，
        形成独立的 biased 文档 section，后续会与原文一起被切分为 chunk 参与检索。
        """
        count = int(biased_cfg.get("count", 0) or 0)
        if count <= 0:
            return []

        label_template = biased_cfg.get("label_template", "[Biased {index}]")
        preserve_full_length = bool(biased_cfg.get("preserve_full_length", True))
        max_source_chars = int(biased_cfg.get("max_source_chars", 8000) or 8000)
        source_chunk_size = biased_cfg.get("source_chunk_size", self.chunk_size)
        max_chunks = int(biased_cfg.get("max_chunks") or 0)
        cache_root = biased_cfg.get("cache_dir")
        cache_enabled = bool(cache_root)

        # 先把原文按句子粗切，再逐段改写后重连，形成单独的 biased 文档
        source_chunks = chunk_text_into_sentences(context_text, chunk_size=source_chunk_size)
        if not source_chunks:
            return []

        sections = []
        for idx in range(count):
            if preserve_full_length:
                selected_chunks = list(source_chunks)
                excerpt = "\n".join(selected_chunks)
            else:
                selected_chunks = list(source_chunks)
                if max_chunks > 0 and len(selected_chunks) > max_chunks:
                    selected_chunks = random.sample(selected_chunks, k=max_chunks)

                # 控制总输入长度，超出时随机截取连续片段
                excerpt = "\n".join(selected_chunks)
                if max_source_chars > 0 and len(excerpt) > max_source_chars:
                    start_pos = random.randint(0, max(0, len(excerpt) - max_source_chars))
                    excerpt = excerpt[start_pos:start_pos + max_source_chars]
                    selected_chunks = excerpt.split("\n")

            cache_path = None
            if cache_enabled:
                cache_path = os.path.join(
                    cache_root,
                    self.sub_dataset,
                    f"context_{dataset_index}_biased_{idx + 1}.json"
                )
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            rewritten_doc = None
            # 优先读取缓存，避免重复改写
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                        rewritten_doc = cached.get("text")
                except Exception:
                    rewritten_doc = None

            # 无缓存或缓存异常时重新改写并写入缓存
            if not rewritten_doc:
                rewritten_chunks = [
                    self._rewrite_chunk_with_llm(chunk, biased_cfg)
                    for chunk in selected_chunks
                ]
                rewritten_doc = "\n".join(rewritten_chunks)
                if cache_path:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "label": label_template.format(index=idx + 1),
                                "text": rewritten_doc,
                                "source_excerpt": "\n".join(selected_chunks)
                            }, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

            excerpt_joined = "\n".join(selected_chunks)

            label = label_template.format(index=idx + 1)
            sections.append({
                "label": label,
                "type": "biased",
                "text": rewritten_doc,
                "biased_index": idx + 1,
                "source_excerpt": excerpt_joined,
                "source_span": {
                    "start": 0,
                    "end": len(excerpt_joined)
                }
            })
        return sections

    def _create_out_of_task_memories(self, oot_cfg, dataset_index, context_text):
        """
        Create out-of-task memories by sampling *unused* documents from the same dataset.

        Priority:
        1) Use documents that were not selected as main samples (beyond max_test_samples).
        2) If不足，则从当前 split 的其他样本（排除当前文档）中随机补足。
        3) 若仍不足，再回落到 content_pool/file_paths 提供的外部文本，仍不够时切分当前文档或使用占位文本。
        """
        count = int(oot_cfg.get("count", 0) or 0)
        if count <= 0:
            return []

        label_template = oot_cfg.get("label_template", "[Out-of-Task {index}]")
        min_tokens_per_doc = int(oot_cfg.get("min_tokens_per_doc") or max(self.chunk_size * 4, 1024))
        match_original = bool(oot_cfg.get("match_original_chunk_count"))
        original_chunk_estimate = 0
        target_chunk_ratio = float(oot_cfg.get("target_chunk_ratio", 1.0) or 1.0)
        max_tokens_per_doc = int(oot_cfg.get("max_tokens_per_doc") or 0)
        if context_text and match_original:
            original_chunk_estimate = len(chunk_text_into_sentences(context_text, chunk_size=self.chunk_size))
            if original_chunk_estimate > 0:
                min_tokens_per_doc = max(min_tokens_per_doc, original_chunk_estimate * self.chunk_size)
                if max_tokens_per_doc <= 0:
                    max_tokens_per_doc = int(original_chunk_estimate * self.chunk_size * target_chunk_ratio)
        use_external_docs = self.dataset == "Test_Time_Learning" and oot_cfg.get("external_dataset_configs")
        if use_external_docs and self.external_oot_docs is None:
            self.external_oot_docs = self.load_out_of_task_docs(oot_cfg["external_dataset_configs"])
        
        sources = []
        if not use_external_docs:
            sources = self._select_out_of_task_sources(dataset_index, count, context_text, oot_cfg)
        
        external_texts = self._load_external_texts(oot_cfg)
        external_docs = list(external_texts)
        random.shuffle(external_docs)
        fallback_windows = self._slice_text_into_windows(context_text, min_tokens_per_doc) if context_text else []
        fallback_cursor = 0

        sections = []
        for idx in range(count):
            # label = label_template.format(index=idx + 1)
            label_context = {
                "index": idx + 1,
                "source": None,
            }

        #     # Pick a base text for this out-of-task doc
        #     if idx < len(sources):
        #         source_idx, source_item = sources[idx]
        #         filler = source_item.get("context", "")
        #         source_dataset_index = source_idx
        #         source_metadata = source_item.get("metadata", {})
        #         label_context["source"] = (
        #             source_item.get("metadata", {}).get("source")
        #             or source_item.get("source")
        #             or source_item.get("metadata",{}).get("sub_dataset")
        #         )
        #     elif external_docs:
        #         filler = external_docs.pop(0)
        #         source_dataset_index = None
        #         source_metadata = {}
        #     else:
        #         # Fallback: use external pools only when同数据集的备选文档不足
        #         if fallback_cursor < len(fallback_windows):
        #             filler = fallback_windows[fallback_cursor]
        #             fallback_cursor += 1
        #         else:
        #             base_text = "Out-of-task memo about unrelated subject matter to distract retrieval."
        #             filler = self._repeat_to_min_tokens(base_text, min_tokens_per_doc)
        #         source_dataset_index = None
        #         source_metadata = {}
                
        #     #ttl
        #     label = label_template.format(**label_context)
        #     #

        #     filler_expanded = self._ensure_min_tokens(filler, min_tokens_per_doc)
        #     if max_tokens_per_doc > 0:
        #         filler_expanded = self._truncate_to_tokens(filler_expanded, max_tokens_per_doc)
        #     sections.append({
        #         "label": label,
        #         "type": "out_of_task",
        #         "text": filler_expanded,
        #         "out_of_task_index": idx + 1,
        #         "source_dataset_index": source_dataset_index,
        #         "source_name": label_context["source"],
        #         "source_metadata": source_metadata
        #     })
        # return sections
        
                    # Pick a base text for this out-of-task doc
            if use_external_docs and self.external_oot_docs and idx < len(self.external_oot_docs):
                external_doc = self.external_oot_docs[idx]
                filler = external_doc.get("text", "")
                source_dataset_index = None
                source_metadata = {"source": external_doc.get("name")}
                label_context["source"] = external_doc.get("name")
            elif idx < len(sources):
                source_idx, source_item = sources[idx]
                filler = source_item.get("context", "")
                source_dataset_index = source_idx
                source_metadata = source_item.get("metadata", {})
                label_context["source"] = (
                    source_item.get("metadata", {}).get("source")
                    or source_item.get("source")
                    or source_item.get("metadata", {}).get("sub_dataset")
                )
            elif external_docs:
                filler = external_docs.pop(0)
                source_dataset_index = None
                source_metadata = {}
            else:
                # Fallback: use external pools only when同数据集的备选文档不足
                if fallback_cursor < len(fallback_windows):
                    filler = fallback_windows[fallback_cursor]
                    fallback_cursor += 1
                else:
                    base_text = "Out-of-task memo about unrelated subject matter to distract retrieval."
                    filler = self._repeat_to_min_tokens(base_text, min_tokens_per_doc)
                source_dataset_index = None
                source_metadata = {}

            label = label_template.format(**label_context)
            filler_expanded = self._ensure_min_tokens(filler, min_tokens_per_doc)
            if max_tokens_per_doc > 0:
                filler_expanded = self._truncate_to_tokens(filler_expanded, max_tokens_per_doc)
            sections.append({
                "label": label,
                "type": "out_of_task",
                "text": filler_expanded,
                "out_of_task_index": idx + 1,
                "source_dataset_index": source_dataset_index,
                "source_name": label_context["source"],
                "source_metadata": source_metadata
            })
        return sections


    def _select_out_of_task_sources(self, current_index, count, current_context_text, oot_cfg):
        """优先选取未被用作主样本的文档，其次随机抽取其他样本，并用 Jaccard 过滤相似度。"""
        sources = []
        max_retries = 50
        threshold = 0.2

        # Load external dataset contexts if configured
        # if self.external_oot_items is None and oot_cfg.get("external_dataset_config"):
        #     self.external_oot_items = self._load_external_oot_items(oot_cfg["external_dataset_config"])
        if self.external_oot_items is None:
            external_configs = oot_cfg.get("external_dataset_configs")
            if external_configs and self.dataset == "Test_Time_Learning":
                self.external_oot_items = self._load_multiple_external_oot_items(external_configs)
            elif oot_cfg.get("external_dataset_config"):
                self.external_oot_items = self._load_external_oot_items(oot_cfg["external_dataset_config"])

        def _pick_from_pool(pool, exclude_current=True):
            nonlocal sources
            retries = 0
            while pool and len(sources) < count and retries < max_retries:
                candidate = pool.pop(0)
                retries += 1
                if exclude_current and candidate[0] == current_index:
                    continue
                # candidate context text
                candidate_ctx = candidate[1].get("context", "")
                sim = self._calculate_jaccard_similarity(current_context_text, candidate_ctx)
                if sim <= threshold:
                    sources.append(candidate)
                else:
                    # put it back to the end to avoid losing it entirely
                    pool.append(candidate)
            return retries
        
        # For TTL with external configs, prioritize external pool and skip internal sources.
        if self.dataset == "Test_Time_Learning" and oot_cfg.get("external_dataset_configs"):
            sources = self._select_from_external_pool(
                current_context_text, count, threshold, max_retries
            )
            return sources

        # Step 1: 未使用的样本池（max_test_samples 之外）
        _ = _pick_from_pool(self.unused_context_pool, exclude_current=True)

        # Step 2: 若不够，用当前 split 其他样本补足，排除当前文档
        if len(sources) < count:
            candidates = [
                item for item in self.all_dataset_items
                if item[0] != current_index and item not in sources
            ]
            random.shuffle(candidates)
            retries = 0
            while candidates and len(sources) < count and retries < max_retries:
                candidate = candidates.pop(0)
                retries += 1
                candidate_ctx = candidate[1].get("context", "")
                sim = self._calculate_jaccard_similarity(current_context_text, candidate_ctx)
                if sim <= threshold:
                    sources.append(candidate)
                else:
                    candidates.append(candidate)

            # Fallback: 如果尝试多次仍不足，强制补足
            if len(sources) < count and candidates:
                needed = count - len(sources)
                sources.extend(candidates[:needed])

        # Step 3: 若仍不足，尝试外部数据集（如果存在）
        # if len(sources) < count and self.external_oot_items:
        #     ext_candidates = [item for item in self.external_oot_items if item not in sources]
        #     random.shuffle(ext_candidates)
        #     retries = 0
        #     while ext_candidates and len(sources) < count and retries < max_retries:
        #         candidate = ext_candidates.pop(0)
        #         retries += 1
        #         candidate_ctx = candidate[1].get("context", "")
        #         sim = self._calculate_jaccard_similarity(current_context_text, candidate_ctx)
        #         if sim <= threshold:
        #             sources.append(candidate)
        #         else:
        #             ext_candidates.append(candidate)

        #     if len(sources) < count and ext_candidates:
        #         needed = count - len(sources)
        #         sources.extend(ext_candidates[:needed])

        # return sources
        if len(sources) < count and self.external_oot_items:
            sources.extend(
                self._select_from_external_pool(
                    current_context_text, count - len(sources), threshold, max_retries
                )
            )

        return sources
    
    def _select_from_external_pool(self, current_context_text, count, threshold, max_retries):
        """Select OOT items from external datasets with Jaccard filtering."""
        if not self.external_oot_items or count <= 0:
            return []
        sources = []
        ext_candidates = [item for item in self.external_oot_items]
        random.shuffle(ext_candidates)
        retries = 0
        while ext_candidates and len(sources) < count and retries < max_retries:
            candidate = ext_candidates.pop(0)
            retries += 1
            candidate_ctx = candidate[1].get("context", "")
            sim = self._calculate_jaccard_similarity(current_context_text, candidate_ctx)
            if sim <= threshold:
                sources.append(candidate)
            else:
                ext_candidates.append(candidate)
        if len(sources) < count and ext_candidates:
            needed = count - len(sources)
            sources.extend(ext_candidates[:needed])
        return sources
    
    def _calculate_jaccard_similarity(self, text1, text2):
        """计算两段文本的 Jaccard 相似度（基于空格分词）。"""
        if not text1 or not text2:
            return 0.0
        set1 = set(text1.split())
        set2 = set(text2.split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
            

    def _ensure_min_tokens(self, text, min_tokens):
        """Pad text (by repetition) until it reaches a minimum token length."""
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        if not text:
            text = "Out-of-task memo about unrelated subject matter to distract retrieval."

        tokenized = encoding.encode(text)
        if len(tokenized) >= min_tokens:
            return text

        # Repeat the text until reaching min_tokens
        repeats = (min_tokens // max(len(tokenized), 1)) + 1
        padded = ("\n\n".join([text] * repeats))[: min_tokens * 8]  # cap length to avoid runaway
        return padded

    def _repeat_to_min_tokens(self, text, min_tokens):
        """Simple wrapper for clarity."""
        return self._ensure_min_tokens(text, min_tokens)

    def _slice_text_into_windows(self, text, target_tokens):
        """Slice a long text into windows (by tokens) to serve as fallback OOT docs."""
        if not text:
            return []
        windows = chunk_text_into_sentences(text, chunk_size=target_tokens)
        if not windows:
            return []
        # Re-join adjacent windows to avoid tiny fragments
        return [" ".join(windows[max(i - 1, 0): i + 1]) if i > 0 else windows[i]
                for i in range(len(windows))]
        
    def _load_external_texts(self, oot_cfg):
        """Load external documents for out-of-task memories from paths or content pool."""
        texts = []
        content_pool = oot_cfg.get("content_pool", [])
        if content_pool:
            texts.extend(content_pool)

        file_paths = oot_cfg.get("file_paths", []) or []
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        file_dir = oot_cfg.get("file_dir")
        if file_dir and os.path.isdir(file_dir):
            file_paths.extend(sorted(glob(os.path.join(file_dir, "**", "*"), recursive=True)))

        for path in file_paths:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except (UnicodeDecodeError, OSError):
                continue

        return texts
        
    def _truncate_to_tokens(self, text, max_tokens):
        """Truncate text to a maximum token count using the default tokenizer."""
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])

    def _load_external_oot_items(self, config_path):
        """Load contexts from another dataset config to use as OOT candidates."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                ext_cfg = yaml.safe_load(f)
        except Exception as exc:
            logger.warning("Failed to load external_dataset_config %s: %s", config_path, exc)
            return []

        try:
            loaded = load_eval_data(ext_cfg)
            items = self._convert_dataset_format(loaded)
            source_name = ext_cfg.get("sub_dataset") or os.path.splitext(os.path.basename(config_path))[0]
            enriched = []
            for idx, item in enumerate(items):
                metadata = dict(item.get("metadata", {}))
                metadata.setdefault("source", source_name)
                item = dict(item)
                item.setdefault("source", source_name)
                item["metadata"] = metadata
                enriched.append((idx, item))
            return enriched
        except Exception as exc:
            logger.warning("Failed to prepare external OOT items from %s: %s", config_path, exc)
            return []
        
    def load_out_of_task_docs(self, config_paths):
        """
        Load external datasets as whole-doc OOT memories.

        Returns:
            list: [{"text": "...", "name": "sub_dataset"}, ...]
        """
        docs = []
        for config_path in config_paths:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    ext_cfg = yaml.safe_load(f)
            except Exception as exc:
                logger.warning("Failed to load external_dataset_config %s: %s", config_path, exc)
                continue

            try:
                loaded = load_eval_data(ext_cfg)
                items = self._convert_dataset_format(loaded)
                source_name = ext_cfg.get("sub_dataset") or os.path.splitext(os.path.basename(config_path))[0]
                contexts = [item.get("context", "") for item in items if item.get("context")]
                text = "\n\n".join(contexts).strip()
                if not text:
                    text = "Out-of-task memo about unrelated subject matter to distract retrieval."
                docs.append({
                    "text": text,
                    "name": source_name
                })
            except Exception as exc:
                logger.warning("Failed to prepare external OOT doc from %s: %s", config_path, exc)
        return docs

    def _load_multiple_external_oot_items(self, config_paths):
        """Load contexts from multiple dataset configs to use as OOT candidates."""
        items = []
        for path in config_paths:
            items.extend(self._load_external_oot_items(path))
        return items

    def _rewrite_chunk_with_llm(self, chunk, biased_cfg):
        """Rewrite a chunk by randomly dropping sentences and paraphrasing via LLM."""
        prompt = (
            "You should do 2 things.First, you are required to split the paragraph into individual sentences, "
            "then randomly delete 20% of those sentences. Remember that this process must be random. "
            "Second, You are required to paraphrase following conversation with different detailes and slightly "
            "different narrative. The paraphrased text should be replaceable of the orginal given text. You can "
            "change the entity, number or other details. Only output the paraphrased text and nothing else.Keep the the format consistent"
        )

        model = biased_cfg.get("model", "gpt-4o-mini")
        temperature = biased_cfg.get("temperature", 0.7)
        max_tokens = biased_cfg.get("max_tokens")
        
        print(f"DEBUG: Sending request to OpenRouter... (Chunk len: {len(chunk)})")

        response = self._create_oai_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunk},
            ],
            temperature=temperature,
            max_tokens=max_tokens if max_tokens else None,
        )
        return response.choices[0].message.content

    def _create_oai_client(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_API_BASE")
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)


    
    def _create_query_answer_pairs(self, question_data):
        """
        Create query-answer pairs from question data, handling both single and multiple Q&A.
        
        Args:
            question_data: Dictionary containing questions, answers, and metadata
            
        Returns:
            list: List of (query, answer, qa_pair_id) tuples
        """
        # Extract questions and answers, ensuring they are lists
        questions = self._ensure_list(question_data.get('questions', []))
        answers = self._ensure_list(question_data.get('answers', []))
        
        # Process question-answer pairs based on actual data structure
        if len(questions) > 1 and len(answers) > 1:
            # Multiple questions and answers - process each pair individually
            return [
                self._create_single_qa_pair(question_data, question, answer, i)
                for i, (question, answer) in enumerate(zip(questions, answers))
            ]
        else:
            # Single question or answer set - process as one unit
            return [self._create_single_qa_pair(
                question_data, questions[0] if questions else "", answers, 0
            )]

    def _create_single_qa_pair(self, question_data, question, answer, question_index):
        """
        Create a single query-answer pair with metadata.
        
        Args:
            question_data: Original question data dictionary
            question: The specific question text
            answer: The specific answer text
            question_index: Index of this question in the list
            
        Returns:
            tuple: (formatted_query, answer, qa_pair_id)
        """
        # Create question-specific metadata
        qa_metadata = self._create_qa_metadata(question_data, question, answer, question_index)
        
        # Generate the formatted query using template
        query_template = get_template(self.sub_dataset, 'query', self.agent_name)
        formatted_query = query_template.format(**qa_metadata)
        
        # Get qa_pair_id for this question if available
        qa_pair_id = qa_metadata.get('qa_pair_ids')
        
        return formatted_query, answer, qa_pair_id

    def _ensure_list(self, value):
        """
        Ensure a value is a list, converting single values to single-item lists.
        
        Args:
            value: Value that should be a list
            
        Returns:
            list: The value as a list
        """
        return value if isinstance(value, list) else [value]

    def _create_qa_metadata(self, question_data, question, answer, question_index):
        """
        Create metadata dictionary for a specific question-answer pair.
        
        Args:
            question_data: Original question data dictionary
            question: The specific question text
            answer: The specific answer text
            question_index: Index of this question in the list
            
        Returns:
            dict: Metadata dictionary for template formatting
        """
        # Start with a copy of the original question data
        qa_metadata = dict(question_data)
        
        # Set the specific question and answer
        qa_metadata.update({'question': question, 'answer': answer})
        
        # Process indexed fields
        indexed_fields = ['question_dates', 'question_types', 'question_ids', 'previous_events', 'qa_pair_ids']
        
        for field_name in indexed_fields:
            field_value = self._get_field_value(question_data, field_name, question_index)
            if field_value is not None:
                qa_metadata[field_name] = field_value
        
        # Handle source field specifically (can be nested under metadata)
        if 'source' not in qa_metadata:
            qa_metadata['source'] = question_data.get('metadata', {}).get('source', '')
        
        return qa_metadata

    def _get_field_value(self, question_data, field_name, question_index):
        """
        Get field value from either top level or nested metadata, handling indexing.
        
        Args:
            question_data: Original question data dictionary
            field_name: Name of the field to retrieve
            question_index: Index for list fields
            
        Returns:
            Field value or None if not found
        """
        # Check direct field first, then nested metadata
        field_value = (question_data.get(field_name) or 
                      question_data.get('metadata', {}).get(field_name))
        
        if field_value is None:
            return None
        
        # Use indexed value if it's a list with enough entries, otherwise use the whole value
        return (field_value[question_index] 
                if isinstance(field_value, list) and question_index < len(field_value)
                else field_value)
    #把记忆切成小块
    def get_chunks(self):
        """
        Get text chunks for all contexts, suitable for memory agent processing.
        
        Returns:
            list: List of lists, where each inner list contains text chunks for one context
        """
        
        # all_context_chunks = [
        #     chunk_text_into_sentences(context, chunk_size=self.chunk_size)
        #     for context in self.contexts
        # ]
        # #以上是原代码
        
        all_context_chunks = []

        for context_idx, (context, sections) in enumerate(zip(self.contexts, self.context_sections)):
            chunk_labels = []
            context_chunks = []

            for section in sections:
                raw_chunks = chunk_text_into_sentences(section["text"], chunk_size=self.chunk_size)
                for local_chunk_idx, raw_chunk in enumerate(raw_chunks, start=1):
                    # Label structure helps trace chunk origin and copy index
                    chunk_label = f"{section['label']}|chunk={len(context_chunks) + 1}"
                    labeled_chunk = f"{chunk_label} {raw_chunk}"
                    context_chunks.append(labeled_chunk)
                    chunk_labels.append({
                        "chunk_id": len(context_chunks),
                        "section_label": section["label"],
                        "section_type": section.get("type"),
                        "copy_index": section.get("copy_index"),
                        "noise_index": section.get("noise_index"),
                        "biased_index": section.get("biased_index"),
                        "out_of_task_index": section.get("out_of_task_index"),
                        "source_name": section.get("source_name")
                    })

            self.context_markers[context_idx]["chunk_labels"] = chunk_labels
            all_context_chunks.append(context_chunks)
        #以上是step8的修改
        
        all_context_chunks = self._apply_chunk_keep_ratio(all_context_chunks)
        #以上是0文档修改
        
        # Validate the output structure
        self._validate_chunks_structure(all_context_chunks)
        return all_context_chunks
    
    def _validate_chunk_keep_ratio(self):
        """Validate the chunk keep ratio configuration."""
        if not (0 <= self.chunk_keep_ratio <= 1):
            raise ValueError(
                f"chunk_keep_ratio must be within [0, 1], got {self.chunk_keep_ratio}"
            )
            
    def _apply_chunk_keep_ratio(self, all_context_chunks):
        """Randomly drop chunks to keep only a ratio of chunks per context."""
        if self.chunk_keep_ratio >= 1:
            return all_context_chunks

        sampling_info = []
        filtered_context_chunks = []
        for context_index, chunks in enumerate(all_context_chunks):
            if not chunks:
                if self.chunk_keep_ratio == 0:
                    target_len = max(self.retrieve_num or 0, 1)
                    filtered_context_chunks.append([""] * target_len)
                else:
                    filtered_context_chunks.append(chunks)
                sampling_info.append({
                    "context_index": context_index,
                    "kept_indices": [],
                    "dropped_indices": [],
                    "dropped_chunks": [],
                })
                continue

            keep_count = int(len(chunks) * self.chunk_keep_ratio)
            if self.chunk_keep_ratio > 0 and keep_count == 0:
                keep_count = 1
            if keep_count >= len(chunks):
                filtered_context_chunks.append(chunks)
                sampling_info.append({
                    "context_index": context_index,
                    "kept_indices": list(range(len(chunks))),
                    "dropped_indices": [],
                    "dropped_chunks": [],
                })
                continue

            rng_seed = None if self.chunk_keep_seed is None else self.chunk_keep_seed + context_index
            rng = random.Random(rng_seed)
            keep_indices = sorted(rng.sample(range(len(chunks)), keep_count)) if keep_count else []
            kept_chunks = [chunks[i] for i in keep_indices]
            if self.chunk_keep_ratio == 0:
                target_len = max(self.retrieve_num or 0, 1)
                kept_chunks = kept_chunks + [""] * max(0, target_len - len(kept_chunks))
            filtered_context_chunks.append(kept_chunks)
            dropped_indices = [i for i in range(len(chunks)) if i not in set(keep_indices)]
            dropped_chunks = [chunks[i] for i in dropped_indices]
            sampling_info.append({
                "context_index": context_index,
                "kept_indices": keep_indices,
                "dropped_indices": dropped_indices,
                "dropped_chunks": dropped_chunks,
            })

            logger.info(
                "Applied chunk_keep_ratio=%.2f to context %s: kept %s/%s chunks",
                self.chunk_keep_ratio,
                context_index,
                keep_count,
                len(chunks),
            )

        self.dataset_config["chunk_keep_info"] = {
            "chunk_keep_ratio": self.chunk_keep_ratio,
            "chunk_keep_seed": self.chunk_keep_seed,
            "contexts": sampling_info,
        }
        return filtered_context_chunks



    def _validate_chunks_structure(self, chunks):
        """
        Validate that the chunks have the expected structure.
        
        Args:
            chunks: The chunks structure to validate
            
        Raises:
            AssertionError: If the structure is not as expected
        """
        assert isinstance(chunks, list), "Chunks should be a list"
        assert len(chunks) > 0, "Chunks should not be empty"
        assert isinstance(chunks[0], list), "Each context should have a list of chunks"
        assert isinstance(chunks[0][0], str), "Each chunk should be a string"

    def get_query_and_answers(self):
        """
        Get the processed query-answer pairs for all contexts.
        
        Returns:
            list: List of lists, where each inner list contains (query, answer, qa_pair_id) tuples for one context
        """
        # Validate the output structure
        self._validate_qa_structure(self.query_and_answers)
        return self.query_and_answers

    def _validate_qa_structure(self, query_and_answers):
        """
        Validate that the query-answer structure is correct.
        
        Args:
            query_and_answers: The query-answer structure to validate
            
        Raises:
            AssertionError: If the structure is not as expected
        """
        assert isinstance(query_and_answers, list), "Query-answers should be a list"
        assert len(query_and_answers) > 0, "Query-answers should not be empty"
        assert isinstance(query_and_answers[0], list), "Each context should have a list of QA pairs"
        # Each QA pair should be a tuple of (query, answer, qa_pair_id)
        if len(query_and_answers[0]) > 0:
            assert len(query_and_answers[0][0]) == 3, "Each QA pair should be a tuple of (query, answer, qa_pair_id)"