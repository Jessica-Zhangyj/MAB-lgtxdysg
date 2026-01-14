
"""
Inspect chunk contents and labels for a given context so you can verify which
chunks (e.g., `[Original]`, `[Repeat 3]`, biased, or out-of-task) are present.

Usage example:
    python tools/inspect_retrieved_chunks.py \
        --dataset_config configs/data_conf/Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k_degration.yaml \
        --chunk_ids 277,1513,1101,689 \
        --context_id 0 \
        --chunk_size_override 512

Notes:
- The script rebuilds the conversation/chunks using `ConversationCreator`, so
  it will re-load the dataset with the provided config and chunk size.
- Chunk ids are 1-based and should match the ids logged in the retrieval trace.
"""

import argparse
import os
from typing import Dict, List, Optional

import yaml

from conversation_creator import ConversationCreator


def _load_yaml_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_chunk_ids(raw: str) -> List[int]:
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            raise ValueError(f"Chunk id must be integer, got '{part}'")
    return ids


def _build_agent_config(agent_name: str) -> Dict:
    """
    Build a minimal agent config so ConversationCreator can determine chunk size.
    Only agent_name is used; chunk size is taken from dataset_config unless
    overridden via CLI.
    """
    return {
        "agent_name": agent_name,
        "agent_chunk_size": None,
    }


def _get_chunk_label(marker: Dict, chunk_id: int) -> Optional[Dict]:
    """Return the label dict for a given 1-based chunk id."""
    for entry in marker.get("chunk_labels", []):
        if entry.get("chunk_id") == chunk_id:
            return entry
    return None


def _print_chunk_info(chunk_text: str, label: Dict, chunk_id: int):
    preview = " ".join(chunk_text.split())[:240]
    print(f"\n### Chunk {chunk_id}")
    if label:
        print(
            f"label={label.get('section_label')} | type={label.get('section_type')} | "
            f"repeat={label.get('copy_index')} | biased={label.get('biased_index')} | "
            f"oot={label.get('out_of_task_index')}"
        )
    else:
        print("label: [unknown]")
    print(f"text preview: {preview}")


def main():
    parser = argparse.ArgumentParser(description="Inspect chunk contents by chunk id.")
    parser.add_argument(
        "--dataset_config",
        required=True,
        help="Path to the dataset YAML used for the run.",
    )
    parser.add_argument(
        "--chunk_ids",
        required=True,
        help="Comma-separated list of 1-based chunk ids to inspect (e.g., 277,1513,1101).",
    )
    parser.add_argument(
        "--context_id",
        type=int,
        default=0,
        help="Context index to inspect (default: 0).",
    )
    parser.add_argument(
        "--chunk_size_override",
        type=int,
        default=None,
        help="Override chunk_size used when recreating chunks (e.g., 512).",
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default="rag_bm25",
        help="Agent name string; only used for chunk size handling (default: rag_bm25).",
    )
    args = parser.parse_args()

    dataset_config = _load_yaml_config(args.dataset_config)
    if args.chunk_size_override:
        dataset_config["chunk_size"] = args.chunk_size_override
        print(f"Using overridden chunk_size: {dataset_config['chunk_size']}")

    agent_config = _build_agent_config(args.agent_name)

    creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = creator.get_chunks()

    if args.context_id >= len(all_chunks):
        raise IndexError(
            f"context_id {args.context_id} out of range (total contexts: {len(all_chunks)})"
        )

    context_chunks = all_chunks[args.context_id]
    marker = creator.context_markers[args.context_id] if args.context_id < len(creator.context_markers) else {}
    chunk_ids = _parse_chunk_ids(args.chunk_ids)

    print(
        f"\nInspecting context {args.context_id}: total chunks={len(context_chunks)}, "
        f"chunk_ids={chunk_ids}"
    )

    for chunk_id in chunk_ids:
        if chunk_id <= 0 or chunk_id > len(context_chunks):
            print(f"\n### Chunk {chunk_id}\nchunk id out of range for this context.")
            continue
        chunk_text = context_chunks[chunk_id - 1]
        label = _get_chunk_label(marker, chunk_id)
        _print_chunk_info(chunk_text, label, chunk_id)


if __name__ == "__main__":
    # Avoid OpenAI network calls inside ConversationCreator by disabling attack rewrites.
    os.environ.setdefault("OPENAI_API_KEY", "placeholder")
    main()