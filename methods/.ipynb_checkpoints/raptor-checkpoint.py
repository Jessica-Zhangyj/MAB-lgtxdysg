import numpy as np
import pandas as pd
from typing import List, Dict, Any
from langchain.chains.llm import LLMChain
from sklearn.mixture import GaussianMixture
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import AIMessage
from langchain.docstore.document import Document
import matplotlib.pyplot as plt
import logging
import os
import sys
from dotenv import load_dotenv
import time
import re

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Helper functions

def extract_text(item):
    """
    Extract text content from either a string or a LangChain message-like object.
    Works for AIMessage and most BaseMessage-like objects that expose `.content`.
    """
    if isinstance(item, str):
        return item
    if hasattr(item, "content"):
        return item.content if item.content is not None else ""
    return str(item)


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate token count for text. Uses tiktoken if available; otherwise heuristic."""
    import tiktoken  # pip install tiktoken

    # Normalize to string first (avoid noisy warnings and type surprises)
    text = extract_text(text)

    if not isinstance(text, str):
        logging.warning(f"Failed to normalize text to str. Got {type(text)}. Returning 0 tokens.")
        return 0

    if not text:
        return 0

    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using OpenAIEmbeddings."""
    embeddings = OpenAIEmbeddings()
    logging.info(f"Embedding {len(texts)} texts")
    return embeddings.embed_documents([extract_text(text) for text in texts])

def perform_clustering(embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """Perform clustering on embeddings using Gaussian Mixture Model."""
    logging.info(f"Performing clustering with {n_clusters} clusters")
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gm.fit_predict(embeddings)

def summarize_texts(texts: List[str], llm: ChatOpenAI) -> str:
    """Summarize a list of texts using OpenAI. Always returns a plain string."""
    logging.info(f"Summarizing {len(texts)} texts")

    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text concisely:\n\n{text}"
    )
    chain = prompt | llm

    # IMPORTANT: join list into a single string; LCEL expects string for {text}
    joined = "\n\n".join([extract_text(t) for t in texts if extract_text(t).strip()])

    resp = chain.invoke({"text": joined})

    # IMPORTANT: normalize invoke output to str (LCEL often returns AIMessage)
    return extract_text(resp)

def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, level: int):
    """Visualize clusters using PCA."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization - Level {level}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

def build_vectorstore(tree_results: Dict[int, pd.DataFrame], embeddings) -> FAISS:
    """Build a FAISS vectorstore from all texts in the RAPTOR tree."""
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    for level, df in tree_results.items():
        all_texts.extend([str(text) for text in df['text'].tolist()])
        all_embeddings.extend([embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in
                               df['embedding'].tolist()])
        all_metadatas.extend(df['metadata'].tolist())

    logging.info(f"Building vectorstore with {len(all_texts)} texts")
    documents = [Document(page_content=str(text), metadata=metadata)
                 for text, metadata in zip(all_texts, all_metadatas)]
    return FAISS.from_documents(documents, embeddings)

def create_retriever(vectorstore: FAISS, llm: ChatOpenAI) -> ContextualCompressionRetriever:
    """Create a retriever with contextual compression."""
    logging.info("Creating contextual compression retriever")
    base_retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Given the following context and question, extract only the relevant information for answering the question:\n\n"
        "Context: {context}\n"
        "Question: {question}\n\n"
        "Relevant Information:"
    )

    extractor = LLMChainExtractor.from_llm(llm, prompt=prompt)
    return ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever
    )

def check_and_split_cluster(
    texts: List[str],
    embeddings: np.ndarray,
    token_budget: int,
    cluster_labels: np.ndarray,
    cluster_id: int,
    token_model_name: str
) -> np.ndarray:
    """
    Check if a cluster exceeds the token budget. If it does, split the cluster into two equal parts.
    """
    idx = np.where(cluster_labels == cluster_id)[0]
    cluster_texts = [extract_text(texts[i]) for i in idx]
    total_tokens = sum(estimate_tokens(text, model=token_model_name) for text in cluster_texts)

    if total_tokens > token_budget:
        logging.info(f"Cluster {cluster_id} exceeds token budget. Splitting it.")
        half = len(cluster_texts) // 2

        new_labels = cluster_labels.copy()
        base = int(np.max(new_labels)) + 1  # deterministic base label for this split

        for i in range(half):
            new_labels[idx[i]] = base       # first half
        for i in range(half, len(cluster_texts)):
            new_labels[idx[i]] = base + 1   # second half

        return new_labels

    return cluster_labels

class RAPTORMethod:
    def __init__(self, texts: List[str], max_levels: int = 3):
        self.start_time = time.time()
        self.texts = texts
        self.max_levels = max_levels
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o-mini")
        self.tree_results = self.build_raptor_tree()

    def build_raptor_tree(self) -> Dict[int, pd.DataFrame]:
        """Build the RAPTOR tree structure with level metadata and parent-child relationships."""
        results = {}

        # Normalize inputs once
        current_texts = [extract_text(text) for text in self.texts]

        # Give every original node a stable id so child_ids isn't [None, None, ...]
        current_metadata = [
            {"id": f"leaf_0_{i}", "level": 0, "origin": "original", "parent_id": None}
            for i in range(len(current_texts))
        ]

        for level in range(1, self.max_levels + 1):
            logging.info(f"Processing level {level}")

            embeddings = embed_texts(current_texts)
            n_clusters = min(10, len(current_texts) // 2)
            cluster_labels = perform_clustering(np.array(embeddings), n_clusters)

            # Check and split clusters if needed
            token_model_name = "gpt-4o-mini"
            for cluster_id in np.unique(cluster_labels):
                cluster_labels = check_and_split_cluster(
                    current_texts,
                    np.array(embeddings),
                    128000,  # 128k token budget
                    cluster_labels,
                    cluster_id,
                    token_model_name
                )

            df = pd.DataFrame({
                'text': current_texts,
                'embedding': embeddings,
                'cluster': cluster_labels,
                'metadata': current_metadata
            })

            results[level - 1] = df

            summaries = []
            new_metadata = []
            for cluster in df['cluster'].unique():
                cluster_docs = df[df['cluster'] == cluster]
                cluster_texts = cluster_docs['text'].tolist()
                cluster_metadata = cluster_docs['metadata'].tolist()

                summary = summarize_texts(cluster_texts, self.llm)  # guaranteed str now
                summaries.append(summary)

                new_metadata.append({
                    "level": level,
                    "origin": f"summary_of_cluster_{cluster}_level_{level - 1}",
                    "child_ids": [meta.get('id') for meta in cluster_metadata],
                    "id": f"summary_{level}_{cluster}"
                })

            current_texts = summaries
            current_metadata = new_metadata

            if len(current_texts) <= 1:
                results[level] = pd.DataFrame({
                    'text': current_texts,
                    'embedding': embed_texts(current_texts),
                    'cluster': [0],
                    'metadata': current_metadata
                })
                logging.info(f"Stopping at level {level} as we have only one summary")
                break

        return results

    def run(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Run the RAPTOR query pipeline."""
        vectorstore = build_vectorstore(self.tree_results, self.embeddings)
        retriever = create_retriever(vectorstore, self.llm)

        logging.info(f"Processing query: {query}")
        relevant_docs = retriever.get_relevant_documents(query)

        doc_details = [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs]

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = ChatPromptTemplate.from_template(
            "Given the following context, please answer the question:\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
        )
        memory_construction_time = time.time() - self.start_time
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=query)
        query_time_len = time.time() - self.start_time - memory_construction_time

        return {
            "query": query,
            "retrieved_documents": doc_details,
            "context_used": context,
            "answer": answer,
            "model_used": self.llm.model_name,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
        }


# Argument Parsing and Validation
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAPTORMethod")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to process.")
    parser.add_argument("--query", type=str, default="What is the greenhouse effect?",
                        help="Query to test the retriever (default: 'What is the main topic of the document?').")
    parser.add_argument('--max_levels', type=int, default=3, help="Max levels for RAPTOR tree")
    return parser.parse_args()

# Main Execution
if __name__ == "__main__":
    pass
