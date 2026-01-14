import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Union
from openai import OpenAI, AzureOpenAI
import torch.nn.functional as F
from utils.eval_data_utils import (
    format_chat,
)
import time
import re
import os
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

# Create a custom embedding class for Contriever
class ContrieverEmbeddings(Embeddings):
    def __init__(self, model_name="facebook/contriever"):
        assert "contriever" in model_name, "Model name must contain 'contriever'"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in tqdm(texts, desc="Embedding documents (Contriever)"):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy()[0].tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()


class Qwen3Embedding4BEmbeddings(Embeddings):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B"):
        assert "Qwen3-Embedding-4B" in model_name or "Qwen/Qwen3-Embedding-4B" in model_name, "Model name must be Qwen/Qwen3-Embedding-4B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        
    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        # Batch encode for efficiency if needed; keep simple loop for parity with ContrieverEmbeddings
        for text in tqdm(texts, desc="Embedding documents (Qwen3-Embedding-4B)"):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            embedding = self._mean_pooling(last_hidden, inputs['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy()[0].tolist())
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        embedding = self._mean_pooling(last_hidden, inputs['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()


class NVEmbedV2Embeddings(Embeddings):
    def __init__(self, model_name="nvidia/NV-Embed-v2"):
        assert "NV" in model_name or "nv" in model_name, "Model name should be an NV-Embed variant"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in tqdm(texts, desc="Embedding documents (NV-Embed-v2)"):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            embedding = self._mean_pooling(last_hidden, inputs['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy()[0].tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        embedding = self._mean_pooling(last_hidden, inputs['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()

class  TextRetriever:
    def __init__(self,
                 embedding_model_name: str = "text-embedding-3-large",
                 sub_dataset=None,
                 use_azure: bool = False,
                 azure_endpoint: str = None,
                 azure_api_key: str = None,
                 azure_api_version: str = "2024-02-01",
                 azure_embedding_deployment: str = None):
        
        if use_azure:
            assert azure_endpoint and azure_api_key and azure_embedding_deployment, (
                "Azure configuration missing: require azure_endpoint, azure_api_key, azure_embedding_deployment"
            )
            self.embedding_model = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                azure_deployment=azure_embedding_deployment,
            )
        elif embedding_model_name == "facebook/contriever":
            self.embedding_model = ContrieverEmbeddings(model_name=embedding_model_name)
        elif embedding_model_name == "Qwen/Qwen3-Embedding-4B":
            self.embedding_model = Qwen3Embedding4BEmbeddings(model_name=embedding_model_name)
        elif embedding_model_name in ["nvidia/NV-Embed-v2", "nvidia/NV-Embed-v2-7B", "NV-Embed-v2-7B", "NV-Embed-v2"]:
            self.embedding_model = NVEmbedV2Embeddings(model_name=embedding_model_name)
        else:
            # Default to OpenAI embeddings (supports text-embedding-3-* and others)
            self.embedding_model = OpenAIEmbeddings(
                model=embedding_model_name,
                # dimensions=1024,
            )
        self.sub_dataset = sub_dataset
        self.vectorstore: FAISS = None
        self._current_documents = None
        
    def save_vectorstore(self, path: str) -> None:
        """Persist the current FAISS vectorstore to disk."""
        if not self.vectorstore:
            return
        self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str) -> bool:
        """Load a FAISS vectorstore from disk if available."""
        if not path:
            return False
        if not os.path.isdir(path):
            return False
        self.vectorstore = FAISS.load_local(
            path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        return True
        
    def build_vectorstore(self, documents: List[str]):
        """Build and cache the vector store from documents"""
        # Convert strings to Document objects if needed
        if isinstance(documents[0], str):
            doc_objects = [Document(page_content=doc) for doc in documents]
        else:
            doc_objects = documents
            
        self.vectorstore = FAISS.from_documents(doc_objects, self.embedding_model)
        self._current_documents = documents
        
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant contexts for a query (auto-caches vectorstore)
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve from vector store
        """
        initial_k = top_k
        
        # Perform similarity search to get initial results
        results = self.vectorstore.similarity_search(query, k=initial_k)
        retrieved_docs = [doc.page_content for doc in results]
        
        # Return results (truncated to top_k if needed)
        return retrieved_docs[:top_k]
    



class RAGSystem:
    def __init__(self,
                 retriever,
                 model,
                 temperature,
                 max_tokens,
                 use_azure: bool = False,
                 base_url: str = None, 
                 api_key: str = None,
                 azure_endpoint: str = None,
                 azure_api_key: str = None,
                 azure_api_version: str = "2024-02-01"):
        self.retriever = retriever
        if use_azure:
            assert azure_endpoint and azure_api_key, (
                "Azure configuration missing: require azure_endpoint and azure_api_key"
            )
            self.llm = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
            )
        else:
            self.llm = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer_query(self, query: str, top_k: int, system_message: str) -> Dict[str, Union[str, float]]:
        """Retrieve relevant information and generate an answer"""
        # Retrieve relevant passages
        start_time = time.time()
        match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query =  ''.join(match.groups())
        else:
            match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
            if match:
                retrieval_query =  ''.join(match.groups())
            else:
                retrieval_query = query
        print(f"Retrieve query: {retrieval_query}")
        retrieved_contexts = self.retriever.retrieve(retrieval_query, top_k)
        
        # Format retrieved contexts
        formatted_context = "\n\n".join([f"Passage {i+1}:\n{text}" 
                                       for i, text in enumerate(retrieved_contexts)])
        memory_construction_time = time.time() - start_time
        
        # Generate prompt
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieved_contexts)])
        ask_llm_message=retrieval_memory_string + "\n" + query
        format_message = format_chat(message=ask_llm_message, system_message=system_message)
        
        # Get response from LLM
        response = self.llm.chat.completions.create(
                                model=self.model,
                                messages=format_message,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens
                            )
        query_time_len = time.time() - start_time - memory_construction_time
        
        return {
            "query": query,
            "context_used": formatted_context,
            "answer": response.choices[0].message.content,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
        }