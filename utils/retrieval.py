import os
import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from operator import itemgetter
from utils.pinecone_manager import PineconeManager

class HybridRetriever:
    """
    A retrieval system that uses Pinecone vector search and reranking.
    """
    
    def __init__(self, pinecone_manager: PineconeManager, index_name: str):
        self.pinecone_manager = pinecone_manager
        self.index_name = index_name
        # BM25 and all_docs are not supported with Pinecone
        self.all_docs = None
        self.bm25_docs = []
        self.bm25_retriever = None
        
    def get_relevant_documents(self, query: str, k: int = 10, hybrid_alpha: float = 0.5, 
                              rerank: bool = True, diversity_bias: float = 0.3) -> List[Document]:
        """
        Retrieve documents using Pinecone vector search and reranking.
        """
        # Vector search (semantic similarity)
        vector_docs = self.pinecone_manager.similarity_search(self.index_name, query, k=k)
        combined_docs = vector_docs[:k]
        # Apply reranking if enabled
        if rerank and combined_docs:
            return self.rerank_with_gemini(query, combined_docs)
        return combined_docs
    
    def _get_doc_id(self, doc: Document) -> str:
        meta = doc.metadata
        content_hash = hash(doc.page_content[:100])
        source = meta.get("source", "unknown")
        page = meta.get("page", "0")
        chunk_id = meta.get("chunk_id", "0")
        return f"{source}_{page}_{chunk_id}_{content_hash}"
    
    def rerank_with_gemini(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs or len(docs) <= 1:
            return docs
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            formatted_chunks = "\n\n".join([
                f"[{i+1}] {c.page_content.strip()}"
                for i, c in enumerate(docs)
            ])
            prompt = f"""
You are evaluating document chunks for relevance to a user query.

User Query: {query}

Below are document chunks. Score each on a 0-10 scale for how relevant, comprehensive, and accurate it is for answering this query.
Prioritize chunks that directly address the query with concrete information over vague or tangential mentions.

Document Chunks:
{formatted_chunks}

Return a JSON object with chunk IDs and scores, like:
{{
    "ranked_ids": [3, 1, 5, 2, 4]
}}

Only include the JSON. No explanations or other text.
"""
            response = model.generate_content(prompt)
            result = response.text.strip()
            import json
            import re
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    ranking = json.loads(json_str)
                    if "ranked_ids" in ranking:
                        indices = [int(i) - 1 for i in ranking["ranked_ids"]]
                        valid_indices = [i for i in indices if 0 <= i < len(docs)]
                        remaining = [i for i in range(len(docs)) if i not in valid_indices]
                        all_indices = valid_indices + remaining
                        return [docs[i] for i in all_indices]
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Error in reranking: {e}")
        return docs
    
    def query_expansion(self, query: str, num_variants: int = 3) -> List[str]:
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            prompt = f"""
Given this search query: "{query}"

Generate {num_variants} alternative versions that:
1. Rephrase with different terminology
2. Expand with related concepts
3. Add technical specificity

Return ONLY a JSON array of strings with the alternative queries. Nothing else.
Example: ["first alternative", "second alternative", "third alternative"]
"""
            response = model.generate_content(prompt)
            result = response.text.strip()
            import json
            import re
            json_pattern = r'\[.*\]'
            json_match = re.search(json_pattern, result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    expanded_queries = json.loads(json_str)
                    if isinstance(expanded_queries, list) and expanded_queries:
                        return [query] + expanded_queries
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Error in query expansion: {e}")
        return [query]
    
    def retrieval_with_expansion(self, query: str, k: int = 10) -> List[Document]:
        expanded_queries = self.query_expansion(query)
        all_docs = []
        doc_dict = {}
        for q in expanded_queries:
            docs = self.get_relevant_documents(q, k=k//2)
            for doc in docs:
                doc_id = self._get_doc_id(doc)
                if doc_id not in doc_dict:
                    doc_dict[doc_id] = doc
                    all_docs.append(doc)
            if len(all_docs) >= k:
                break
        if all_docs:
            return self.rerank_with_gemini(query, all_docs[:k])
        return all_docs[:k] 