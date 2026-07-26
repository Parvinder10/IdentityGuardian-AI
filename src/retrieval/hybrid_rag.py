import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

class CrossEncoderRanker(nn.Module):
    """
    PyTorch-native neural cross-encoder for re-ranking retrieved document chunks.
    Computes a relevance score by projecting concatenated query and chunk embeddings.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, query_emb: torch.Tensor, chunk_emb: torch.Tensor) -> torch.Tensor:
        # Concatenate query and chunk embeddings along feature dimension
        combined = torch.cat([query_emb, chunk_emb], dim=-1)
        return self.fc(combined)


class IdentityHybridRetriever:
    """
    Retrieves and links matching identity records from a secure database.
    Integrates dense-sparse matching and layout knowledge graph Entity Linking.
    """
    def __init__(self, chunks: List[Dict[str, Any]], embedding_dim: int = 128):
        self.chunks = chunks
        self.embedding_dim = embedding_dim
        
        tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        
        # Identity projection embeddings
        self.chunk_embeddings = np.array([self._get_embedding(c["text"]) for c in chunks])
        
        # Knowledge Graph mapping connections between verified fields
        self.kg = nx.DiGraph()
        self._build_identity_graph()
        
        self.re_ranker = CrossEncoderRanker(embedding_dim=embedding_dim)

    def _get_embedding(self, text: str) -> np.ndarray:
        char_sum = sum(ord(c) for c in text)
        np.random.seed(char_sum % 10000)
        emb = np.random.randn(self.embedding_dim)
        return emb / np.linalg.norm(emb)

    def _build_identity_graph(self):
        """Maps relations between identity nodes (e.g. matching name to photo and numbers)."""
        for idx, chunk in enumerate(self.chunks):
            self.kg.add_node(idx, text=chunk["text"], label=chunk["label"])
            
        # Link name blocks to ID numbers and photos
        name_indices = [i for i, c in enumerate(self.chunks) if c["label"] == "Name"]
        photo_indices = [i for i, c in enumerate(self.chunks) if c["label"] == "Profile_Photo"]
        id_indices = [i for i, c in enumerate(self.chunks) if c["label"] == "ID_Number"]
        
        for n_idx in name_indices:
            for p_idx in photo_indices:
                self.kg.add_edge(n_idx, p_idx, relation="owns_photo")
            for id_idx in id_indices:
                self.kg.add_edge(n_idx, id_idx, relation="owns_id_number")

    def retrieve_identity(self, query: str, top_k: int = 2, alpha: float = 0.5) -> List[Tuple[int, float]]:
        if not self.chunks:
            return []
            
        # BM25
        query_tokens = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        if bm25_scores.max() != bm25_scores.min():
            bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_norm = bm25_scores
            
        # Dense
        query_emb = self._get_embedding(query)
        dense_scores = np.dot(self.chunk_embeddings, query_emb)
        if dense_scores.max() != dense_scores.min():
            dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        else:
            dense_norm = dense_scores
            
        hybrid_scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm
        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [(int(idx), float(hybrid_scores[idx])) for idx in ranked_indices]

    def re_rank_candidates(self, query: str, candidate_indices: List[int]) -> List[Tuple[int, float]]:
        if not candidate_indices:
            return []
        query_emb = torch.tensor(self._get_embedding(query), dtype=torch.float32).unsqueeze(0).repeat(len(candidate_indices), 1)
        chunk_embs = torch.tensor(self.chunk_embeddings[candidate_indices], dtype=torch.float32)
        
        self.re_ranker.eval()
        with torch.no_grad():
            scores = self.re_ranker(query_emb, chunk_embs).squeeze(-1).tolist()
            
        return sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)

    def traverse_identity_links(self, start_idx: int) -> List[Dict[str, Any]]:
        results = []
        if start_idx not in self.kg:
            return results
        for neighbor in self.kg.neighbors(start_idx):
            relation = self.kg.get_edge_data(start_idx, neighbor)["relation"]
            results.append({
                "target_idx": neighbor,
                "text": self.kg.nodes[neighbor]["text"],
                "relation": relation
            })
        return results
