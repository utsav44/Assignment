from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-np.array(x)))


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", normalize_scores: bool = True):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name
        self.normalize_scores = normalize_scores
        logger.info(f"Reranker loaded. Normalize scores: {normalize_scores}")

    def rerank(
        self,
        query: str,
        texts: List[str],
        metadatas: List[Dict],
        original_scores: List[float],
        top_k: int = 5
    ) -> Tuple[List[str], List[Dict], List[float]]:
        if not texts:
            return [], [], []

        # Create query-document pairs
        pairs = [[query, text] for text in texts]

        # Get cross-encoder scores (raw logits)
        raw_scores = self.model.predict(pairs)

        # Normalize to 0-1 range for readability
        if self.normalize_scores:
            scores = sigmoid(raw_scores)
        else:
            scores = raw_scores

        # Create list with all info for sorting
        scored_results = []
        for i, (score, raw, text, meta, orig) in enumerate(
            zip(scores, raw_scores, texts, metadatas, original_scores)
        ):
            scored_results.append({
                'score': score,
                'raw_score': raw,
                'text': text,
                'metadata': meta,
                'original_score': orig,
                'original_rank': i + 1
            })

        # Sort by normalized score descending
        scored_results.sort(key=lambda x: x['score'], reverse=True)

        # Take top_k
        top_results = scored_results[:top_k]

        # Extract results
        reranked_texts = [r['text'] for r in top_results]
        reranked_metadatas = [r['metadata'] for r in top_results]
        reranked_scores = [float(r['score']) for r in top_results]

        # Log reranking effect
        logger.info(f"Reranked {len(texts)} candidates → top {top_k}")
        logger.info(f"Score range: {min(scores):.3f} to {max(scores):.3f}")

        for i, r in enumerate(top_results[:5]):
            page = r['metadata'].get('page', '?')
            orig_rank = r['original_rank']
            rank_change = orig_rank - (i + 1)
            arrow = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "="
            logger.info(
                f"  Rank {i+1}: Page {page} | "
                f"Score: {r['score']:.3f} (raw: {r['raw_score']:.2f}) | "
                f"Was #{orig_rank} ({arrow}{abs(rank_change)})"
            )

        return reranked_texts, reranked_metadatas, reranked_scores

    def rerank_with_details(
        self,
        query: str,
        texts: List[str],
        metadatas: List[Dict],
        original_scores: List[float]
    ) -> List[Dict]:
        if not texts:
            return []

        pairs = [[query, text] for text in texts]
        raw_scores = self.model.predict(pairs)

        if self.normalize_scores:
            norm_scores = sigmoid(raw_scores)
        else:
            norm_scores = raw_scores

        results = []
        for i, (text, meta, orig_score, norm_score, raw) in enumerate(
            zip(texts, metadatas, original_scores, norm_scores, raw_scores)
        ):
            results.append({
                "text": text,
                "metadata": meta,
                "original_score": float(orig_score),
                "rerank_score": float(norm_score),
                "raw_score": float(raw),
                "original_rank": i + 1
            })

        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Add new rank and change
        for i, r in enumerate(results):
            r["rerank_rank"] = i + 1
            r["rank_change"] = r["original_rank"] - r["rerank_rank"]

        return results


class HybridReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        keyword_boost: float = 0.2
    ):
        self.cross_encoder = CrossEncoderReranker(model_name, normalize_scores=True)
        self.keyword_boost = keyword_boost

    def rerank(
        self,
        query: str,
        texts: List[str],
        metadatas: List[Dict],
        original_scores: List[float],
        top_k: int = 5
    ) -> Tuple[List[str], List[Dict], List[float]]:
        if not texts:
            return [], [], []

        # Get cross-encoder scores
        pairs = [[query, text] for text in texts]
        raw_scores = self.cross_encoder.model.predict(pairs)
        ce_scores = sigmoid(raw_scores)

        # Get keyword scores
        keywords = self._extract_keywords(query)
        kw_scores = [self._keyword_score(text, keywords) for text in texts]

        # Combine scores
        combined_scores = [
            (1 - self.keyword_boost) * ce + self.keyword_boost * kw
            for ce, kw in zip(ce_scores, kw_scores)
        ]

        # Sort and return
        scored_results = list(zip(combined_scores, texts, metadatas, ce_scores, kw_scores))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        top_results = scored_results[:top_k]

        # Log
        logger.info(f"Hybrid rerank: {len(texts)} → {top_k}")
        for i, (score, text, meta, ce, kw) in enumerate(top_results[:3]):
            page = meta.get('page', '?')
            logger.info(f"  Rank {i+1}: Page {page} | Combined: {score:.3f} (CE: {ce:.3f}, KW: {kw:.3f})")

        return (
            [r[1] for r in top_results],
            [r[2] for r in top_results],
            [r[0] for r in top_results]
        )

    def _extract_keywords(self, query: str) -> List[str]:

        stopwords = {
            'how', 'what', 'which', 'when', 'where', 'why', 'is', 'are', 'the',
            'a', 'an', 'in', 'on', 'for', 'to', 'of', 'with', 'my', 'i', 'should',
            'can', 'do', 'does', 'use', 'used', 'need', 'want', 'tell', 'me'
        }
        words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _keyword_score(self, text: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords)


class KeywordBoostReranker:
    def __init__(self, keyword_weight: float = 0.3):
        self.keyword_weight = keyword_weight

    def rerank(
        self,
        query: str,
        texts: List[str],
        metadatas: List[Dict],
        original_scores: List[float],
        top_k: int = 5
    ) -> Tuple[List[str], List[Dict], List[float]]:
        if not texts:
            return [], [], []

        keywords = self._extract_keywords(query)

        combined_scores = []
        for text, orig_score in zip(texts, original_scores):
            keyword_score = self._keyword_match_score(text, keywords)
            combined = (1 - self.keyword_weight) * orig_score + self.keyword_weight * keyword_score
            combined_scores.append(combined)

        scored_results = list(zip(combined_scores, texts, metadatas))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        top_results = scored_results[:top_k]

        return (
            [r[1] for r in top_results],
            [r[2] for r in top_results],
            [r[0] for r in top_results]
        )

    def _extract_keywords(self, query: str) -> List[str]:
        stopwords = {
            'how', 'what', 'which', 'when', 'where', 'why', 'is', 'are', 'the',
            'a', 'an', 'in', 'on', 'for', 'with', 'my', 'i', 'should', 'can', 'do'
        }
        words = query.lower().split()
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _keyword_match_score(self, text: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords)