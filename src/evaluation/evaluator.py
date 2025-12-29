import logging
import time
import pandas as pd
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, rag_system):
        self.rag = rag_system

    def evaluate_retrieval(self, test_set: list, top_k: int = 5):
        results = []
        logger.info(f"Evaluating {len(test_set)} questions...")

        for item in tqdm(test_set, desc="Running Eval"):
            query = item['question']
            ground_truth_context = item['context']

            # --- Test 1: Without Reranker (Bi-Encoder Only) ---
            self.rag.use_reranker = False
            start_time = time.time()

            # Note: We use retrieve() to get docs, not generate answer, to save API costs
            docs_base = self.rag.retrieve(query, top_k=top_k)
            time_base = time.time() - start_time

            rank_base = self._get_rank(docs_base, ground_truth_context)
            hit_base = 1 if rank_base > 0 else 0

            # --- Test 2: With Reranker (Cross-Encoder) ---
            self.rag.use_reranker = True
            start_time = time.time()

            docs_rerank = self.rag.retrieve(query, top_k=top_k)
            time_rerank = time.time() - start_time

            rank_rerank = self._get_rank(docs_rerank, ground_truth_context)
            hit_rerank = 1 if rank_rerank > 0 else 0

            # Image Retrieval Check
            # Check if metadata indicates an image content type
            images_found = any(
                doc.get('metadata', {}).get('content_type') == 'image'
                for doc in docs_rerank
            )

            results.append({
                "question": query,
                "hit_base": hit_base,
                "rank_base": rank_base,
                "time_base": time_base,
                "hit_rerank": hit_rerank,
                "rank_rerank": rank_rerank,
                "time_rerank": time_rerank,
                "images_found": images_found
            })

        return pd.DataFrame(results)

    def _get_rank(self, retrieved_docs, ground_truth_substring):

        if not retrieved_docs:
            return 0

        # Clean string for comparison
        clean_truth = " ".join(ground_truth_substring.lower().split()[:20])  # Compare first 20 words

        for i, doc in enumerate(retrieved_docs):
            retrieved_text = doc.get('text', '').lower()

            # Check if specific unique phrase exists in retrieved text
            # OR check simple overlap > 60%
            if clean_truth in retrieved_text or self._calculate_overlap(ground_truth_substring, retrieved_text) > 0.6:
                return i + 1
        return 0

    def _calculate_overlap(self, s1, s2):

        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 or not set2: return 0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def print_report(self, df):

        print("\n" + "=" * 50)
        print(" 📊 RAG EVALUATION REPORT")
        print("=" * 50)

        # Avoid division by zero
        if len(df) == 0:
            print("No results.")
            return

        mrr_base = (1 / df[df['rank_base'] > 0]['rank_base']).sum() / len(df)
        mrr_rerank = (1 / df[df['rank_rerank'] > 0]['rank_rerank']).sum() / len(df)

        hit_rate_base = df['hit_base'].mean()
        hit_rate_rerank = df['hit_rerank'].mean()

        avg_time_base = df['time_base'].mean()
        avg_time_rerank = df['time_rerank'].mean()

        print(f"\n1. RETRIEVAL ACCURACY (Hit Rate @ 5)")
        print(f"   - Without Reranker: {hit_rate_base:.2%}")
        print(f"   - With Reranker:    {hit_rate_rerank:.2%}")
        print(f"   - Improvement:      {(hit_rate_rerank - hit_rate_base):.2%}")

        print(f"\n2. RANKING QUALITY (MRR - Higher is better)")
        print(f"   - Without Reranker: {mrr_base:.3f}")
        print(f"   - With Reranker:    {mrr_rerank:.3f}")

        print(f"\n3. LATENCY (Seconds)")
        print(f"   - Avg Time:         {avg_time_rerank:.3f}s")

        print(f"\n4. MULTIMODAL")
        print(f"   - Queries retrieving images: {df['images_found'].sum()}/{len(df)}")
        print("=" * 50 + "\n")