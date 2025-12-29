from typing import List, Dict, Optional
import logging

from src.query.query_processor import QueryProcessor
from src.query.reranker import CrossEncoderReranker, HybridReranker

logger = logging.getLogger(__name__)


class ManualRAG:
    def __init__(
        self,
        vector_store,
        embedder,
        metadata_manager,
        llm_client,
        config,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base",
        use_hybrid_reranking: bool = True,
        keyword_boost: float = 0.15
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.metadata_manager = metadata_manager
        self.llm = llm_client
        self.config = config

        self.query_processor = QueryProcessor(
            metadata_manager,
            embedder,
            threshold=config.MODEL_MATCH_THRESHOLD
        )

        self.use_reranker = use_reranker
        self.reranker = None

        try:
            if use_hybrid_reranking:
                self.reranker = HybridReranker(
                    model_name=reranker_model,
                    keyword_boost=keyword_boost
                )
                logger.info(f"Hybrid reranker initialized (keyword_boost={keyword_boost})")
            else:
                self.reranker = CrossEncoderReranker(
                    model_name=reranker_model,
                    normalize_scores=True
                )
                logger.info("Cross-encoder reranker initialized")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            self.reranker = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:

        is_valid, _ = self.query_processor.validate_query(query)
        if not is_valid:
            return []

        manual_info, _ = self.query_processor.identify_car_model(query)
        if not manual_info:
            return []

        car_model_key = self.query_processor.get_car_model_key(manual_info)

        initial_k = getattr(self.config, 'RETRIEVAL_CANDIDATES', 30) if (self.use_reranker and self.reranker) else top_k

        query_embedding = self.embedder.embed_query(query)
        texts, metadatas, scores = self.vector_store.search_by_model(
            query_embedding,
            car_model_key,
            k=initial_k
        )

        if not texts:
            return []

        if self.use_reranker and self.reranker:
            texts, metadatas, scores = self.reranker.rerank(
                query, texts, metadatas, scores, top_k=top_k
            )

        results = []
        for text, meta, score in zip(texts, metadatas, scores):
            results.append({
                "text": text,
                "metadata": meta,
                "score": score,
                "manual_info": manual_info
            })

        return results

    def query(self, query: str, top_k: int = 5) -> Dict:


        retrieved_docs = self.retrieve(query, top_k)

        if not retrieved_docs:
            return {
                "status": "not_found",
                "message": "No relevant information found.",
                "answer": None,
                "sources": []
            }

        texts = [doc['text'] for doc in retrieved_docs]
        metadatas = [doc['metadata'] for doc in retrieved_docs]
        scores = [doc['score'] for doc in retrieved_docs]

        manual_info = retrieved_docs[0].get('manual_info', {})

        answer = self._generate_answer(query, texts, metadatas, manual_info)

        formatted_sources = self._format_sources(texts, metadatas, scores)

        return {
            "status": "success",
            "answer": answer,
            "sources": formatted_sources,
            "car_info": manual_info,
            "similarity_score": manual_info.get("similarity_score", 0),
            "chunks_retrieved": len(texts),
            "reranked": self.use_reranker and (self.reranker is not None)
        }

    def _generate_answer(
        self,
        query: str,
        contexts: List[str],
        metadatas: List[Dict],
        manual_info: Dict
    ) -> str:

        context_parts = []
        for i, (ctx, meta) in enumerate(zip(contexts, metadatas), 1):
            page = meta.get("page", "N/A")
            context_parts.append(f"[Source {i}, Page {page}]:\n{ctx}")

        context_str = "\n\n---\n\n".join(context_parts)
        car_name = manual_info.get("display_name", "Unknown Car")

        prompt = f"""You are a helpful car manual assistant for the {car_name}.

Answer the user's question based ONLY on the provided manual excerpts. Follow these rules:
1. Use ONLY information from the provided sources
2. Include citations like [Source 1, Page X] after relevant information
3. If the information is not in the sources, say so clearly
4. Be concise but complete
5. If steps are involved, number them clearly
6. Look at ALL sources - the answer might not be in the first source

User Question: {query}

Manual Excerpts from {car_name}:
{context_str}

Answer:"""

        try:
            response = self.llm.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=getattr(self.config, 'LLM_TEMPERATURE', 0.1),
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_answer(contexts, metadatas)

    def _fallback_answer(
        self,
        contexts: List[str],
        metadatas: List[Dict]
    ) -> str:

        parts = ["Here's the relevant information from the manual:\n"]
        for i, (ctx, meta) in enumerate(zip(contexts, metadatas), 1):
            page = meta.get("page", "N/A")
            parts.append(f"\n**[Source {i}, Page {page}]:**\n{ctx[:500]}...")
        return "\n".join(parts)

    def _format_sources(
        self,
        texts: List[str],
        metadatas: List[Dict],
        scores: List[float]
    ) -> List[Dict]:
        sources = []
        for i, (text, meta, score) in enumerate(zip(texts, metadatas, scores)):
            sources.append({
                "source_number": i + 1,
                "text": text,
                "page": meta.get("page", "N/A"),
                "chunk_id": meta.get("chunk_id", "N/A"),
                "similarity_score": float(score),
                "metadata": meta
            })
        return sources


class SimpleRAG(ManualRAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_answer(
        self,
        query: str,
        contexts: List[str],
        metadatas: List[Dict],
        manual_info: Dict
    ) -> str:
        car_name = manual_info.get("display_name", "Unknown Car")
        parts = [f"**Relevant information from {car_name} manual:**\n"]
        for i, (ctx, meta) in enumerate(zip(contexts, metadatas), 1):
            page = meta.get("page", "N/A")
            parts.append(f"\n---\n**[Source {i}, Page {page}]:**\n{ctx}")
        return "\n".join(parts)