import os
import openai
from pathlib import Path
import logging
import json

from config import Config
from src.embedding.embedder import BGEEmbedder
from src.vectorstore.vector_db import FAISSVectorStore
from src.utils.metadata import MetadataManager
from src.query.retriever import ManualRAG
from src.ingestion.pipeline import IngestionPipeline

# Import evaluation components
from src.evaluation.generator import TestSetGenerator
from src.evaluation.evaluator import RAGEvaluator

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config = Config()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    client = openai.OpenAI(
        api_key=api_key
    )

    logger.info("Loading RAG System...")
    embedder = BGEEmbedder(model_name=config.EMBEDDING_MODEL)
    vector_store = FAISSVectorStore(index_path=config.VECTORSTORE_DIR / "index.faiss")
    metadata = MetadataManager(config.METADATA_DB, embedder)

    if vector_store.total_documents == 0:
        logger.warning("Vector store is empty! Please run the ingestion app first or point to a PDF.")
        return


    rag = ManualRAG(vector_store, embedder, metadata, client, config)

    # 3. Generate or Load Test Data
    generator = TestSetGenerator(config, client)
    dataset_path = config.DATA_DIR / "evaluation_dataset.json"
    
    if dataset_path.exists():
        logger.info(f"Loading existing test set from {dataset_path}")
        with open(dataset_path, 'r') as f:
            test_set = json.load(f)
    else:

        pdf_files = list(config.MANUALS_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.error("No PDFs found...")
            return

        target_pdf = pdf_files[0]
        logger.info(f"Generating new test set from {target_pdf.name}")
        test_set = generator.generate_test_set(target_pdf, num_questions=100)

    evaluator = RAGEvaluator(rag)
    results_df = evaluator.evaluate_retrieval(test_set, top_k=5)

    evaluator.print_report(results_df)
    
    # Save detailed results
    results_df.to_csv("evaluation_results.csv", index=False)
    logger.info("Detailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    main()