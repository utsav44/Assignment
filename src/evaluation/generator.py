import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import ManualChunker

logger = logging.getLogger(__name__)


GENERATE_QA_PROMPT = """
You are an expert at creating evaluation datasets for Car Manual RAG systems.
The manual belongs to the car model: "{car_model}".

Context:
"{context}"

Task: Generate 1 specific question and answer pair based ONLY on the context above.
The question MUST explicitly mention the car model "{car_model}" (e.g., "How do I adjust the seat in {car_model}?", "What does this light mean in {car_model}?").
The answer must be found strictly in the text.

Output format (JSON only):
{{
    "question": "The generated question including the car model",
    "answer": "The answer based on text",
    "context": "The exact substring from the text that contains the answer"
}}
"""


class TestSetGenerator:
    def __init__(self, config, llm_client):
        self.config = config
        self.client = llm_client
        self.parser = PDFParser()
        # Use a larger chunk size for generation to give the LLM enough context
        self.chunker = ManualChunker(chunk_size=1000, chunk_overlap=0)

    def _get_car_model_from_metadata(self, pdf_filename: str) -> str:
        metadata_path = self.config.METADATA_DB
        default_model = "the vehicle"

        if not metadata_path.exists():
            logger.warning(f"Metadata file not found at {metadata_path}. Using default.")
            return default_model

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)

            manuals = data.get("manuals", [])

            # If manuals is a dict, iterate values; if list, iterate items
            iterator = manuals.values() if isinstance(manuals, dict) else manuals

            for manual in iterator:
                if manual.get("filename") == pdf_filename:
                    brand = manual.get("brand", "")
                    model = manual.get("model", "")
                    return f"{brand} {model}".strip()

            logger.warning(f"Filename {pdf_filename} not found in metadata. Using default.")
            return default_model

        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
            return default_model

    def generate_test_set(self, pdf_path: Path, num_questions: int = 20):
        logger.info(f"Generating {num_questions} QA pairs from {pdf_path.name}...")

        # 1. Fetch Car/Model Name
        car_model_name = self._get_car_model_from_metadata(pdf_path.name)
        logger.info(f"Context for generation: {car_model_name}")

        # 2. Extract Text
        pages = self.parser.extract_text_by_page(pdf_path)
        chunks = self.chunker.chunk_pages(pages, {})

        # 3. Select random chunks to generate questions from
        import random
        if len(chunks) > num_questions:
            selected_chunks = random.sample(chunks, num_questions)
        else:
            selected_chunks = chunks

        test_set = []

        for chunk in tqdm(selected_chunks, desc="Generating QA Pairs"):
            try:
                # Format prompt with the specific car model
                prompt = GENERATE_QA_PROMPT.format(
                    context=chunk['text'],
                    car_model=car_model_name
                )

                response = self.client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )

                content = response.choices[0].message.content

                # Cleanup JSON markdown if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content.strip())

                # Add metadata for traceability
                data['source_page'] = chunk['metadata'].get('page')
                data['content_type'] = chunk['metadata'].get('content_type', 'text')
                data['car_model'] = car_model_name

                test_set.append(data)

            except Exception as e:
                logger.warning(f"Failed to generate QA for chunk: {e}")
                continue

        # Save to file
        output_path = self.config.DATA_DIR / "evaluation_dataset.json"
        with open(output_path, "w") as f:
            json.dump(test_set, f, indent=2)

        logger.info(f"Saved {len(test_set)} QA pairs to {output_path}")
        return test_set