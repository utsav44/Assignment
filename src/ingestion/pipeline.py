from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
import uuid
import shutil
import numpy as np
from config import Config
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import ManualChunker
from src.ingestion.table_extractor import TableExtractor, format_table_for_chunking
from src.ingestion.image_extractor import ImageExtractor
from src.ingestion.image_embedder import ImageEmbedder
from src.embedding.embedder import BGEEmbedder

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, config: Config, embedder: BGEEmbedder, vector_store, llm_client=None):
        self.config = config
        self.parser = PDFParser()
        self.chunker = ManualChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.table_extractor = TableExtractor(
            min_accuracy=getattr(config, 'TABLE_MIN_ACCURACY', 60.0),
            min_row_length=getattr(config, 'MIN_ROW_LENGTH', 30)
        )
        self.image_extractor = ImageExtractor(
            min_image_size=getattr(config, 'MIN_IMAGE_SIZE', 5000),
            min_width=getattr(config, 'MIN_WIDTH', 150),
            min_height=getattr(config, 'MIN_HEIGHT', 150),
            max_aspect_ratio=getattr(config, 'MAX_ASPECT_RATIO', 4.0),
            min_variance=getattr(config, 'MIN_VARIANCE_THRESHOLD', 250.0),
            header_footer_cutoff=getattr(config, 'HEADER_FOOTER_CUTOFF', 0.10),
        )
        self.image_embedder = ImageEmbedder(
            embedder=embedder,
            llm_client=llm_client,
            llm_model=getattr(config, 'OPENAI_MODEL', 'gpt-3.5-turbo')
        )
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_client = llm_client

    def process_manual(
        self,
        pdf_path: Path,
        model: str,
        brand: Optional[str] = None,
        year: Optional[str] = None,
        extract_tables: bool = True,
        extract_images: bool = True
    ) -> Tuple[str, Dict, str]:
        display_name = self._create_display_name(brand, model, year)
        car_model_key = self._create_car_model_key(brand, model, year)

        logger.info(f"Processing: {display_name}")

        base_metadata = {
            "brand": brand,
            "model": model,
            "year": year,
            "source_file": pdf_path.name,
            "car_model_key": car_model_key
        }

        stats = {
            "text_chunks": 0,
            "table_chunks": 0,
            "image_chunks": 0,
            "total_chunks": 0
        }

        all_chunks = []

        # Step 1: Extract and chunk regular text
        logger.info("Extracting text from PDF...")
        pages = self.parser.extract_text_by_page(pdf_path)

        if pages:
            logger.info(f"Chunking {len(pages)} pages of text...")
            text_chunks = self.chunker.chunk_pages(pages, base_metadata)
            for chunk in text_chunks:
                chunk['metadata']['content_type'] = 'text'
            all_chunks.extend(text_chunks)
            stats['text_chunks'] = len(text_chunks)
            logger.info(f"Created {len(text_chunks)} text chunks")

        # Step 2: Extract tables
        if extract_tables:
            logger.info("Extracting tables...")
            try:
                tables_by_page = self.table_extractor.extract_tables_by_page(pdf_path)
                table_chunks = []

                for page_num, tables in tables_by_page.items():
                    for table in tables:
                        table['page'] = page_num
                        chunks = format_table_for_chunking(table,self.config.TABLE_CHUNK_SIZE,self.config.TABLE_CHUNK_OVERLAP)
                        for chunk in chunks:
                            chunk['metadata'].update(base_metadata)
                        table_chunks.extend(chunks)

                all_chunks.extend(table_chunks)
                stats['table_chunks'] = len(table_chunks)
                logger.info(f"Created {len(table_chunks)} table chunks")

            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")

        # Step 3: Extract and process images
        if extract_images:
            logger.info("Extracting images...")
            try:
                images = self.image_extractor.extract_images_from_pdf(pdf_path)

                if images:
                    logger.info(f"Processing {len(images)} images...")

                    # Create temp directory for images
                    temp_dir = self.config.DATA_DIR / f"temp_images_{uuid.uuid4().hex[:8]}"
                    temp_dir.mkdir(exist_ok=True)

                    # Save images temporarily
                    for img in images:
                        img_path = temp_dir / f"{img['image_id']}.png"
                        with open(img_path, 'wb') as f:
                            f.write(img['image_data'])
                        img['temp_path'] = img_path

                    # Generate tags and embeddings
                    processed_images = self.image_embedder.process_image_batch(images)

                    # Create final image directory
                    images_dir = self.config.DATA_DIR / "images" / car_model_key
                    images_dir.mkdir(parents=True, exist_ok=True)

                    # Move images to final location and create chunks
                    image_chunks = []
                    for img in processed_images:
                        # Move image to final location
                        final_path = images_dir / f"{img['tag'].replace(' ', '_')}_{img['image_id']}.png"
                        shutil.move(img['temp_path'], final_path)

                        # Create chunk
                        chunk_text = f"Image: {img['tag']}. Context: {img['text_context'][:300]}"
                        image_chunks.append({
                            'text': chunk_text,
                            'embedding': img['embedding'],
                            'metadata': {
                                **base_metadata,
                                'content_type': 'image',
                                'page': img['page_num'],
                                'image_id': img['image_id'],
                                'image_path': str(final_path),
                                'tag': img['tag'],
                                'full_context': img['text_context']
                            }
                        })

                    all_chunks.extend(image_chunks)
                    stats['image_chunks'] = len(image_chunks)
                    logger.info(f"Created {len(image_chunks)} image chunks")

                    # Cleanup temp directory
                    shutil.rmtree(temp_dir, ignore_errors=True)

            except Exception as e:
                logger.warning(f"Image extraction failed: {e}")

        if not all_chunks:
            raise ValueError("No content extracted from document")

        stats['total_chunks'] = len(all_chunks)

        # Step 4: Generate embeddings for text and table chunks
        logger.info(f"Generating embeddings for {stats['text_chunks'] + stats['table_chunks']} text/table chunks...")

        # Separate chunks that need embeddings from those that already have them
        chunks_need_embedding = [c for c in all_chunks if 'embedding' not in c]
        chunks_have_embedding = [c for c in all_chunks if 'embedding' in c]

        if chunks_need_embedding:
            texts = [chunk["text"] for chunk in chunks_need_embedding]
            embeddings = self.embedder.embed_documents(texts, show_progress=True)

            for chunk, embedding in zip(chunks_need_embedding, embeddings):
                chunk['embedding'] = embedding

        # Step 5: Store in vector database
        logger.info("Storing in vector database...")

        # Combine all chunks
        all_processed_chunks = chunks_need_embedding + chunks_have_embedding

        embeddings = np.array([c['embedding'] for c in all_processed_chunks])
        texts = [c['text'] for c in all_processed_chunks]
        metadatas = [c['metadata'] for c in all_processed_chunks]

        self.vector_store.add_documents(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
            car_model_key=car_model_key
        )

        logger.info(
            f"Successfully stored {stats['total_chunks']} chunks "
            f"({stats['text_chunks']} text, {stats['table_chunks']} table, "
            f"{stats['image_chunks']} image) for {car_model_key}"
        )

        return car_model_key, stats, display_name

    def _create_display_name(self, brand: Optional[str], model: str,
                             year: Optional[str]) -> str:
        parts = []
        if brand:
            parts.append(brand)
        parts.append(model)
        if year:
            parts.append(f"({year})")
        return " ".join(parts)

    def _create_car_model_key(self, brand: Optional[str], model: str,
                              year: Optional[str]) -> str:
        brand_part = brand.lower().replace(" ", "_") if brand else "unknown"
        model_part = model.lower().replace(" ", "_")
        year_part = year if year else "unknown"
        return f"{brand_part}_{model_part}_{year_part}"


