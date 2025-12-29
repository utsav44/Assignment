import camelot
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
import pypdf
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import io
import warnings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pypdf")
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")

logger = logging.getLogger(__name__)

def process_page_worker(pdf_path_str: str, page_num: int, min_accuracy: float) -> List[Dict]:

    page_tables = []
    try:
        # 1. Try Lattice (Tables with lines)
        tables = camelot.read_pdf(
            pdf_path_str,
            pages=str(page_num),
            flavor='lattice',
            line_scale=40,
            split_text=True,
            suppress_stdout=True
        )

        # 2. Fallback to Stream (Whitespace defined tables)
        if len(tables) == 0:
            tables = camelot.read_pdf(
                pdf_path_str,
                pages=str(page_num),
                flavor='stream',
                row_tol=10,
                suppress_stdout=True
            )

        for i, table in enumerate(tables):
            if table.accuracy >= min_accuracy and table.parsing_report['whitespace'] < 90:
                page_tables.append({
                    "page": page_num,
                    "table_index": i,
                    "df_json": table.df.to_json(),
                    "accuracy": table.accuracy
                })
    except Exception:
        pass

    return page_tables


class TableExtractor:
    def __init__(self, min_accuracy: float = 70.0, min_row_length: int = 20):
        self.min_accuracy = min_accuracy
        self.min_row_length = min_row_length

    def extract_tables_by_page(self, pdf_path: Path) -> Dict[int, List[Dict]]:
        tables_by_page = {}
        pdf_path_str = str(pdf_path)

        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                total_pages = len(reader.pages)

            logger.info(f"Scanning {total_pages} pages for tables (Parallel execution)...")

            # Determine workers based on CPU cores
            max_workers = max(1, (os.cpu_count() or 2) - 1)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Pass self.min_accuracy explicitly to the worker
                future_to_page = {
                    executor.submit(process_page_worker, pdf_path_str, page_num, self.min_accuracy): page_num
                    for page_num in range(1, total_pages + 1)
                }

                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        results = future.result()
                        if results:
                            processed_page_tables = []
                            for res in results:
                                # FIX: Wrap JSON string in StringIO
                                json_buffer = io.StringIO(res['df_json'])
                                df = pd.read_json(json_buffer)

                                processed_table = self._process_dataframe(df, res['accuracy'], res['table_index'])
                                if processed_table:
                                    processed_table['page'] = page_num
                                    processed_page_tables.append(processed_table)

                            if processed_page_tables:
                                tables_by_page[page_num] = processed_page_tables
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize table extraction: {e}")

        return tables_by_page

    def _process_dataframe(self, df: pd.DataFrame, accuracy: float, table_index: int) -> Optional[Dict]:
        if df.empty or len(df) < 2:
            return None

        df = self._clean_dataframe(df)
        if df.empty or len(df) < 2:
            return None

        # --- Smart Header / Title Detection ---
        table_title = ""
        headers = []
        data_start_index = 1

        row_0 = df.iloc[0].tolist()
        unique_vals_r0 = set([str(x).strip() for x in row_0 if str(x).strip()])

        if len(unique_vals_r0) == 1 and len(df.columns) > 1:
            table_title = list(unique_vals_r0)[0]
            if len(df) > 2:
                headers = [self._clean_cell(str(h)) for h in df.iloc[1].tolist()]
                data_start_index = 2
            else:
                headers = [f"Col_{i}" for i in range(len(df.columns))]
                data_start_index = 1
        else:
            headers = [self._clean_cell(str(h)) for h in df.iloc[0].tolist()]
            for idx, h in enumerate(headers):
                if not h:
                    headers[idx] = "Item" if idx == 0 else f"Column_{idx}"
            data_start_index = 1

        data_df = df.iloc[data_start_index:].reset_index(drop=True)
        if data_df.empty:
            return None

        formatted_rows = self._format_rows(headers, data_df)
        valid_rows = [row for row in formatted_rows if len(row) >= self.min_row_length]

        if not valid_rows:
            return None

        return {
            'table_index': table_index,
            'accuracy': accuracy,
            'title': table_title,
            'headers': headers,
            'rows': valid_rows,
            'row_count': len(valid_rows)
        }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.astype(str)
        df = df.apply(lambda x: x.str.replace(r'\n', ' ', regex=True))
        df = df.apply(lambda x: x.str.replace(r'\s+', ' ', regex=True))
        df = df.apply(lambda x: x.str.strip())

        with pd.option_context("future.no_silent_downcasting", True):
            df = df.replace(['nan', 'None', 'NaN', '', '.'], np.nan)

        # Drop completely empty rows/cols
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')

        # Fill NaN back to string
        df = df.fillna('')

        return df.reset_index(drop=True)

    def _clean_cell(self, cell: str) -> str:
        if not cell: return ''
        return ' '.join(cell.replace('\x00', '').split())

    def _format_rows(self, headers: List[str], df: pd.DataFrame) -> List[str]:
        formatted_rows = []
        for _, row in df.iterrows():
            parts = []
            for header, value in zip(headers, row):
                value = str(value).strip()
                if not value or value.lower() in ['nan', 'none', '-']:
                    continue
                parts.append(f"{header}: {value}")

            if parts:
                formatted_rows.append(', '.join(parts))
        return formatted_rows


def format_table_for_chunking(table_data: Dict, chunk_size: int = 300,overlap_chunk: int = 0 ) -> List[Dict]:
    chunks = []
    page = table_data.get('page', 0)
    rows = table_data.get('rows', [])
    title = table_data.get('title', "")

    # Prepare the context header
    if title:
        context_prefix = f"Table '{title}': "
    else:
        context_prefix = "Specification Table: "

    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_chunk,
        separators=[", ", " ", ""]
    )

    for i, row in enumerate(rows):
        full_text = f"{context_prefix}{row}"

        # Check if this single row exceeds the table chunk limit
        if len(full_text) > chunk_size:
            # If the row is huge (e.g., a long text description inside a table),
            # split it into sub-chunks so we don't lose info due to context window limits.
            sub_splits = table_splitter.split_text(full_text)

            for j, split_text in enumerate(sub_splits):
                chunks.append({
                    'text': split_text,
                    'metadata': {
                        'page': page,
                        'content_type': 'table',
                        'row_index': i,
                        'split_part': j,
                        'accuracy': table_data.get('accuracy', 0),
                        'table_title': title
                    }
                })
        else:
            # Standard case: The row fits (ideal for specs like "Oil: 5W30")
            chunks.append({
                'text': full_text,
                'metadata': {
                    'page': page,
                    'content_type': 'table',
                    'row_index': i,
                    'accuracy': table_data.get('accuracy', 0),
                    'table_title': title
                }
            })

    return chunks