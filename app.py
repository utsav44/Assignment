"""
Car Manual RAG Assistant - Streamlit Web Application with Image Support
"""
import streamlit as st
from pathlib import Path
from datetime import datetime
import logging
import sys
import os
import shutil
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import Config
from src.utils.metadata import MetadataManager
from src.embedding.embedder import BGEEmbedder
from src.vectorstore.vector_db import FAISSVectorStore
from src.query.retriever import ManualRAG, SimpleRAG
from src.ingestion.pipeline import IngestionPipeline


@st.cache_resource
def initialize_system():
    logger.info("Initializing RAG system...")

    config = Config()

    # Initialize embedder
    with st.spinner("Loading embedding model..."):
        embedder = BGEEmbedder(
            model_name=config.EMBEDDING_MODEL,
            query_prefix=config.QUERY_PREFIX
        )

    # Initialize metadata manager
    metadata = MetadataManager(config.METADATA_DB, embedder)

    # Initialize vector store
    vector_store = FAISSVectorStore(
        index_path=config.VECTORSTORE_DIR / "index.faiss",
        dimension=config.EMBEDDING_DIMENSION
    )

    # Check for OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openai_key:
        import openai
        openai_client = openai.OpenAI(api_key=openai_key)

        # Initialize pipeline with LLM support
        pipeline = IngestionPipeline(config, embedder, vector_store, openai_client)

        rag = ManualRAG(
            vector_store, embedder, metadata, openai_client, config,
            use_reranker=config.USE_RERANKER,
            reranker_model=config.RERANKER_MODEL
        )
        use_llm = True
    else:
        # Initialize pipeline without LLM (fallback tags)
        pipeline = IngestionPipeline(config, embedder, vector_store, None)

        rag = SimpleRAG(
            vector_store, embedder, metadata, None, config,
            use_reranker=True
        )
        use_llm = False
        logger.warning("OpenAI API key not found. Using fallback mode.")

    logger.info("RAG system initialized successfully")

    return config, embedder, metadata, vector_store, rag, pipeline, use_llm


def clear_data_callback():
    """Callback to clear all data safely."""
    try:
        config = Config()

        # 1. Force release of file handles by clearing cache FIRST
        st.cache_resource.clear()

        # 2. Clear Vector Store File
        if config.VECTORSTORE_DIR.exists():
            shutil.rmtree(config.VECTORSTORE_DIR)
            config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Clear Metadata
        if config.METADATA_DB.exists():
            config.METADATA_DB.unlink()

        # 4. Clear Manual PDFs
        if config.MANUALS_DIR.exists():
            for f in config.MANUALS_DIR.glob("*.pdf"):
                try:
                    f.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete {f}: {e}")

        # 5. Clear Extracted Images
        if config.IMAGES_DIR.exists():
            shutil.rmtree(config.IMAGES_DIR)
            config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        # 6. Reset Session State fully
        st.session_state.clear()

        st.toast("✅ All data cleared successfully! System reset.", icon="🗑️")

    except Exception as e:
        st.error(f"Error clearing data: {e}")
        logger.error(f"Clear data error: {e}", exc_info=True)


def main():
    # Page configuration
    st.set_page_config(
        page_title="Car Manual RAG Assistant",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        input { autocomplete: off !important; }
        div[data-testid="stForm"] small { display: none; }
        </style>
        """, unsafe_allow_html=True)

    # Initialize Session State
    if 'is_processing' not in st.session_state:
        st.session_state['is_processing'] = False
    if 'confirm_delete' not in st.session_state:
        st.session_state['confirm_delete'] = False

    # Initialize system
    config, embedder, metadata, vector_store, rag, pipeline, use_llm = initialize_system()

    # Show processing banner if active
    if st.session_state['is_processing']:
        st.warning("⚠️ Processing manual... Please wait. Do not close this tab.")

    with st.container():
        st.title("🚗 Car Manual RAG Assistant")
        st.markdown(
            "Ask questions about your car manuals. The system will find relevant text, "
            "tables, and images to answer your questions."
        )

    if not use_llm:
        st.info(
            "💡 **Tip:** Set `OPENAI_API_KEY` for AI-generated answers and image tagging. "
            "Currently showing raw results with fallback tags."
        )

    # Sidebar
    with st.sidebar:
        st.header("📤 Upload New Manual")

        # RESTRICTION: Disable upload form completely if processing
        if st.session_state['is_processing']:
            st.info("🔒 System is processing. Upload disabled.")
        else:
            render_upload_form(config, pipeline, metadata, use_llm)

        st.divider()

        st.header("📚 Available Manuals")
        render_manual_list(metadata)

        st.divider()

        # Pass disabled state to admin section
        render_admin_section(disabled=st.session_state['is_processing'])

    # Main query interface
    # Disable querying if processing or if delete confirmation is pending
    disable_query = st.session_state['is_processing'] or st.session_state['confirm_delete']
    render_query_interface(rag, use_llm, config, disabled=disable_query)


def render_upload_form(config, pipeline, metadata, use_llm):
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Choose PDF Manual",
            type="pdf",
            help="Upload a car owner's manual in PDF format"
        )

        model = st.text_input(
            "Model Name *",
            placeholder="e.g., Astor, Tiago, Swift, Creta",
            help="Required: The car model name"
        )

        col1, col2 = st.columns(2)
        with col1:
            brand = st.text_input(
                "Brand",
                placeholder="e.g., MG, Tata, Maruti, Hyundai",
                help="Optional: The car manufacturer"
            )
        with col2:
            year = st.text_input(
                "Year",
                placeholder="e.g., 2023, 2024",
                help="Optional: The model year"
            )

        # Extraction options
        extract_tables = st.checkbox("Extract Tables", value=True)
        extract_images = st.checkbox(
            "Extract Images",
            value=use_llm,
            help="Requires OpenAI API key for best results"
        )

        submitted = st.form_submit_button("📥 Upload & Process", use_container_width=True)

        if submitted:
            if not uploaded_file:
                st.error("Please upload a PDF file")
            elif not model.strip():
                st.error("Model name is required!")
            else:
                # Set processing flag
                st.session_state['is_processing'] = True

                try:
                    process_upload(
                        config, pipeline, metadata,
                        uploaded_file, model.strip(),
                        brand.strip() if brand else None,
                        year.strip() if year else None,
                        extract_tables, extract_images
                    )
                finally:
                    # Reset flag even if error occurs
                    st.session_state['is_processing'] = False
                    st.rerun()


def process_upload(config, pipeline, metadata, uploaded_file, model,
                   brand, year, extract_tables, extract_images):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_brand = (brand or "Unknown").replace(" ", "_")
    safe_model = model.replace(" ", "_")
    filename = f"{safe_brand}_{safe_model}_{year or 'Unknown'}_{timestamp}.pdf"

    pdf_path = config.MANUALS_DIR / filename

    # Save uploaded file
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Process
    progress_bar = st.progress(0, text="Starting processing...")

    try:
        progress_bar.progress(20, text="Extracting content from PDF...")

        car_key, stats, display_name = pipeline.process_manual(
            pdf_path=pdf_path,
            model=model,
            brand=brand,
            year=year,
            extract_tables=extract_tables,
            extract_images=extract_images
        )

        progress_bar.progress(80, text="Updating metadata...")

        metadata.add_manual(
            model=model,
            filename=filename,
            brand=brand,
            year=year,
            chunk_count=stats['total_chunks']
        )

        progress_bar.progress(100, text="Complete!")

        st.success(
            f"✅ Successfully processed **{display_name}**\n\n"
            f"📄 Text chunks: {stats['text_chunks']}\n"
            f"📊 Table chunks: {stats['table_chunks']}\n"
            f"🖼️ Image chunks: {stats['image_chunks']}\n"
            f"📦 Total: {stats['total_chunks']}"
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        st.error(f"Processing failed: {str(e)}")

        if pdf_path.exists():
            pdf_path.unlink()

    finally:
        progress_bar.empty()


def render_manual_list(metadata):
    manuals = metadata.list_all_manuals()

    if manuals:
        for manual in manuals:
            brand = manual.get("brand") or "Unknown Brand"
            model = manual.get("model", "Unknown")
            year = manual.get("year") or ""
            chunks = manual.get("chunk_count", 0)

            with st.expander(f"📖 {brand} {model} {year}".strip()):
                st.write(f"**Chunks:** {chunks}")
                st.write(f"**Added:** {manual.get('added_at', 'N/A')[:10]}")
                st.write(f"**File:** {manual.get('filename', 'N/A')}")
    else:
        st.info("No manuals uploaded yet.")


def render_admin_section(disabled=False):
    with st.expander("⚙️ Admin", expanded=True):
        st.warning("⚠️ Danger Zone")

        # Logic to handle the two-step deletion process
        if not st.session_state['confirm_delete']:
            if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True, disabled=disabled):
                st.session_state['confirm_delete'] = True
                st.rerun()
        else:
            st.error("Are you sure? This will delete all manuals, indexes, and images. This cannot be undone.")
            col_yes, col_no = st.columns(2)

            with col_yes:
                st.button(
                    "✅ Yes, Delete",
                    type="primary",
                    on_click=clear_data_callback,
                    use_container_width=True
                )

            with col_no:
                if st.button("❌ Cancel", type="secondary", use_container_width=True):
                    st.session_state['confirm_delete'] = False
                    st.rerun()


def render_query_interface(rag, use_llm, config, disabled=False):
    st.header("🔍 Ask a Question")

    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How to turn on indicators in MG Astor?",
        help="Include the car model name for best results",
        disabled=disabled
    )

    # UI Logic based on LLM availability
    if use_llm:
        # If LLM is available, force optimal top_k from config
        top_k = getattr(config, 'TOP_K_RESULTS', 7)

        # Single column for button since we hid the selectbox
        search_clicked = st.button(
            "🔎 Search",
            type="primary",
            use_container_width=True,
            disabled=disabled
        )
    else:
        # If no LLM, allow user to adjust results (Fallback mode)
        col1, col2 = st.columns([2, 1])
        with col1:
            search_clicked = st.button(
                "🔎 Search",
                type="primary",
                use_container_width=True,
                disabled=disabled
            )
        with col2:
            top_k = st.selectbox(
                "Results",
                [5, 7, 10],
                index=1,
                disabled=disabled,
                label_visibility="collapsed",
                help="Number of results to retrieve"
            )

    if query and search_clicked and not disabled:
        with st.spinner("Searching manual..."):
            result = rag.query(query, top_k=top_k)

        render_query_result(result, use_llm)


def render_query_result(result, use_llm):
    status = result.get("status")

    if status == "success":
        car_info = result.get("car_info", {})
        confidence = result.get("similarity_score", 0)

        st.success(
            f"✅ **Matched:** {car_info.get('display_name', 'Unknown')} "
            f"(Confidence: {confidence:.1%})"
        )

        # Show answer
        st.markdown("### 📝 Answer")
        st.markdown(result.get("answer", "No answer generated."))

        # Show sources with images
        sources = result.get("sources", [])
        if sources:
            st.markdown("### 📚 Sources")

            # Separate by content type
            text_sources = [s for s in sources if s['metadata'].get('content_type') in ['text', 'table']]
            image_sources = [s for s in sources if s['metadata'].get('content_type') == 'image']

            # Display text sources
            if text_sources:
                st.markdown("#### 📄 Text & Table Sources")
                for source in text_sources:
                    page = source.get("page", "N/A")
                    score = source.get("similarity_score", 0)
                    content_type = source['metadata'].get('content_type', 'text')

                    icon = "📊" if content_type == "table" else "📄"

                    with st.expander(
                            f"{icon} Source {source.get('source_number', '?')} | "
                            f"Page {page} | Score: {score:.3f}"
                    ):
                        st.text(source.get("text", ""))

            # Display image sources
            if image_sources:
                st.markdown("#### 🖼️ Image Sources")
                for source in image_sources:
                    page = source.get("page", "N/A")
                    score = source.get("similarity_score", 0)
                    tag = source['metadata'].get('tag', 'No tag')
                    image_path = source['metadata'].get('image_path')

                    with st.expander(
                            f"🖼️ Source {source.get('source_number', '?')} | "
                            f"Page {page} | {tag} | Score: {score:.3f}"
                    ):
                        if image_path and Path(image_path).exists():
                            try:
                                img = Image.open(image_path)
                                st.image(img)
                            except Exception as e:
                                st.error(f"Could not load image: {e}")
                        else:
                            st.warning("Image file not found")

                        # Show context
                        full_context = source['metadata'].get('full_context', '')
                        if full_context:
                            st.caption("Context:")
                            st.text(full_context[:300] + "...")

    elif status == "not_found":
        st.error(result.get("message", "Manual not found."))

    elif status == "no_relevant_content":
        st.warning(result.get("message", "No relevant content found."))

    elif status == "invalid_query":
        st.warning(result.get("message", "Invalid query."))

    else:
        st.error("An unexpected error occurred.")


if __name__ == "__main__":
    main()