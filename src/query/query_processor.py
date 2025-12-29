from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class QueryProcessor:
    def __init__(self, metadata_manager, embedder, threshold: float = 0.65):
        self.metadata_manager = metadata_manager
        self.embedder = embedder
        self.threshold = threshold

    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        if not query:
            return False, "Please enter a question."

        query = query.strip()

        if len(query) < 5:
            return False, "Please enter a more detailed question."

        # Check for actual question content
        if len(query.split()) < 2:
            return False, "Please enter a complete question."

        return True, None

    def identify_car_model(
            self,
            query: str
    ) -> Tuple[Optional[Dict], Optional[str]]:
        # Check if any manuals are available
        if self.metadata_manager.manual_count == 0:
            return None, self._create_no_manuals_error()

        # Find best matching car model using embedding similarity
        manual_info = self.metadata_manager.find_matching_car_model(
            query,
            threshold=self.threshold
        )

        if manual_info is None:
            return None, self._create_not_found_error(query)

        logger.info(
            f"Matched query to {manual_info['display_name']} "
            f"(confidence: {manual_info['similarity_score']:.2%})"
        )

        return manual_info, None

    def get_car_model_key(self, manual_info: Dict) -> str:

        return manual_info.get("car_model_key") or self._create_key(manual_info)

    def _create_key(self, manual_info: Dict) -> str:
        brand = manual_info.get("brand")
        model = manual_info.get("model", "")
        year = manual_info.get("year")

        brand_part = brand.lower().replace(" ", "_") if brand else "unknown"
        model_part = model.lower().replace(" ", "_")
        year_part = year if year else "unknown"

        return f"{brand_part}_{model_part}_{year_part}"

    def _create_not_found_error(self, query: str) -> str:
        """Create error message when manual is not found."""
        available = self.metadata_manager.list_available()

        if not available:
            return self._create_no_manuals_error()

        # Format available manuals
        formatted = []
        for brand, model, year in available:
            parts = []
            if brand:
                parts.append(brand)
            parts.append(model)
            if year:
                parts.append(f"({year})")
            formatted.append(" ".join(parts))

        available_list = "\n".join(f"• {m}" for m in formatted)

        return f"""❌ **Manual Not Available**

Could not identify a matching car model in your query.

**Available Manuals:**
{available_list}

**Tips:**
- Include the car model name in your question
- Example: "How to turn on indicators in **MG Astor**?"
- Example: "What engine oil to use in **Tata Tiago**?"
"""

    def _create_no_manuals_error(self) -> str:
        return """❌ **No Manuals Available**

No car manuals have been uploaded yet.

Please upload a car manual PDF using the sidebar to get started.
"""

    def extract_keywords(self, query: str) -> List[str]:

        stop_words = {
            "how", "to", "what", "which", "when", "where", "why", "is", "are",
            "the", "a", "an", "in", "on", "for", "with", "my", "i", "can",
            "do", "does", "should", "would", "could", "use", "using", "turn",
            "switch", "find", "get", "change", "check", "need"
        }

        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords