from typing import List, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Common words to filter from tags
COMMON_WORDS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
    'those', 'image', 'diagram', 'illustration', 'figure', 'shown', 'display'
}

SYSTEM_INSTRUCTIONS = (
    "You are an automotive technical writer.\n"
    "Generate a concise caption (5-8 words) for the image based on the context.\n"
    "Focus on functionality: usage, activation, warnings, or location.\n"
    "Include specific component names like 'turn signal', 'indicator stalk', 'fuse box', 'isofix point'.\n"
    "Example: 'turn signal indicator stalk operation', 'engine oil dipstick location'"
)


class ImageEmbedder:
    def __init__(self, embedder, llm_client=None, llm_model: str = "gpt-3.5-turbo"):
        self.embedder = embedder
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.use_llm = llm_client is not None

    def generate_tag(
            self,
            text_context: str,
            max_attempts: int = 3
    ) -> str:
        if not self.use_llm:
            return self._generate_fallback_tag(text_context)

        text_sample = text_context[:500].strip() or "automotive component"

        base_prompt = f"""
Based on the contextual text below, generate EXACTLY 3 or 4 UNIQUE descriptive keywords.
Focus ONLY on the specific automotive component, feature, or action shown.

Context:
\"\"\"
{text_sample}
\"\"\"

Output only the keywords:
"""

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": base_prompt},
        ]

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=20,
                    temperature=0.4,
                )

                raw_output = response.choices[0].message.content.strip()
                tag = self._clean_tag(raw_output)

                if tag and not self._is_too_generic(tag):
                    logger.debug(f"Generated tag (attempt {attempt}): {tag}")
                    return tag

                # Retry with feedback
                messages.extend([
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            "INVALID OUTPUT. Exactly 3 or 4 UNIQUE lowercase words. "
                            "Be highly specific. Correct and output ONLY the keywords."
                        )
                    }
                ])

            except Exception as e:
                logger.error(f"LLM error on attempt {attempt}: {e}")

        # Fallback
        tag = self._generate_fallback_tag(text_context)
        logger.warning(f"Using fallback tag: {tag}")
        return tag

    def _clean_tag(self, text: str) -> Optional[str]:
        if not text:
            return None

        # Keep only alphabetic and spaces
        cleaned = ''.join(c for c in text.lower() if c.isalpha() or c == ' ')
        words = cleaned.split()

        # Get unique words (3-4 words)
        unique_words = []
        for w in words:
            if len(w) >= 2 and w not in unique_words:
                unique_words.append(w)
            if len(unique_words) == 4:
                break

        if not (3 <= len(unique_words) <= 4):
            return None

        return " ".join(unique_words)

    def _is_too_generic(self, tag: Optional[str]) -> bool:
        if not tag:
            return True
        generic_patterns = {
            "automotive safety instruction",
            "automotive safety warning",
            "safety instruction warning",
            "vehicle safety instruction",
            "generic automotive illustration",
        }
        return tag in generic_patterns

    def _generate_fallback_tag(self, text_context: str) -> str:
        words = []
        for token in text_context.lower().split():
            cleaned = ''.join(c for c in token if c.isalpha())
            if len(cleaned) >= 3 and cleaned not in COMMON_WORDS:
                if cleaned not in words:
                    words.append(cleaned)
                if len(words) == 4:
                    break

        if len(words) < 3:
            words = ["automotive", "manual", "illustration"]

        return " ".join(words[:4])

    def create_embedding(
            self,
            text_context: str,
            tag: str
    ) -> np.ndarray:
        # Combine context and tag for richer embedding
        combined_text = f"{text_context} {tag}".strip()
        return self.embedder.embed_query(combined_text)

    def process_image_batch(
            self,
            images: List[Dict]
    ) -> List[Dict]:

        logger.info(f"Processing {len(images)} images...")

        processed = []
        for i, img in enumerate(images):
            try:
                # Generate tag
                tag = self.generate_tag(img['text_context'])

                # Create embedding
                embedding = self.create_embedding(
                    img['text_context'],
                    tag
                )

                # Add to processed list
                processed.append({
                    **img,
                    'tag': tag,
                    'embedding': embedding
                })

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(images)} images")

            except Exception as e:
                logger.error(f"Error processing image {img.get('image_id')}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed)}/{len(images)} images")
        return processed