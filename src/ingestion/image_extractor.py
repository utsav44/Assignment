import fitz
from pathlib import Path
from typing import List, Dict, Optional
import logging
import io
from PIL import Image, ImageStat

logger = logging.getLogger(__name__)


class ImageExtractor:
    def __init__(self,
                 min_image_size: int = 5000,
                 min_width: int = 100,
                 min_height: int = 100,
                 max_aspect_ratio: float = 5.0,
                 min_variance: float = 100.0,
                 header_footer_cutoff: float = 0.10):
        self.min_image_size = min_image_size
        self.min_width = min_width
        self.min_height = min_height
        self.max_aspect_ratio = max_aspect_ratio
        self.min_variance = min_variance
        self.header_footer_cutoff = header_footer_cutoff

    def extract_images_from_pdf(self, pdf_path: Path) -> List[Dict]:
        logger.info(f"Starting image extraction: {pdf_path}")
        images = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_width = page.rect.width
                page_height = page.rect.height

                # Get page text and text blocks
                page_text = page.get_text("text").strip()
                text_blocks = page.get_text("blocks")

                # Get all images on the page
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]

                        # Get image position
                        img_bbox = self._get_image_bbox(page, xref)

                        # 1. Position Filter: Check if image is in Header or Footer
                        if img_bbox:
                            if self._is_header_or_footer(img_bbox, page_height):
                                logger.debug(f"Skipping P{page_num}-I{img_index}: Header/Footer detected")
                                continue

                        # Extract image data
                        pix = fitz.Pixmap(doc, xref)

                        # 2. Dimensions Filter (Fitz level)
                        if pix.width < self.min_width or pix.height < self.min_height:
                            logger.debug(
                                f"Skipping P{page_num}-I{img_index}: Dimensions too small ({pix.width}x{pix.height})")
                            pix = None
                            continue

                        if pix.width * pix.height < self.min_image_size:
                            pix = None
                            continue

                        # Convert to PNG for further analysis
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                        else:
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix_rgb.tobytes("png")
                            pix_rgb = None

                        pix = None  # Free memory
                        if self._is_junk_image(img_data, image_id=f"P{page_num}-I{img_index}"):
                            continue

                        # Build contextual text
                        text_context = self._build_image_context(
                            doc=doc,
                            page=page,
                            page_num=page_num,
                            img_bbox=img_bbox,
                            text_blocks=text_blocks,
                            page_text=page_text
                        )

                        image_id = f"page{page_num}_img{img_index}"
                        images.append({
                            "image_id": image_id,
                            "text_context": text_context,
                            "image_data": img_data,
                            "page_num": page_num,
                            "img_index": img_index,
                            "bbox": img_bbox
                        })

                    except Exception as e:
                        logger.error(
                            f"Error extracting image {img_index} on page {page_num}: {e}"
                        )
                        continue

            doc.close()
            logger.info(f"Extracted {len(images)} valid images from PDF")
            return images

        except Exception as e:
            logger.error(f"Image extraction failed: {e}", exc_info=True)
            raise

    def _is_header_or_footer(self, bbox: fitz.Rect, page_height: float) -> bool:
        top_cutoff = page_height * self.header_footer_cutoff
        bottom_cutoff = page_height * (1 - self.header_footer_cutoff)

        # If the bottom of the image is in the top header zone
        if bbox.y1 < top_cutoff:
            return True
        # If the top of the image is in the bottom footer zone
        if bbox.y0 > bottom_cutoff:
            return True
        return False

    def _is_junk_image(self, img_data: bytes, image_id: str) -> bool:

        try:
            with Image.open(io.BytesIO(img_data)) as img:
                width, height = img.size

                # 1. Aspect Ratio Check
                # Removes long thin lines (dividers)
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > self.max_aspect_ratio:
                    logger.debug(f"Skipping {image_id}: Extreme aspect ratio {aspect_ratio:.2f}")
                    return True

                # 2. Variance/Entropy Check (The "Black Box" Killer)
                # Convert to grayscale to check for information density
                grayscale = img.convert("L")
                stat = ImageStat.Stat(grayscale)
                variance = stat.var[0]

                # Solid blocks (black bars) will have variance near 0
                # Simple gradients will have low variance
                # Complex diagrams will have high variance
                if variance < self.min_variance:
                    logger.debug(f"Skipping {image_id}: Low variance {variance:.2f} (likely solid block)")
                    return True

                return False
        except Exception as e:
            logger.warning(f"Failed to analyze image content for {image_id}: {e}")
            return False  # Keep it if we can't analyze it

    def _get_image_bbox(self, page, xref) -> Optional[fitz.Rect]:

        try:
            img_rects = page.get_image_rects(xref)
            if img_rects:
                return img_rects[0]
        except:
            pass
        return None

    def _build_image_context(self, doc, page, page_num: int, img_bbox: Optional[fitz.Rect], text_blocks: List,
                             page_text: str) -> str:
        if not img_bbox:
            return page_text[:2000] if page_text else ""

        context_parts = []
        text_above = self._extract_text_above(text_blocks, img_bbox)
        if text_above: context_parts.append(text_above)
        text_below = self._extract_text_below(text_blocks, img_bbox)
        if text_below: context_parts.append(text_below)
        text_beside = self._extract_text_beside(text_blocks, img_bbox)
        if text_beside: context_parts.append(text_beside)

        page_height = page.rect.height
        if img_bbox.y0 < page_height * 0.35 and page_num > 0:
            prev_text = self._extract_previous_page_bottom(doc, page_num)
            if prev_text: context_parts.append(prev_text)

        combined_context = " ".join(context_parts)
        if len(combined_context) < 50:
            combined_context = page_text[:2000] if page_text else ""

        return combined_context

    def _extract_text_above(self, text_blocks: List, img_bbox: fitz.Rect, margin: int = 10) -> str:
        texts = []
        for block in text_blocks:
            if len(block) < 5: continue
            block_rect = fitz.Rect(block[:4])
            if block_rect.y1 <= img_bbox.y0 + margin:
                content = block[4].strip()
                if content: texts.append(content)
        return " ".join(texts[-3:]) if texts else ""

    def _extract_text_below(self, text_blocks: List, img_bbox: fitz.Rect, margin: int = 10) -> str:
        texts = []
        for block in text_blocks:
            if len(block) < 5: continue
            block_rect = fitz.Rect(block[:4])
            if block_rect.y0 >= img_bbox.y1 - margin:
                content = block[4].strip()
                if content: texts.append(content)
        return " ".join(texts[:3]) if texts else ""

    def _extract_text_beside(self, text_blocks: List, img_bbox: fitz.Rect, margin: int = 20) -> str:
        texts = []
        for block in text_blocks:
            if len(block) < 5: continue
            block_rect = fitz.Rect(block[:4])
            is_aligned = (block_rect.y0 < img_bbox.y1 + margin and block_rect.y1 > img_bbox.y0 - margin)
            is_beside = (block_rect.x1 <= img_bbox.x0 + margin or block_rect.x0 >= img_bbox.x1 - margin)
            if is_aligned and is_beside:
                content = block[4].strip()
                if content: texts.append(content)
        return " ".join(texts) if texts else ""

    def _extract_previous_page_bottom(self, doc, current_page_num: int, region_fraction: float = 0.5) -> str:
        if current_page_num <= 0: return ""
        try:
            prev_page = doc[current_page_num - 1]
            prev_blocks = prev_page.get_text("blocks")
            prev_height = prev_page.rect.height
            bottom_rect = fitz.Rect(0, prev_height * region_fraction, prev_page.rect.width, prev_height)
            texts = []
            for block in prev_blocks:
                if len(block) < 5: continue
                block_rect = fitz.Rect(block[:4])
                if block_rect.intersects(bottom_rect):
                    content = block[4].strip()
                    if content: texts.append(content)
            return " ".join(texts) if texts else ""
        except Exception as e:
            logger.warning(f"Could not extract from previous page: {e}")
            return ""