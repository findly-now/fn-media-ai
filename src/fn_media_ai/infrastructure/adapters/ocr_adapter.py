"""
OCR adapter for text extraction from images.

Provides text recognition capabilities using Tesseract and EasyOCR
for extracting brands, serial numbers, and other textual information.
"""

import asyncio
import io
import logging
import re
from typing import List, Optional, Tuple

import cv2
import easyocr
import numpy as np
import pytesseract
from PIL import Image
import aiohttp

from fn_media_ai.domain.value_objects.confidence import (
    BoundingBox,
    ConfidenceScore,
    ModelVersion,
    TextExtraction,
)
from fn_media_ai.infrastructure.config.settings import get_settings

logger = logging.getLogger(__name__)


class OCRAdapter:
    """
    OCR adapter for text extraction from images.

    Provides capabilities for:
    - Text extraction using Tesseract and EasyOCR
    - Brand and serial number detection
    - Multi-language text recognition
    - Text preprocessing and enhancement
    """

    def __init__(self):
        """Initialize OCR adapter with configurations."""
        self.settings = get_settings()

        # OCR engine instances (lazy loaded)
        self._easyocr_reader = None
        self._tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

        # Model versions for traceability
        self.tesseract_version = ModelVersion(
            name="tesseract",
            version="5.0",
            provider="google"
        )

        self.easyocr_version = ModelVersion(
            name="easyocr",
            version="1.7.0",
            provider="jaided_ai"
        )

        # Common brand patterns for Lost & Found items
        self.brand_patterns = {
            'electronics': [
                r'\b(APPLE|SAMSUNG|GOOGLE|ONEPLUS|HUAWEI|XIAOMI|OPPO|VIVO)\b',
                r'\b(SONY|CANON|NIKON|PANASONIC|FUJIFILM|OLYMPUS)\b',
                r'\b(DELL|HP|LENOVO|ASUS|ACER|MSI|MICROSOFT)\b',
                r'\b(APPLE|BEATS|BOSE|SENNHEISER|SONY|JBL|SKULLCANDY)\b'
            ],
            'fashion': [
                r'\b(NIKE|ADIDAS|PUMA|REEBOK|NEW BALANCE|CONVERSE|VANS)\b',
                r'\b(GUCCI|PRADA|LV|LOUIS VUITTON|CHANEL|HERMÈS|DIOR)\b',
                r'\b(H&M|ZARA|UNIQLO|GAP|LEVI\'S|TOMMY HILFIGER)\b'
            ],
            'watches': [
                r'\b(ROLEX|OMEGA|TAG HEUER|BREITLING|CARTIER|PATEK PHILIPPE)\b',
                r'\b(CASIO|SEIKO|CITIZEN|TIMEX|FOSSIL|GARMIN|FITBIT)\b'
            ]
        }

        # Serial number patterns
        self.serial_patterns = [
            r'\b[A-Z0-9]{8,20}\b',  # General alphanumeric serial
            r'\b\d{10,15}\b',       # Numeric serial
            r'\b[A-Z]{2,4}\d{6,12}\b', # Letter prefix + numbers
            r'\bSN:?\s*[A-Z0-9]+\b',   # Serial number prefix
            r'\bS/N:?\s*[A-Z0-9]+\b',  # S/N prefix
            r'\bIMEI:?\s*\d{15}\b',     # IMEI numbers
        ]

    async def _get_easyocr_reader(self) -> easyocr.Reader:
        """Get EasyOCR reader instance (lazy loaded)."""
        if self._easyocr_reader is None:
            try:
                logger.info("Loading EasyOCR reader")

                # Load in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self._easyocr_reader = await loop.run_in_executor(
                    None,
                    lambda: easyocr.Reader(['en'], gpu=self.settings.enable_gpu)
                )

                logger.info("EasyOCR reader loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load EasyOCR reader: {e}")
                raise

        return self._easyocr_reader

    async def _download_image(self, image_url: str) -> Image.Image:
        """Download and prepare image for OCR processing."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download image: {response.status}")

                    image_data = await response.read()
                    image = Image.open(io.BytesIO(image_data))

                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    return image

        except Exception as e:
            logger.error(f"Failed to download image {image_url}: {e}")
            raise

    def _preprocess_image_for_ocr(self, image: Image.Image) -> List[np.ndarray]:
        """
        Preprocess image to improve OCR accuracy.

        Args:
            image: PIL Image to preprocess

        Returns:
            List of preprocessed image arrays for different OCR strategies
        """
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        preprocessed_images = []

        # Original image
        preprocessed_images.append(cv_image)

        # Grayscale conversion
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(gray)

        # Increase contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        preprocessed_images.append(enhanced)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        preprocessed_images.append(blurred)

        # Threshold for binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(thresh)

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        preprocessed_images.append(morph)

        return preprocessed_images

    async def _extract_text_tesseract(
        self,
        image_arrays: List[np.ndarray]
    ) -> List[TextExtraction]:
        """
        Extract text using Tesseract OCR.

        Args:
            image_arrays: List of preprocessed image arrays

        Returns:
            List of text extractions with confidence scores
        """
        extractions = []
        loop = asyncio.get_event_loop()

        for i, img_array in enumerate(image_arrays):
            try:
                # Extract text with confidence scores
                data = await loop.run_in_executor(
                    None,
                    lambda: pytesseract.image_to_data(
                        img_array,
                        config=self._tesseract_config,
                        output_type=pytesseract.Output.DICT
                    )
                )

                # Process results
                n_boxes = len(data['text'])
                for j in range(n_boxes):
                    confidence = int(data['conf'][j])
                    text = data['text'][j].strip()

                    # Skip low confidence or empty text
                    if confidence < 30 or len(text) < 2:
                        continue

                    # Calculate normalized bounding box
                    x, y, w, h = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
                    img_height, img_width = img_array.shape[:2]

                    bbox = BoundingBox(
                        x=float(x / img_width),
                        y=float(y / img_height),
                        width=float(w / img_width),
                        height=float(h / img_height)
                    )

                    extraction = TextExtraction(
                        text=text,
                        confidence=ConfidenceScore(confidence / 100.0),
                        bounding_box=bbox,
                        language='en'
                    )

                    extractions.append(extraction)

            except Exception as e:
                logger.warning(f"Tesseract extraction failed for image variant {i}: {e}")
                continue

        return extractions

    async def _extract_text_easyocr(
        self,
        image_arrays: List[np.ndarray]
    ) -> List[TextExtraction]:
        """
        Extract text using EasyOCR.

        Args:
            image_arrays: List of preprocessed image arrays

        Returns:
            List of text extractions with confidence scores
        """
        extractions = []
        reader = await self._get_easyocr_reader()
        loop = asyncio.get_event_loop()

        for i, img_array in enumerate(image_arrays):
            try:
                # Run EasyOCR
                results = await loop.run_in_executor(
                    None,
                    lambda: reader.readtext(img_array, detail=1)
                )

                # Process results
                img_height, img_width = img_array.shape[:2]

                for bbox_coords, text, confidence in results:
                    # Skip low confidence or short text
                    if confidence < 0.5 or len(text.strip()) < 2:
                        continue

                    # Calculate normalized bounding box
                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    bbox = BoundingBox(
                        x=float(x_min / img_width),
                        y=float(y_min / img_height),
                        width=float((x_max - x_min) / img_width),
                        height=float((y_max - y_min) / img_height)
                    )

                    extraction = TextExtraction(
                        text=text.strip(),
                        confidence=ConfidenceScore(confidence),
                        bounding_box=bbox,
                        language='en'
                    )

                    extractions.append(extraction)

            except Exception as e:
                logger.warning(f"EasyOCR extraction failed for image variant {i}: {e}")
                continue

        return extractions

    def _combine_and_deduplicate_extractions(
        self,
        extractions: List[TextExtraction]
    ) -> List[TextExtraction]:
        """
        Combine and deduplicate text extractions from multiple methods.

        Args:
            extractions: List of all text extractions

        Returns:
            Deduplicated and ranked text extractions
        """
        # Group similar texts
        grouped = {}
        for extraction in extractions:
            text_lower = extraction.text.lower().strip()

            # Skip very short text
            if len(text_lower) < 2:
                continue

            # Find existing similar text
            found_group = None
            for existing_text in grouped.keys():
                # Check for exact match or high similarity
                if (text_lower == existing_text or
                    text_lower in existing_text or
                    existing_text in text_lower):
                    found_group = existing_text
                    break

            if found_group:
                grouped[found_group].append(extraction)
            else:
                grouped[text_lower] = [extraction]

        # Select best extraction from each group
        final_extractions = []
        for text_group in grouped.values():
            # Choose extraction with highest confidence
            best_extraction = max(text_group, key=lambda x: x.confidence.value)
            final_extractions.append(best_extraction)

        # Sort by confidence descending
        final_extractions.sort(key=lambda x: x.confidence.value, reverse=True)

        return final_extractions

    def _detect_brands_and_serials(
        self,
        extractions: List[TextExtraction]
    ) -> Tuple[List[str], List[str]]:
        """
        Detect brand names and serial numbers from extracted text.

        Args:
            extractions: List of text extractions

        Returns:
            Tuple of (brand_names, serial_numbers)
        """
        brands = set()
        serials = set()

        for extraction in extractions:
            text = extraction.text.upper()

            # Check for brand patterns
            for category, patterns in self.brand_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    brands.update(matches)

            # Check for serial number patterns
            for pattern in self.serial_patterns:
                matches = re.findall(pattern, text)
                serials.update(matches)

        return list(brands), list(serials)

    async def extract_text(
        self,
        image_url: str,
        use_both_engines: bool = True
    ) -> List[TextExtraction]:
        """
        Extract text from image using OCR.

        Args:
            image_url: URL of the image to analyze
            use_both_engines: Whether to use both Tesseract and EasyOCR

        Returns:
            List of text extractions with confidence scores
        """
        try:
            # Download and preprocess image
            image = await self._download_image(image_url)
            image_arrays = self._preprocess_image_for_ocr(image)

            # Run OCR engines
            all_extractions = []

            if use_both_engines:
                # Run both engines concurrently
                tesseract_task = self._extract_text_tesseract(image_arrays)
                easyocr_task = self._extract_text_easyocr(image_arrays)

                tesseract_results, easyocr_results = await asyncio.gather(
                    tesseract_task, easyocr_task, return_exceptions=True
                )

                if not isinstance(tesseract_results, Exception):
                    all_extractions.extend(tesseract_results)

                if not isinstance(easyocr_results, Exception):
                    all_extractions.extend(easyocr_results)
            else:
                # Use only EasyOCR (generally more accurate)
                easyocr_results = await self._extract_text_easyocr(image_arrays)
                all_extractions.extend(easyocr_results)

            # Combine and deduplicate results
            final_extractions = self._combine_and_deduplicate_extractions(all_extractions)

            logger.info(f"Extracted {len(final_extractions)} text segments from {image_url}")
            return final_extractions

        except Exception as e:
            logger.error(f"Text extraction failed for {image_url}: {e}")
            return []

    async def detect_brands_and_serials(
        self,
        image_url: str
    ) -> dict:
        """
        Detect brand names and serial numbers in image.

        Args:
            image_url: URL of the image to analyze

        Returns:
            Dictionary with detected brands and serial numbers
        """
        try:
            # Extract text first
            extractions = await self.extract_text(image_url)

            # Detect brands and serials
            brands, serials = self._detect_brands_and_serials(extractions)

            result = {
                'brands': brands,
                'serial_numbers': serials,
                'confidence_scores': {
                    'brands': [0.8] * len(brands),  # High confidence for pattern matches
                    'serials': [0.9] * len(serials)  # Very high confidence for serials
                }
            }

            logger.info(f"Detected {len(brands)} brands and {len(serials)} serials in {image_url}")
            return result

        except Exception as e:
            logger.error(f"Brand/serial detection failed for {image_url}: {e}")
            return {'brands': [], 'serial_numbers': [], 'confidence_scores': {'brands': [], 'serials': []}}

    async def process_batch(
        self,
        image_urls: List[str],
        batch_size: int = 3
    ) -> dict:
        """
        Process multiple images in batches.

        Args:
            image_urls: List of image URLs to process
            batch_size: Number of images to process concurrently

        Returns:
            Dictionary with OCR results per image URL
        """
        results = {}

        # Process in smaller batches (OCR is memory intensive)
        for i in range(0, len(image_urls), batch_size):
            batch = image_urls[i:i + batch_size]

            # Create tasks for concurrent processing
            batch_tasks = []
            for image_url in batch:
                task = asyncio.gather(
                    self.extract_text(image_url),
                    self.detect_brands_and_serials(image_url),
                    return_exceptions=True
                )
                batch_tasks.append((image_url, task))

            # Execute batch
            for image_url, task in batch_tasks:
                try:
                    text_extractions, brand_serial_data = await task

                    results[image_url] = {
                        'text_extractions': text_extractions if not isinstance(text_extractions, Exception) else [],
                        'brands': brand_serial_data.get('brands', []) if not isinstance(brand_serial_data, Exception) else [],
                        'serial_numbers': brand_serial_data.get('serial_numbers', []) if not isinstance(brand_serial_data, Exception) else []
                    }
                except Exception as e:
                    logger.error(f"OCR batch processing failed for {image_url}: {e}")
                    results[image_url] = {
                        'text_extractions': [],
                        'brands': [],
                        'serial_numbers': []
                    }

            # Brief pause between batches
            await asyncio.sleep(0.2)

        return results

    def get_model_versions(self) -> dict:
        """Get model version information."""
        return {
            'tesseract': self.tesseract_version,
            'easyocr': self.easyocr_version
        }

    async def health_check(self) -> bool:
        """
        Perform health check on OCR engines.

        Returns:
            True if OCR engines are working, False otherwise
        """
        try:
            # Test Tesseract
            test_image = Image.new('RGB', (100, 50), color='white')
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pytesseract.image_to_string(test_image)
            )

            # Test EasyOCR
            await self._get_easyocr_reader()

            logger.info("OCR adapter health check passed")
            return True

        except Exception as e:
            logger.error(f"OCR adapter health check failed: {e}")
            return False

    def cleanup(self):
        """Clean up OCR resources."""
        if self._easyocr_reader is not None:
            del self._easyocr_reader
            self._easyocr_reader = None

        logger.info("OCR adapter cleaned up")