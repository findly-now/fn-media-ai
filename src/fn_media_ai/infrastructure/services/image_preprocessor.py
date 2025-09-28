"""
Image Preprocessor for consistent image formatting and enhancement.

Provides image preprocessing capabilities to optimize images for
different AI models and improve inference accuracy.
"""

import asyncio
import io
import logging
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import aiohttp

from fn_media_ai.infrastructure.config.settings import get_settings

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessor for AI model optimization.

    Provides capabilities for:
    - Image format standardization
    - Size and resolution optimization
    - Quality enhancement
    - Model-specific preprocessing
    - Batch processing
    """

    def __init__(self):
        """Initialize image preprocessor."""
        self.settings = get_settings()

        # Preprocessing configurations
        self.max_image_size = (2048, 2048)  # Max dimensions for processing
        self.target_sizes = {
            'yolo': (640, 640),
            'scene_classification': (224, 224),
            'ocr': None,  # Keep original size for OCR
            'openai_vision': (1024, 1024),
        }

        # Quality enhancement settings
        self.enhancement_presets = {
            'object_detection': {
                'sharpen': True,
                'contrast': 1.2,
                'brightness': 1.0,
                'saturation': 1.1,
            },
            'scene_classification': {
                'sharpen': False,
                'contrast': 1.1,
                'brightness': 1.0,
                'saturation': 1.0,
            },
            'ocr': {
                'sharpen': True,
                'contrast': 1.5,
                'brightness': 1.1,
                'saturation': 0.8,  # Reduce saturation for better text recognition
                'denoise': True,
                'binarize': True,
            },
            'general': {
                'sharpen': False,
                'contrast': 1.1,
                'brightness': 1.0,
                'saturation': 1.0,
            }
        }

    async def download_image(self, image_url: str) -> Image.Image:
        """
        Download image from URL and convert to PIL Image.

        Args:
            image_url: URL of the image to download

        Returns:
            PIL Image object

        Raises:
            ValueError: If image download or conversion fails
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download image: HTTP {response.status}")

                    image_data = await response.read()

                    # Validate image data
                    if len(image_data) == 0:
                        raise ValueError("Downloaded image is empty")

                    # Check file size
                    max_size_mb = self.settings.max_photo_size_mb
                    if len(image_data) > max_size_mb * 1024 * 1024:
                        raise ValueError(f"Image too large: {len(image_data) / (1024*1024):.1f}MB > {max_size_mb}MB")

                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_data))

                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Validate image dimensions
                    if image.width < 32 or image.height < 32:
                        raise ValueError(f"Image too small: {image.width}x{image.height}")

                    logger.debug(f"Downloaded image: {image.width}x{image.height}, {len(image_data)} bytes")
                    return image

        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            raise

    async def preprocess_for_model(
        self,
        image: Union[Image.Image, str],
        model_type: str,
        enhance_quality: bool = True
    ) -> Image.Image:
        """
        Preprocess image for specific AI model.

        Args:
            image: PIL Image or image URL
            model_type: Target model type ('yolo', 'scene_classification', 'ocr', etc.)
            enhance_quality: Whether to apply quality enhancements

        Returns:
            Preprocessed PIL Image

        Raises:
            ValueError: If preprocessing fails
        """
        try:
            # Download image if URL provided
            if isinstance(image, str):
                image = await self.download_image(image)

            # Apply model-specific preprocessing
            if model_type == 'yolo':
                processed = await self._preprocess_for_object_detection(image, enhance_quality)
            elif model_type == 'scene_classification':
                processed = await self._preprocess_for_scene_classification(image, enhance_quality)
            elif model_type == 'ocr':
                processed = await self._preprocess_for_ocr(image, enhance_quality)
            elif model_type == 'openai_vision':
                processed = await self._preprocess_for_openai_vision(image, enhance_quality)
            else:
                # General preprocessing
                processed = await self._preprocess_general(image, enhance_quality)

            logger.debug(f"Preprocessed image for {model_type}: {processed.width}x{processed.height}")
            return processed

        except Exception as e:
            logger.error(f"Image preprocessing failed for {model_type}: {e}")
            raise

    async def _preprocess_for_object_detection(
        self,
        image: Image.Image,
        enhance: bool
    ) -> Image.Image:
        """Preprocess image for object detection models."""
        # Resize to target size while maintaining aspect ratio
        target_size = self.target_sizes['yolo']
        processed = self._resize_with_padding(image, target_size)

        if enhance:
            preset = self.enhancement_presets['object_detection']
            processed = await self._enhance_image(processed, preset)

        return processed

    async def _preprocess_for_scene_classification(
        self,
        image: Image.Image,
        enhance: bool
    ) -> Image.Image:
        """Preprocess image for scene classification models."""
        # Resize to target size (usually square)
        target_size = self.target_sizes['scene_classification']
        processed = image.resize(target_size, Image.Resampling.LANCZOS)

        if enhance:
            preset = self.enhancement_presets['scene_classification']
            processed = await self._enhance_image(processed, preset)

        return processed

    async def _preprocess_for_ocr(
        self,
        image: Image.Image,
        enhance: bool
    ) -> Image.Image:
        """Preprocess image for OCR models."""
        # Keep original size for OCR, but limit maximum size
        if image.width > self.max_image_size[0] or image.height > self.max_image_size[1]:
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

        processed = image

        if enhance:
            preset = self.enhancement_presets['ocr']
            processed = await self._enhance_image_for_ocr(processed, preset)

        return processed

    async def _preprocess_for_openai_vision(
        self,
        image: Image.Image,
        enhance: bool
    ) -> Image.Image:
        """Preprocess image for OpenAI Vision API."""
        # OpenAI has specific size limits
        target_size = self.target_sizes['openai_vision']

        # Resize if too large
        if image.width > target_size[0] or image.height > target_size[1]:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)

        processed = image

        if enhance:
            preset = self.enhancement_presets['general']
            processed = await self._enhance_image(processed, preset)

        return processed

    async def _preprocess_general(
        self,
        image: Image.Image,
        enhance: bool
    ) -> Image.Image:
        """General image preprocessing."""
        # Limit size to maximum
        if image.width > self.max_image_size[0] or image.height > self.max_image_size[1]:
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

        processed = image

        if enhance:
            preset = self.enhancement_presets['general']
            processed = await self._enhance_image(processed, preset)

        return processed

    def _resize_with_padding(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        fill_color: Tuple[int, int, int] = (128, 128, 128)
    ) -> Image.Image:
        """
        Resize image to target size with padding to maintain aspect ratio.

        Args:
            image: Source image
            target_size: Target (width, height)
            fill_color: Color for padding

        Returns:
            Resized image with padding
        """
        # Calculate scale factor to fit image in target size
        scale = min(target_size[0] / image.width, target_size[1] / image.height)

        # Calculate new size
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target size and fill color
        padded = Image.new('RGB', target_size, fill_color)

        # Paste resized image in center
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))

        return padded

    async def _enhance_image(
        self,
        image: Image.Image,
        preset: dict
    ) -> Image.Image:
        """
        Apply image enhancements based on preset.

        Args:
            image: Source image
            preset: Enhancement preset configuration

        Returns:
            Enhanced image
        """
        enhanced = image.copy()

        # Apply enhancements in executor to avoid blocking
        loop = asyncio.get_event_loop()

        enhanced = await loop.run_in_executor(
            None,
            self._apply_pil_enhancements,
            enhanced,
            preset
        )

        return enhanced

    def _apply_pil_enhancements(self, image: Image.Image, preset: dict) -> Image.Image:
        """Apply PIL-based image enhancements."""
        enhanced = image

        # Adjust brightness
        if preset.get('brightness', 1.0) != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(preset['brightness'])

        # Adjust contrast
        if preset.get('contrast', 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(preset['contrast'])

        # Adjust saturation
        if preset.get('saturation', 1.0) != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(preset['saturation'])

        # Apply sharpening
        if preset.get('sharpen', False):
            enhanced = enhanced.filter(ImageFilter.SHARPEN)

        return enhanced

    async def _enhance_image_for_ocr(
        self,
        image: Image.Image,
        preset: dict
    ) -> Image.Image:
        """
        Apply OCR-specific image enhancements.

        Args:
            image: Source image
            preset: OCR enhancement preset

        Returns:
            Enhanced image optimized for OCR
        """
        enhanced = image.copy()

        # Apply basic enhancements first
        enhanced = await self._enhance_image(enhanced, preset)

        # Apply OCR-specific processing in executor
        loop = asyncio.get_event_loop()

        enhanced = await loop.run_in_executor(
            None,
            self._apply_ocr_enhancements,
            enhanced,
            preset
        )

        return enhanced

    def _apply_ocr_enhancements(self, image: Image.Image, preset: dict) -> Image.Image:
        """Apply OpenCV-based OCR enhancements."""
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply denoising
        if preset.get('denoise', False):
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Apply binarization for text
        if preset.get('binarize', False):
            # Use adaptive threshold for varying lighting conditions
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            gray = binary

        # Morphological operations to clean up text
        if preset.get('denoise', False):
            kernel = np.ones((2, 2), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Convert back to PIL Image
        if len(gray.shape) == 2:  # Grayscale
            enhanced = Image.fromarray(gray, mode='L').convert('RGB')
        else:
            enhanced = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

        return enhanced

    async def batch_preprocess(
        self,
        image_urls: List[str],
        model_type: str,
        enhance_quality: bool = True,
        max_concurrent: int = 3
    ) -> List[Tuple[str, Optional[Image.Image]]]:
        """
        Preprocess multiple images concurrently.

        Args:
            image_urls: List of image URLs to preprocess
            model_type: Target model type
            enhance_quality: Whether to apply quality enhancements
            max_concurrent: Maximum concurrent preprocessing tasks

        Returns:
            List of (url, preprocessed_image) tuples
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def preprocess_single(url: str) -> Tuple[str, Optional[Image.Image]]:
            async with semaphore:
                try:
                    preprocessed = await self.preprocess_for_model(
                        url, model_type, enhance_quality
                    )
                    return (url, preprocessed)
                except Exception as e:
                    logger.error(f"Batch preprocessing failed for {url}: {e}")
                    return (url, None)

        # Create tasks for all images
        tasks = [preprocess_single(url) for url in image_urls]

        # Execute with progress logging
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            if (i + 1) % 5 == 0 or (i + 1) == len(tasks):
                logger.info(f"Batch preprocessing: {i + 1}/{len(tasks)} completed")

        # Sort results to maintain original order
        url_to_result = dict(results)
        ordered_results = [(url, url_to_result[url]) for url in image_urls]

        successful_count = sum(1 for _, img in ordered_results if img is not None)
        logger.info(f"Batch preprocessing completed: {successful_count}/{len(image_urls)} successful")

        return ordered_results

    async def analyze_image_properties(self, image: Union[Image.Image, str]) -> dict:
        """
        Analyze image properties for preprocessing decisions.

        Args:
            image: PIL Image or image URL

        Returns:
            Dictionary with image properties and recommendations
        """
        try:
            # Download image if URL provided
            if isinstance(image, str):
                image = await self.download_image(image)

            # Convert to numpy for analysis
            img_array = np.array(image)

            # Calculate properties
            properties = {
                'width': image.width,
                'height': image.height,
                'aspect_ratio': image.width / image.height,
                'total_pixels': image.width * image.height,
                'channels': len(img_array.shape) if len(img_array.shape) > 2 else 1,
                'mean_brightness': np.mean(img_array),
                'std_brightness': np.std(img_array),
                'is_grayscale': len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)) < 256,
            }

            # Add recommendations
            properties['recommendations'] = self._generate_preprocessing_recommendations(properties)

            return properties

        except Exception as e:
            logger.error(f"Image property analysis failed: {e}")
            return {}

    def _generate_preprocessing_recommendations(self, properties: dict) -> dict:
        """Generate preprocessing recommendations based on image properties."""
        recommendations = {
            'needs_resizing': properties['total_pixels'] > 4194304,  # > 2048x2048
            'is_low_contrast': properties['std_brightness'] < 30,
            'is_very_dark': properties['mean_brightness'] < 80,
            'is_very_bright': properties['mean_brightness'] > 200,
            'suitable_for_ocr': (
                properties['width'] > 200 and
                properties['height'] > 100 and
                properties['std_brightness'] > 20
            ),
            'suitable_for_object_detection': (
                properties['total_pixels'] > 40000 and  # > 200x200
                properties['std_brightness'] > 15
            )
        }

        return recommendations

    def get_preprocessing_stats(self) -> dict:
        """Get preprocessing performance statistics."""
        return {
            'max_image_size': self.max_image_size,
            'target_sizes': self.target_sizes,
            'enhancement_presets': list(self.enhancement_presets.keys()),
        }