"""
OpenAI adapter for GPT-4 Vision API integration.

Provides advanced visual analysis capabilities using GPT-4 Vision
for location inference, scene understanding, and enhanced descriptions.
"""

import asyncio
import base64
import io
import logging
from typing import List, Optional

import aiohttp
from PIL import Image

from fn_media_ai.domain.value_objects.confidence import (
    ConfidenceScore,
    LocationInference,
    ModelVersion,
    ObjectDetection,
    SceneClassification,
)
from fn_media_ai.infrastructure.config.settings import get_settings

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """
    OpenAI GPT-4 Vision adapter for advanced visual analysis.

    Provides capabilities for:
    - Location inference from landmarks and visual cues
    - Enhanced scene understanding and context
    - Natural language description generation
    - Complex visual reasoning for Lost & Found items
    """

    def __init__(self):
        """Initialize OpenAI adapter with configuration."""
        self.settings = get_settings()
        self.api_key = self.settings.openai_api_key
        self.model = self.settings.openai_model
        self.max_tokens = self.settings.openai_max_tokens
        self.temperature = self.settings.openai_temperature
        self.base_url = "https://api.openai.com/v1"

        # Model version for traceability
        self.model_version = ModelVersion(
            name=self.model,
            version="2024-04-09",  # GPT-4 Vision model version
            provider="openai"
        )

        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _encode_image_base64(self, image_url: str) -> str:
        """Download and encode image to base64."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download image: {response.status}")

                    image_data = await response.read()

                    # Resize image if too large (OpenAI has size limits)
                    image = Image.open(io.BytesIO(image_data))
                    if image.width > 1024 or image.height > 1024:
                        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

                        buffer = io.BytesIO()
                        image.save(buffer, format='JPEG', quality=85)
                        image_data = buffer.getvalue()

                    return base64.b64encode(image_data).decode('utf-8')

        except Exception as e:
            logger.error(f"Failed to encode image {image_url}: {e}")
            raise

    async def _make_vision_request(
        self,
        image_base64: str,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """Make a request to GPT-4 Vision API."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"OpenAI API error {response.status}: {error_text}")

                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenAI Vision API request failed: {e}")
            raise

    async def infer_location_from_landmarks(
        self,
        image_url: str
    ) -> Optional[LocationInference]:
        """
        Infer location from landmarks and visual cues in the image.

        Args:
            image_url: URL of the image to analyze

        Returns:
            LocationInference if landmarks are detected, None otherwise
        """
        try:
            image_base64 = await self._encode_image_base64(image_url)

            prompt = """
            Analyze this image for recognizable landmarks, buildings, or location markers that could help determine the geographic location.

            Focus on:
            - Famous landmarks or buildings
            - Street signs or place names
            - Business signs with locations
            - Geographic features (mountains, coastlines, etc.)
            - Architecture styles that indicate specific regions

            If you can identify a specific location, respond with:
            LOCATION: [landmark/place name]
            LATITUDE: [latitude in decimal degrees]
            LONGITUDE: [longitude in decimal degrees]
            CONFIDENCE: [confidence score 0.0-1.0]

            If no clear location can be determined, respond with:
            NO_LOCATION_DETECTED
            """

            response = await self._make_vision_request(image_base64, prompt, 200)

            if "NO_LOCATION_DETECTED" in response:
                return None

            # Parse response
            lines = response.strip().split('\n')
            location_data = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    location_data[key.strip()] = value.strip()

            if 'LATITUDE' in location_data and 'LONGITUDE' in location_data:
                try:
                    latitude = float(location_data['LATITUDE'])
                    longitude = float(location_data['LONGITUDE'])
                    confidence = float(location_data.get('CONFIDENCE', '0.7'))
                    landmark_name = location_data.get('LOCATION', 'Unknown landmark')

                    return LocationInference(
                        latitude=latitude,
                        longitude=longitude,
                        confidence=ConfidenceScore(confidence),
                        source="ai_landmark_detection",
                        landmark_name=landmark_name
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse location data: {e}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Location inference failed for {image_url}: {e}")
            return None

    async def analyze_scene_context(
        self,
        image_url: str
    ) -> List[SceneClassification]:
        """
        Analyze scene context and environment in the image.

        Args:
            image_url: URL of the image to analyze

        Returns:
            List of scene classifications with confidence scores
        """
        try:
            image_base64 = await self._encode_image_base64(image_url)

            prompt = """
            Analyze this image and identify the scene/environment where it was taken.
            This is for a Lost & Found system, so focus on location contexts that would help with item recovery.

            Identify:
            - Primary scene type (indoor/outdoor, public/private)
            - Specific location type (park, street, office, restaurant, etc.)
            - Environmental context that could help locate lost items

            Respond in this format:
            SCENE: [primary scene type]
            CONFIDENCE: [confidence score 0.0-1.0]
            SUBSCENES: [comma-separated list of specific location types]
            """

            response = await self._make_vision_request(image_base64, prompt, 300)

            # Parse response
            lines = response.strip().split('\n')
            scene_data = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    scene_data[key.strip()] = value.strip()

            classifications = []

            if 'SCENE' in scene_data:
                try:
                    confidence = float(scene_data.get('CONFIDENCE', '0.8'))
                    subscenes = []

                    if 'SUBSCENES' in scene_data:
                        subscenes = [
                            sub.strip() for sub in scene_data['SUBSCENES'].split(',')
                            if sub.strip()
                        ]

                    classification = SceneClassification(
                        scene=scene_data['SCENE'],
                        confidence=ConfidenceScore(confidence),
                        sub_scenes=subscenes
                    )
                    classifications.append(classification)

                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse scene data: {e}")

            return classifications

        except Exception as e:
            logger.error(f"Scene analysis failed for {image_url}: {e}")
            return []

    async def detect_lost_found_items(
        self,
        image_url: str
    ) -> List[ObjectDetection]:
        """
        Detect and identify items commonly associated with Lost & Found.

        Args:
            image_url: URL of the image to analyze

        Returns:
            List of detected objects with confidence scores
        """
        try:
            image_base64 = await self._encode_image_base64(image_url)

            prompt = """
            Analyze this image for items commonly found in Lost & Found systems.
            Focus on identifying:
            - Personal belongings (phones, wallets, keys, bags, etc.)
            - Electronics (laptops, tablets, headphones, cameras)
            - Clothing and accessories (jackets, shoes, glasses, jewelry)
            - Personal items (books, bottles, umbrellas, toys)

            For each item detected, provide:
            ITEM: [item name]
            CONFIDENCE: [confidence score 0.0-1.0]
            DESCRIPTION: [brief description including color, brand if visible]

            Only include items you're confident about. If no relevant items are visible, respond with:
            NO_ITEMS_DETECTED
            """

            response = await self._make_vision_request(image_base64, prompt, 500)

            if "NO_ITEMS_DETECTED" in response:
                return []

            # Parse response for multiple items
            detections = []
            current_item = {}

            for line in response.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'ITEM':
                        # Save previous item if complete
                        if current_item.get('ITEM') and current_item.get('CONFIDENCE'):
                            try:
                                confidence = float(current_item['CONFIDENCE'])
                                detection = ObjectDetection(
                                    name=current_item['ITEM'],
                                    confidence=ConfidenceScore(confidence),
                                    attributes={
                                        'description': current_item.get('DESCRIPTION', '')
                                    }
                                )
                                detections.append(detection)
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Failed to parse item data: {e}")

                        # Start new item
                        current_item = {'ITEM': value}
                    else:
                        current_item[key] = value

            # Add final item
            if current_item.get('ITEM') and current_item.get('CONFIDENCE'):
                try:
                    confidence = float(current_item['CONFIDENCE'])
                    detection = ObjectDetection(
                        name=current_item['ITEM'],
                        confidence=ConfidenceScore(confidence),
                        attributes={
                            'description': current_item.get('DESCRIPTION', '')
                        }
                    )
                    detections.append(detection)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse final item data: {e}")

            return detections

        except Exception as e:
            logger.error(f"Object detection failed for {image_url}: {e}")
            return []

    async def enhance_description(
        self,
        image_url: str,
        original_description: str
    ) -> str:
        """
        Generate enhanced description using visual analysis.

        Args:
            image_url: URL of the image to analyze
            original_description: Original post description

        Returns:
            Enhanced description with visual details
        """
        try:
            image_base64 = await self._encode_image_base64(image_url)

            prompt = f"""
            Original description: "{original_description}"

            Analyze the image and enhance the description with specific visual details that would help someone identify and locate this lost item.

            Add details about:
            - Colors, materials, and condition
            - Brand markings or distinctive features
            - Size and shape characteristics
            - Any damage or wear patterns
            - Unique identifiers or personalization

            Keep the enhanced description concise but detailed enough for identification.
            Focus on factual visual information only.

            Enhanced description:
            """

            response = await self._make_vision_request(image_base64, prompt, 300)

            # Clean up response
            enhanced = response.strip()
            if enhanced.startswith("Enhanced description:"):
                enhanced = enhanced.replace("Enhanced description:", "").strip()

            return enhanced

        except Exception as e:
            logger.error(f"Description enhancement failed for {image_url}: {e}")
            return original_description

    async def analyze_batch(
        self,
        image_urls: List[str],
        analysis_type: str = "comprehensive"
    ) -> dict:
        """
        Analyze multiple images concurrently.

        Args:
            image_urls: List of image URLs to analyze
            analysis_type: Type of analysis ('comprehensive', 'objects_only', 'scene_only')

        Returns:
            Dictionary with analysis results per image
        """
        tasks = []

        for image_url in image_urls:
            if analysis_type == "comprehensive":
                # Run all analysis types concurrently
                task_group = asyncio.gather(
                    self.detect_lost_found_items(image_url),
                    self.analyze_scene_context(image_url),
                    self.infer_location_from_landmarks(image_url),
                    return_exceptions=True
                )
            elif analysis_type == "objects_only":
                task_group = self.detect_lost_found_items(image_url)
            elif analysis_type == "scene_only":
                task_group = self.analyze_scene_context(image_url)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")

            tasks.append((image_url, task_group))

        # Execute all tasks concurrently
        results = {}
        for image_url, task in tasks:
            try:
                result = await task
                results[image_url] = result
            except Exception as e:
                logger.error(f"Batch analysis failed for {image_url}: {e}")
                results[image_url] = None

        return results

    def get_model_version(self) -> ModelVersion:
        """Get model version information."""
        return self.model_version

    async def health_check(self) -> bool:
        """
        Perform health check on OpenAI API.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple API test
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False