"""
HuggingFace adapter for local model inference.

Provides object detection, scene classification, and other computer vision
capabilities using pre-trained models from HuggingFace and Ultralytics.
"""

import asyncio
import io
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline,
)
from ultralytics import YOLO
import aiohttp

from fn_media_ai.domain.value_objects.confidence import (
    BoundingBox,
    ConfidenceScore,
    ModelVersion,
    ObjectDetection,
    SceneClassification,
)
from fn_media_ai.infrastructure.config.settings import get_settings

logger = logging.getLogger(__name__)


class HuggingFaceAdapter:
    """
    HuggingFace adapter for local computer vision model inference.

    Provides capabilities for:
    - Object detection using YOLO models
    - Scene classification using ResNet/EfficientNet
    - Feature extraction for similarity matching
    - Batch processing for efficiency
    """

    def __init__(self):
        """Initialize HuggingFace adapter with model configurations."""
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() and self.settings.enable_gpu else "cpu"

        # Model storage
        self._yolo_model = None
        self._scene_classifier = None
        self._scene_processor = None

        # Model versions for traceability
        self.yolo_version = ModelVersion(
            name="yolov8n",
            version="8.0.196",
            provider="ultralytics"
        )

        self.scene_version = ModelVersion(
            name="resnet50",
            version="1.0",
            provider="huggingface"
        )

        # Create model cache directory
        self.settings.create_model_cache_dir()

    async def _load_yolo_model(self) -> YOLO:
        """Load YOLO model for object detection."""
        if self._yolo_model is None:
            try:
                logger.info(f"Loading YOLO model: {self.settings.yolo_model_path}")

                # Load in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self._yolo_model = await loop.run_in_executor(
                    None,
                    lambda: YOLO(self.settings.yolo_model_path)
                )

                # Configure device
                if self.device == "cuda":
                    self._yolo_model.to(self.device)

                logger.info(f"YOLO model loaded successfully on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise

        return self._yolo_model

    async def _load_scene_classifier(self) -> Tuple:
        """Load scene classification model and processor."""
        if self._scene_classifier is None or self._scene_processor is None:
            try:
                logger.info("Loading scene classification model")

                loop = asyncio.get_event_loop()

                # Load model and processor concurrently
                model_task = loop.run_in_executor(
                    None,
                    lambda: AutoModelForImageClassification.from_pretrained(
                        "microsoft/resnet-50",
                        cache_dir=self.settings.model_cache_dir
                    )
                )

                processor_task = loop.run_in_executor(
                    None,
                    lambda: AutoImageProcessor.from_pretrained(
                        "microsoft/resnet-50",
                        cache_dir=self.settings.model_cache_dir
                    )
                )

                self._scene_classifier, self._scene_processor = await asyncio.gather(
                    model_task, processor_task
                )

                # Move model to device
                self._scene_classifier.to(self.device)

                logger.info(f"Scene classifier loaded successfully on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load scene classifier: {e}")
                raise

        return self._scene_classifier, self._scene_processor

    async def _download_image(self, image_url: str) -> Image.Image:
        """Download and prepare image for processing."""
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

    async def detect_objects(
        self,
        image_url: str,
        confidence_threshold: Optional[float] = None
    ) -> List[ObjectDetection]:
        """
        Detect objects in image using YOLO.

        Args:
            image_url: URL of the image to analyze
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of detected objects with bounding boxes and confidence
        """
        try:
            # Use settings threshold if not provided
            threshold = confidence_threshold or self.settings.confidence_threshold

            # Load model and image concurrently
            model_task = self._load_yolo_model()
            image_task = self._download_image(image_url)

            model, image = await asyncio.gather(model_task, image_task)

            # Convert PIL image to format expected by YOLO
            image_array = np.array(image)

            # Run inference in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: model(image_array, conf=threshold, verbose=False)
            )

            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Get detection data
                        confidence = float(boxes.conf[i])
                        class_id = int(boxes.cls[i])
                        class_name = model.names[class_id]

                        # Get bounding box in normalized coordinates
                        box = boxes.xyxyn[i].cpu().numpy()
                        x1, y1, x2, y2 = box

                        # Convert to normalized bounding box format
                        bbox = BoundingBox(
                            x=float(x1),
                            y=float(y1),
                            width=float(x2 - x1),
                            height=float(y2 - y1)
                        )

                        # Create detection object
                        detection = ObjectDetection(
                            name=class_name,
                            confidence=ConfidenceScore(confidence),
                            bounding_box=bbox,
                            attributes={
                                'model': 'yolo',
                                'class_id': class_id
                            }
                        )

                        detections.append(detection)

            # Filter for Lost & Found relevant objects
            relevant_detections = self._filter_lost_found_objects(detections)

            logger.info(f"Detected {len(relevant_detections)} relevant objects in {image_url}")
            return relevant_detections

        except Exception as e:
            logger.error(f"Object detection failed for {image_url}: {e}")
            return []

    def _filter_lost_found_objects(
        self,
        detections: List[ObjectDetection]
    ) -> List[ObjectDetection]:
        """
        Filter detections for objects relevant to Lost & Found.

        Args:
            detections: List of all detected objects

        Returns:
            List of objects relevant to Lost & Found scenarios
        """
        # Define Lost & Found relevant object categories
        lost_found_classes = {
            # Personal electronics
            'cell phone', 'phone', 'laptop', 'tablet', 'camera', 'headphones',

            # Personal belongings
            'handbag', 'backpack', 'suitcase', 'purse', 'wallet', 'keys',

            # Clothing and accessories
            'shoe', 'sneaker', 'boot', 'hat', 'cap', 'glasses', 'sunglasses',
            'watch', 'tie', 'scarf', 'glove', 'jacket', 'coat',

            # Personal items
            'book', 'bottle', 'cup', 'umbrella', 'toy', 'teddy bear',

            # Sports equipment
            'sports ball', 'tennis racket', 'bicycle', 'skateboard',

            # Musical instruments
            'guitar', 'keyboard', 'violin',
        }

        relevant = []
        for detection in detections:
            # Check if object name matches any Lost & Found category
            object_name = detection.name.lower()

            # Direct match
            if object_name in lost_found_classes:
                relevant.append(detection)
                continue

            # Partial match for compound words
            for lf_class in lost_found_classes:
                if (lf_class in object_name or
                    any(word in object_name for word in lf_class.split())):
                    relevant.append(detection)
                    break

        return relevant

    async def classify_scene(
        self,
        image_url: str,
        top_k: int = 3
    ) -> List[SceneClassification]:
        """
        Classify scene/environment in image.

        Args:
            image_url: URL of the image to analyze
            top_k: Number of top classifications to return

        Returns:
            List of scene classifications with confidence scores
        """
        try:
            # Load model and image concurrently
            model_task = self._load_scene_classifier()
            image_task = self._download_image(image_url)

            (model, processor), image = await asyncio.gather(model_task, image_task)

            # Preprocess image
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_executor(
                None,
                lambda: processor(image, return_tensors="pt").to(self.device)
            )

            # Run inference
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None,
                    lambda: model(**inputs)
                )

                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)

            classifications = []

            for i in range(top_k):
                prob = float(top_probs[0][i])
                class_idx = int(top_indices[0][i])

                # Get class name from model config
                class_name = model.config.id2label[class_idx]

                # Create scene classification
                classification = SceneClassification(
                    scene=class_name,
                    confidence=ConfidenceScore(prob),
                    sub_scenes=[]  # Could be enhanced with hierarchical classification
                )

                classifications.append(classification)

            # Filter and enhance for Lost & Found context
            enhanced_classifications = self._enhance_scene_classifications(classifications)

            logger.info(f"Classified scene for {image_url}: {[c.scene for c in enhanced_classifications]}")
            return enhanced_classifications

        except Exception as e:
            logger.error(f"Scene classification failed for {image_url}: {e}")
            return []

    def _enhance_scene_classifications(
        self,
        classifications: List[SceneClassification]
    ) -> List[SceneClassification]:
        """
        Enhance scene classifications for Lost & Found context.

        Args:
            classifications: Original scene classifications

        Returns:
            Enhanced classifications with Lost & Found relevant context
        """
        # Mapping from generic scenes to Lost & Found relevant contexts
        scene_mapping = {
            'street': 'outdoor_street',
            'park': 'outdoor_park',
            'restaurant': 'indoor_restaurant',
            'office': 'indoor_office',
            'store': 'indoor_retail',
            'beach': 'outdoor_beach',
            'airport': 'transit_airport',
            'train_station': 'transit_station',
            'bus': 'transit_bus',
            'car': 'vehicle_interior',
            'classroom': 'indoor_educational',
            'gym': 'indoor_recreational',
            'hospital': 'indoor_medical',
            'library': 'indoor_library',
            'shopping_mall': 'indoor_shopping',
        }

        enhanced = []
        for classification in classifications:
            scene_name = classification.scene.lower()

            # Check for direct mapping
            mapped_scene = None
            for key, value in scene_mapping.items():
                if key in scene_name:
                    mapped_scene = value
                    break

            if mapped_scene:
                # Create enhanced classification with sub-scenes
                enhanced_classification = SceneClassification(
                    scene=mapped_scene,
                    confidence=classification.confidence,
                    sub_scenes=[classification.scene]
                )
                enhanced.append(enhanced_classification)
            else:
                # Keep original if no mapping found
                enhanced.append(classification)

        return enhanced

    async def extract_features(
        self,
        image_url: str
    ) -> Optional[np.ndarray]:
        """
        Extract feature vectors from image for similarity matching.

        Args:
            image_url: URL of the image to analyze

        Returns:
            Feature vector as numpy array, None if extraction fails
        """
        try:
            # Load model and image
            (model, processor), image = await asyncio.gather(
                self._load_scene_classifier(),
                self._download_image(image_url)
            )

            # Preprocess image
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_executor(
                None,
                lambda: processor(image, return_tensors="pt").to(self.device)
            )

            # Extract features from second-to-last layer
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None,
                    lambda: model(**inputs, output_hidden_states=True)
                )

                # Get pooled features before classification layer
                features = outputs.hidden_states[-2].mean(dim=1)  # Global average pooling
                features = features.cpu().numpy().flatten()

            logger.info(f"Extracted {len(features)} features from {image_url}")
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed for {image_url}: {e}")
            return None

    async def process_batch(
        self,
        image_urls: List[str],
        batch_size: int = 4
    ) -> dict:
        """
        Process multiple images in batches for efficiency.

        Args:
            image_urls: List of image URLs to process
            batch_size: Number of images to process concurrently

        Returns:
            Dictionary with results per image URL
        """
        results = {}

        # Process in batches to avoid overwhelming the system
        for i in range(0, len(image_urls), batch_size):
            batch = image_urls[i:i + batch_size]

            # Create tasks for concurrent processing
            batch_tasks = []
            for image_url in batch:
                task = asyncio.gather(
                    self.detect_objects(image_url),
                    self.classify_scene(image_url),
                    self.extract_features(image_url),
                    return_exceptions=True
                )
                batch_tasks.append((image_url, task))

            # Execute batch
            for image_url, task in batch_tasks:
                try:
                    objects, scenes, features = await task
                    results[image_url] = {
                        'objects': objects if not isinstance(objects, Exception) else [],
                        'scenes': scenes if not isinstance(scenes, Exception) else [],
                        'features': features if not isinstance(features, Exception) else None
                    }
                except Exception as e:
                    logger.error(f"Batch processing failed for {image_url}: {e}")
                    results[image_url] = {
                        'objects': [],
                        'scenes': [],
                        'features': None
                    }

            # Brief pause between batches
            await asyncio.sleep(0.1)

        return results

    def get_model_versions(self) -> dict:
        """Get model version information."""
        return {
            'yolo': self.yolo_version,
            'scene_classifier': self.scene_version
        }

    async def health_check(self) -> bool:
        """
        Perform health check on local models.

        Returns:
            True if models can be loaded successfully, False otherwise
        """
        try:
            # Test YOLO model loading
            await self._load_yolo_model()

            # Test scene classifier loading
            await self._load_scene_classifier()

            logger.info("HuggingFace adapter health check passed")
            return True

        except Exception as e:
            logger.error(f"HuggingFace adapter health check failed: {e}")
            return False

    def cleanup(self):
        """Clean up loaded models to free memory."""
        if self._yolo_model is not None:
            del self._yolo_model
            self._yolo_model = None

        if self._scene_classifier is not None:
            del self._scene_classifier
            self._scene_classifier = None

        if self._scene_processor is not None:
            del self._scene_processor
            self._scene_processor = None

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available() and self.settings.enable_gpu:
            torch.cuda.empty_cache()

        logger.info("HuggingFace adapter cleaned up")