"""
Model Manager for loading and managing AI models.

Provides centralized model lifecycle management, memory optimization,
and coordination between different AI model adapters.
"""

import asyncio
import logging
import os
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from fn_media_ai.domain.value_objects.confidence import ModelVersion
from fn_media_ai.infrastructure.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    adapter_type: str
    version: ModelVersion
    memory_usage_mb: float = 0.0
    last_used: datetime = field(default_factory=datetime.utcnow)
    load_time: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    is_loaded: bool = False
    error_count: int = 0


class ModelManager:
    """
    Model Manager for coordinating AI model lifecycle.

    Provides capabilities for:
    - Model loading and unloading
    - Memory management and optimization
    - Performance monitoring
    - Error handling and recovery
    - Resource allocation
    """

    def __init__(self):
        """Initialize model manager."""
        self.settings = get_settings()

        # Model registry
        self._models: Dict[str, ModelInfo] = {}
        self._adapters: Dict[str, Any] = {}

        # Resource management
        self._lock = threading.Lock()
        self._max_memory_usage_gb = 8.0  # Maximum memory for all models
        self._memory_check_interval = 60  # seconds
        self._model_timeout_hours = 2  # Unload unused models after 2 hours

        # Performance tracking
        self._load_times: Dict[str, List[float]] = {}
        self._inference_times: Dict[str, List[float]] = {}

        # Background tasks
        self._cleanup_task = None
        self._monitoring_task = None

    async def start(self):
        """Start model manager background tasks."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Model manager started")

    async def stop(self):
        """Stop model manager and clean up resources."""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Unload all models
        await self.unload_all_models()

        logger.info("Model manager stopped")

    async def register_adapter(self, adapter_type: str, adapter_instance: Any):
        """
        Register an AI adapter with the model manager.

        Args:
            adapter_type: Type of adapter (e.g., 'openai', 'huggingface', 'ocr')
            adapter_instance: The adapter instance
        """
        self._adapters[adapter_type] = adapter_instance

        # Register models from adapter
        if hasattr(adapter_instance, 'get_model_versions'):
            try:
                versions = adapter_instance.get_model_versions()
                if isinstance(versions, dict):
                    for model_name, version in versions.items():
                        full_model_name = f"{adapter_type}_{model_name}"
                        self._models[full_model_name] = ModelInfo(
                            name=full_model_name,
                            adapter_type=adapter_type,
                            version=version,
                            is_loaded=False
                        )
                else:
                    # Single model version
                    self._models[adapter_type] = ModelInfo(
                        name=adapter_type,
                        adapter_type=adapter_type,
                        version=versions,
                        is_loaded=False
                    )

                logger.info(f"Registered adapter: {adapter_type}")

            except Exception as e:
                logger.error(f"Failed to register adapter {adapter_type}: {e}")

    async def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """
        Load a specific model.

        Args:
            model_name: Name of the model to load
            force_reload: Whether to force reload if already loaded

        Returns:
            True if model loaded successfully
        """
        with self._lock:
            if model_name not in self._models:
                logger.error(f"Model {model_name} not registered")
                return False

            model_info = self._models[model_name]

            # Check if already loaded
            if model_info.is_loaded and not force_reload:
                model_info.last_used = datetime.utcnow()
                model_info.usage_count += 1
                return True

        try:
            # Check memory before loading
            if not await self._check_memory_availability():
                await self._free_memory()

            start_time = asyncio.get_event_loop().time()

            # Load model through adapter
            adapter = self._adapters.get(model_info.adapter_type)
            if not adapter:
                logger.error(f"Adapter {model_info.adapter_type} not found")
                return False

            # Call adapter-specific loading method
            success = await self._load_model_via_adapter(adapter, model_name)

            if success:
                load_time = asyncio.get_event_loop().time() - start_time

                with self._lock:
                    model_info.is_loaded = True
                    model_info.last_used = datetime.utcnow()
                    model_info.load_time = datetime.utcnow()
                    model_info.usage_count += 1
                    model_info.error_count = 0  # Reset error count on successful load

                    # Record load time
                    if model_name not in self._load_times:
                        self._load_times[model_name] = []
                    self._load_times[model_name].append(load_time)

                # Update memory usage
                await self._update_memory_usage(model_name)

                logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
                return True
            else:
                with self._lock:
                    model_info.error_count += 1
                logger.error(f"Failed to load model {model_name}")
                return False

        except Exception as e:
            with self._lock:
                self._models[model_name].error_count += 1
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    async def _load_model_via_adapter(self, adapter: Any, model_name: str) -> bool:
        """Load model through its specific adapter."""
        try:
            # Different adapters have different loading patterns
            adapter_type = type(adapter).__name__

            if 'HuggingFace' in adapter_type:
                # Preload HuggingFace models
                await adapter._load_yolo_model()
                await adapter._load_scene_classifier()
                return True

            elif 'OCR' in adapter_type:
                # Preload OCR models
                await adapter._get_easyocr_reader()
                return True

            elif 'OpenAI' in adapter_type:
                # OpenAI models are API-based, test connection
                return await adapter.health_check()

            elif 'Vision' in adapter_type:
                # Vision models adapter doesn't need explicit loading
                return True

            else:
                # Generic health check
                if hasattr(adapter, 'health_check'):
                    return await adapter.health_check()
                return True

        except Exception as e:
            logger.error(f"Adapter loading failed: {e}")
            return False

    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model to free memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if model unloaded successfully
        """
        with self._lock:
            if model_name not in self._models:
                logger.warning(f"Model {model_name} not registered")
                return False

            model_info = self._models[model_name]

            if not model_info.is_loaded:
                return True

        try:
            # Unload through adapter
            adapter = self._adapters.get(model_info.adapter_type)
            if adapter and hasattr(adapter, 'cleanup'):
                if asyncio.iscoroutinefunction(adapter.cleanup):
                    await adapter.cleanup()
                else:
                    adapter.cleanup()

            with self._lock:
                model_info.is_loaded = False
                model_info.memory_usage_mb = 0.0

            logger.info(f"Model {model_name} unloaded")
            return True

        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False

    async def unload_all_models(self):
        """Unload all models to free memory."""
        model_names = list(self._models.keys())

        for model_name in model_names:
            await self.unload_model(model_name)

        logger.info("All models unloaded")

    async def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all registered models."""
        status = {}

        with self._lock:
            for model_name, model_info in self._models.items():
                status[model_name] = {
                    'adapter_type': model_info.adapter_type,
                    'version': str(model_info.version),
                    'is_loaded': model_info.is_loaded,
                    'memory_usage_mb': model_info.memory_usage_mb,
                    'last_used': model_info.last_used.isoformat(),
                    'usage_count': model_info.usage_count,
                    'error_count': model_info.error_count,
                    'avg_load_time': self._get_avg_load_time(model_name),
                }

        return status

    def _get_avg_load_time(self, model_name: str) -> float:
        """Get average load time for a model."""
        load_times = self._load_times.get(model_name, [])
        return sum(load_times) / len(load_times) if load_times else 0.0

    async def _check_memory_availability(self) -> bool:
        """Check if sufficient memory is available for loading models."""
        try:
            # Get current memory usage
            process = psutil.Process()
            current_memory_gb = process.memory_info().rss / (1024 ** 3)

            # Check system memory
            system_memory = psutil.virtual_memory()
            available_gb = system_memory.available / (1024 ** 3)

            # Allow loading if we have enough memory
            return (current_memory_gb < self._max_memory_usage_gb and
                    available_gb > 2.0)  # Keep 2GB available for system

        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            return True  # Assume available if check fails

    async def _free_memory(self):
        """Free memory by unloading least recently used models."""
        with self._lock:
            # Sort models by last used time (oldest first)
            loaded_models = [
                (name, info) for name, info in self._models.items()
                if info.is_loaded
            ]

            loaded_models.sort(key=lambda x: x[1].last_used)

        # Unload oldest models until memory is available
        for model_name, _ in loaded_models:
            await self.unload_model(model_name)

            # Check if we have enough memory now
            if await self._check_memory_availability():
                break

        logger.info("Memory freed by unloading unused models")

    async def _update_memory_usage(self, model_name: str):
        """Update memory usage for a specific model."""
        try:
            # Estimate memory usage (this is approximate)
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / (1024 ** 2)

            with self._lock:
                if model_name in self._models:
                    # Simple estimation - divide current usage by number of loaded models
                    loaded_count = sum(1 for info in self._models.values() if info.is_loaded)
                    estimated_usage = current_memory_mb / max(1, loaded_count)
                    self._models[model_name].memory_usage_mb = estimated_usage

        except Exception as e:
            logger.warning(f"Memory usage update failed: {e}")

    async def _cleanup_loop(self):
        """Background task to clean up unused models."""
        while True:
            try:
                await asyncio.sleep(self._memory_check_interval)

                current_time = datetime.utcnow()
                models_to_unload = []

                with self._lock:
                    for model_name, model_info in self._models.items():
                        if (model_info.is_loaded and
                            current_time - model_info.last_used > timedelta(hours=self._model_timeout_hours)):
                            models_to_unload.append(model_name)

                # Unload expired models
                for model_name in models_to_unload:
                    await self.unload_model(model_name)
                    logger.info(f"Unloaded unused model: {model_name}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _monitoring_loop(self):
        """Background task to monitor model performance."""
        while True:
            try:
                await asyncio.sleep(300)  # Monitor every 5 minutes

                # Update memory usage for all loaded models
                with self._lock:
                    loaded_models = [
                        name for name, info in self._models.items()
                        if info.is_loaded
                    ]

                for model_name in loaded_models:
                    await self._update_memory_usage(model_name)

                # Log resource usage
                total_memory = sum(
                    info.memory_usage_mb for info in self._models.values()
                    if info.is_loaded
                )

                logger.info(f"Model manager: {len(loaded_models)} models loaded, "
                           f"{total_memory:.1f}MB total memory usage")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

    async def preload_essential_models(self):
        """Preload essential models for faster inference."""
        essential_models = [
            'huggingface_yolo',
            'huggingface_scene_classifier',
            'ocr_easyocr'
        ]

        load_tasks = []
        for model_name in essential_models:
            if model_name in self._models:
                load_tasks.append(self.load_model(model_name))

        if load_tasks:
            results = await asyncio.gather(*load_tasks, return_exceptions=True)
            successful_loads = sum(1 for result in results if result is True)
            logger.info(f"Preloaded {successful_loads}/{len(load_tasks)} essential models")

    async def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all adapters and models.

        Returns:
            Dictionary with health status of each adapter
        """
        health_status = {}

        for adapter_type, adapter in self._adapters.items():
            try:
                if hasattr(adapter, 'health_check'):
                    health_status[adapter_type] = await adapter.health_check()
                else:
                    health_status[adapter_type] = True
            except Exception as e:
                logger.error(f"Health check failed for {adapter_type}: {e}")
                health_status[adapter_type] = False

        return health_status

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        stats = {
            'total_models': len(self._models),
            'loaded_models': sum(1 for info in self._models.values() if info.is_loaded),
            'total_memory_mb': sum(info.memory_usage_mb for info in self._models.values()),
            'load_times': {
                model: self._get_avg_load_time(model)
                for model in self._load_times.keys()
            },
            'usage_counts': {
                name: info.usage_count
                for name, info in self._models.items()
            },
            'error_counts': {
                name: info.error_count
                for name, info in self._models.items()
                if info.error_count > 0
            }
        }

        return stats