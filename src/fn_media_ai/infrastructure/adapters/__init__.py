"""
Infrastructure adapters for AI model integration.

Provides concrete implementations of AI services and external integrations.
"""

from .huggingface_adapter import HuggingFaceAdapter
from .ocr_adapter import OCRAdapter
from .openai_adapter import OpenAIAdapter
from .vision_models import VisionModelsAdapter

__all__ = [
    'HuggingFaceAdapter',
    'OCRAdapter',
    'OpenAIAdapter',
    'VisionModelsAdapter',
]