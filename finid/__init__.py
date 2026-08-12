"""Streaming fin-identification pipeline and desktop application support."""

from .models import IdentificationModel, discover_detection_models, discover_identification_models
from .pipeline import PipelineConfig, PipelineSummary, run_pipeline

__all__ = [
    "IdentificationModel",
    "PipelineConfig",
    "PipelineSummary",
    "discover_detection_models",
    "discover_identification_models",
    "run_pipeline",
]
