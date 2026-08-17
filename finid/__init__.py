"""Streaming fin-identification pipeline and desktop application support."""

from .models import (
    IdentificationModel,
    discover_detection_models,
    discover_identification_models,
)
from .pipeline import (
    Encounter,
    PipelineConfig,
    PipelineSummary,
    discover_encounters,
    run_pipeline,
)

__all__ = [
    "Encounter",
    "IdentificationModel",
    "PipelineConfig",
    "PipelineSummary",
    "discover_detection_models",
    "discover_encounters",
    "discover_identification_models",
    "run_pipeline",
]
