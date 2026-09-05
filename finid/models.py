"""Model discovery, validation, and bounded identification inference."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True, slots=True)
class DetectionModel:
    """Describe one discovered fin-detection checkpoint."""

    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class IdentificationModel:
    """Describe one validated identification checkpoint and gallery."""

    name: str
    path: Path
    kind: str
    score_label: str
    gallery_path: Path | None = None
    identity_count: int = 0
    embedding_dim: int | None = None


@dataclass(frozen=True, slots=True)
class IdentityCandidate:
    """Store one ranked identity candidate for a fin crop."""

    identity: str
    score: float
    score_type: str


@dataclass(frozen=True, slots=True)
class IdentityPrediction:
    """Store the best identity and ranked candidates for a fin crop."""

    identity: str
    score: float
    score_type: str
    candidates: tuple[IdentityCandidate, ...] = ()


def _friendly_name(path: Path, prefix: str = "") -> str:
    stem = path.stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem.strip("_").replace("_", " ").strip().title() or path.stem


def discover_detection_models(directory: Path) -> tuple[list[DetectionModel], list[str]]:
    """Discover fin-detection checkpoints and return any warnings."""

    directory = Path(directory)
    if not directory.is_dir():
        return [], [f"Fin-recognition model folder is missing: {directory}"]
    models = [
        DetectionModel(_friendly_name(path, "model_"), path.resolve())
        for path in sorted(directory.glob("model_*.pt"))
        if path.is_file()
    ]
    warnings = [] if models else [f"No model_*.pt fin-recognition models found in {directory}"]
    return models, warnings


def _torch_load(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _gallery_for(checkpoint: Path) -> Path | None:
    matched = checkpoint.with_name(f"{checkpoint.stem}.gallery.pt")
    if matched.is_file():
        return matched
    if checkpoint.stem == "arc_face":
        legacy = checkpoint.with_name("gallery.pt")
        if legacy.is_file():
            return legacy
    return None


def _validate_gallery(
    gallery_path: Path,
    class_to_idx: dict[object, object],
    embedding_dim: int,
    loader: Callable[[Path], Any],
) -> int:
    import torch

    gallery = loader(gallery_path)
    identities = gallery.get("identities") if isinstance(gallery, dict) else None
    prototypes = gallery.get("prototypes") if isinstance(gallery, dict) else None
    gallery_mapping = gallery.get("class_to_idx") if isinstance(gallery, dict) else None
    normalized_mapping = {str(key): int(value) for key, value in class_to_idx.items()}
    if (
        not isinstance(identities, list)
        or not isinstance(prototypes, torch.Tensor)
        or prototypes.ndim != 2
        or prototypes.shape[1] != embedding_dim
        or len(identities) != prototypes.shape[0]
        or gallery_mapping != normalized_mapping
    ):
        raise ValueError("gallery identity mapping or prototype dimensions do not match")
    expected = [
        identity
        for identity, _index in sorted(normalized_mapping.items(), key=lambda item: item[1])
    ]
    if [str(value) for value in identities] != expected:
        raise ValueError("gallery identities are not in checkpoint class order")
    return len(identities)


def discover_identification_models(
    directory: Path,
    *,
    loader: Callable[[Path], Any] = _torch_load,
) -> tuple[list[IdentificationModel], list[str]]:
    """Discover supported checkpoints by their safe payload schemas.

    Parameters:
        directory: Directory containing identification checkpoints.
        loader: Safe checkpoint loader, injectable for tests.

    Returns:
        Valid model descriptors and user-facing warnings.
    """

    directory = Path(directory)
    if not directory.is_dir():
        return [], [f"Fin-identification model folder is missing: {directory}"]
    models: list[IdentificationModel] = []
    warnings: list[str] = []
    for path in sorted(directory.glob("*.pt")):
        try:
            payload = loader(path)
            if not isinstance(payload, dict):
                warnings.append(f"{path.name}: unsupported checkpoint payload")
                continue
            if "prototypes" in payload and "identities" in payload:
                continue
            if {"class_to_idx", "embedding_dim", "model_state_dict"}.issubset(payload):
                mapping = payload["class_to_idx"]
                embedding_dim = payload["embedding_dim"]
                if not isinstance(mapping, dict) or not mapping or not isinstance(embedding_dim, int):
                    raise ValueError("invalid ArcFace class mapping or embedding dimension")
                gallery_path = _gallery_for(path)
                if gallery_path is None:
                    raise ValueError(
                        f"missing {path.stem}.gallery.pt companion gallery"
                    )
                identity_count = _validate_gallery(
                    gallery_path, mapping, embedding_dim, loader
                )
                models.append(
                    IdentificationModel(
                        name=f"{_friendly_name(path)} (ArcFace)",
                        path=path.resolve(),
                        kind="arcface",
                        score_label="cosine similarity",
                        gallery_path=gallery_path.resolve(),
                        identity_count=identity_count,
                        embedding_dim=embedding_dim,
                    )
                )
            elif {"class_to_index", "model_name", "model_state_dict"}.issubset(payload):
                mapping = payload["class_to_index"]
                model_name = payload["model_name"]
                if not isinstance(mapping, dict) or not mapping or model_name not in {
                    "resnet18",
                    "resnet50",
                }:
                    raise ValueError("invalid or unsupported ResNet checkpoint")
                models.append(
                    IdentificationModel(
                        name=f"{_friendly_name(path)} ({model_name})",
                        path=path.resolve(),
                        kind="resnet",
                        score_label="probability",
                        identity_count=len(mapping),
                    )
                )
            else:
                warnings.append(f"{path.name}: unsupported identification checkpoint")
        except Exception as exc:
            warnings.append(f"{path.name}: {exc}")
        finally:
            try:
                del payload
            except UnboundLocalError:
                pass
            gc.collect()
    if not models:
        warnings.append(f"No usable identification checkpoints found in {directory}")
    return models, warnings


def _is_mps_oom(exc: BaseException) -> bool:
    try:
        from findetection_core import is_mps_out_of_memory

        return bool(is_mps_out_of_memory(exc))
    except ImportError:
        message = str(exc).lower()
        return "mps" in message and ("out of memory" in message or "allocation" in message)


class IdentifierOutOfMemory(RuntimeError):
    """Raised when identifier inference exhausts device memory."""

    pass


class IdentifierRuntime:
    """One loaded identifier with bounded batches and MPS OOM recovery."""

    def __init__(
        self,
        descriptor: IdentificationModel,
        batch_size: int,
        *,
        log: Callable[[str], None] | None = None,
        device_name: str = "mps",
        prefer_fp16: bool = True,
    ) -> None:
        import torch
        from torchvision import transforms

        self.torch = torch
        self.descriptor = descriptor
        self.device = torch.device(device_name)
        self.log = log or (lambda _message: None)
        self.effective_batch_size = batch_size
        self.precision = "FP32"
        checkpoint = _torch_load(descriptor.path)

        if descriptor.kind == "resnet":
            mapping = checkpoint["class_to_index"]
            self.identities = [
                identity
                for identity, _index in sorted(
                    ((str(key), int(value)) for key, value in mapping.items()),
                    key=lambda item: item[1],
                )
            ]
            image_size = int(checkpoint.get("image_size", 224))
            self.model = self._build_classifier(
                str(checkpoint["model_name"]),
                len(self.identities),
                float(checkpoint.get("dropout", 0.0)),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.prototypes = None
        elif descriptor.kind == "arcface":
            mapping = checkpoint["class_to_idx"]
            self.identities = [
                identity
                for identity, _index in sorted(
                    ((str(key), int(value)) for key, value in mapping.items()),
                    key=lambda item: item[1],
                )
            ]
            image_size = 224
            self.model = self._build_embedding_model(
                int(checkpoint["embedding_dim"]),
                float(checkpoint.get("dropout", 0.0)),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if descriptor.gallery_path is None:
                raise ValueError("ArcFace model has no gallery")
            gallery = _torch_load(descriptor.gallery_path)
            if [str(value) for value in gallery["identities"]] != self.identities:
                raise ValueError("ArcFace gallery identities do not match the checkpoint")
            self.prototypes = torch.nn.functional.normalize(
                gallery["prototypes"].float(), p=2, dim=1
            )
        else:
            raise ValueError(f"Unsupported identification model type: {descriptor.kind}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(round(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )
        self.image_size = image_size
        self.model.eval().to(self.device)
        if self.prototypes is not None:
            self.prototypes = self.prototypes.to(self.device)
        self._warm_precision(prefer_fp16)

    def _build_classifier(self, name: str, classes: int, dropout: float) -> Any:
        from torch import nn
        from torchvision.models import resnet18, resnet50

        builders = {"resnet18": resnet18, "resnet50": resnet50}
        model = builders[name](weights=None)
        features = model.fc.in_features
        classifier = nn.Linear(features, classes)
        model.fc = nn.Sequential(nn.Dropout(dropout), classifier) if dropout > 0 else classifier
        return model

    def _build_embedding_model(self, dimension: int, dropout: float) -> Any:
        torch = self.torch
        nn = torch.nn
        functional = torch.nn.functional
        from torchvision.models import resnet18

        class EmbeddingModel(nn.Module):
            """ResNet-backed normalized embedding network."""

            def __init__(self) -> None:
                super().__init__()
                backbone = resnet18(weights=None)
                features = backbone.fc.in_features
                backbone.fc = nn.Identity()
                self.backbone = backbone
                self.embedding = nn.Linear(features, dimension)
                self.batch_norm = nn.BatchNorm1d(dimension)
                self.dropout = nn.Dropout(dropout)
                self.embedding_dim = dimension
                self.dropout_probability = dropout

            def forward(self, images: Any) -> Any:
                """Return normalized embeddings for an image batch."""

                values = self.backbone(images)
                values = self.dropout(self.batch_norm(self.embedding(values)))
                return functional.normalize(values, p=2, dim=1)

        return EmbeddingModel()

    def _warm_precision(self, prefer_fp16: bool) -> None:
        torch = self.torch
        size = self.image_size
        device_label = "MPS" if self.device.type == "mps" else self.device.type.upper()
        if prefer_fp16:
            try:
                self.model.half()
                if self.prototypes is not None:
                    self.prototypes = self.prototypes.half()
                with torch.inference_mode():
                    self.model(torch.zeros((1, 3, size, size), device=self.device, dtype=torch.float16))
                if self.device.type == "mps":
                    torch.mps.synchronize()
                self.precision = "FP16"
                self.log(f"Identification model ready on {device_label} using FP16.")
                return
            except RuntimeError as exc:
                if _is_mps_oom(exc):
                    raise IdentifierOutOfMemory(
                        "The identification model could not warm up in available MPS memory."
                    ) from exc
                self.log(
                    f"Identifier FP16 is unsupported ({exc}); using FP32 on {device_label}."
                )
                self._empty_cache()
        self.model.float()
        if self.prototypes is not None:
            self.prototypes = self.prototypes.float()
        with torch.inference_mode():
            self.model(torch.zeros((1, 3, size, size), device=self.device, dtype=torch.float32))
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.precision = "FP32"
        self.log(f"Identification model ready on {device_label} using FP32.")

    @property
    def dtype(self) -> Any:
        """Return the torch dtype selected by the active precision."""

        return self.torch.float16 if self.precision == "FP16" else self.torch.float32

    def predict(self, crops: Sequence[Any]) -> list[IdentityPrediction]:
        """Identify fin crops in bounded, recoverable batches."""

        predictions: list[IdentityPrediction] = []
        offset = 0
        while offset < len(crops):
            attempt_size = min(self.effective_batch_size, len(crops) - offset)
            try:
                predictions.extend(
                    self._predict_attempt(crops[offset : offset + attempt_size])
                )
                offset += attempt_size
            except RuntimeError as exc:
                if not _is_mps_oom(exc):
                    raise
                if self.effective_batch_size == 1:
                    raise IdentifierOutOfMemory(
                        "Identification ran out of MPS memory at batch size 1. "
                        "Close other GPU-heavy apps or lower detector image size."
                    ) from exc
                previous = self.effective_batch_size
                self.effective_batch_size = max(1, previous // 2)
                self.log(
                    f"MPS memory pressure during identification at batch {previous}; "
                    f"retrying with batch {self.effective_batch_size}."
                )
                self._empty_cache()
        return predictions

    def _predict_attempt(self, crops: Sequence[Any]) -> list[IdentityPrediction]:
        """Return the three highest-scoring identities for each crop.

        Parameters:
            crops: FinSaddle crops in one bounded inference attempt.

        Returns:
            One best prediction per crop, carrying up to three ranked candidates.
        """
        tensors = [self.transform(crop) for crop in crops]
        batch = self.torch.stack(tensors).to(self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            values = self.model(batch)
            if self.descriptor.kind == "resnet":
                candidate_scores = values.softmax(dim=1)
            else:
                candidate_scores = values @ self.prototypes.T
            candidate_count = min(3, len(self.identities))
            scores, indexes = candidate_scores.topk(candidate_count, dim=1)
        scores = scores.float().cpu()
        indexes = indexes.cpu()
        output: list[IdentityPrediction] = []
        for crop_scores, crop_indexes in zip(scores, indexes):
            candidates = tuple(
                IdentityCandidate(
                    identity=self.identities[int(index)],
                    score=float(score),
                    score_type=self.descriptor.score_label,
                )
                for score, index in zip(crop_scores, crop_indexes)
            )
            best = candidates[0]
            output.append(
                IdentityPrediction(
                    identity=best.identity,
                    score=best.score,
                    score_type=best.score_type,
                    candidates=candidates,
                )
            )
        del batch, values, candidate_scores, scores, indexes, tensors
        return output

    def _empty_cache(self) -> None:
        gc.collect()
        if self.device.type == "mps":
            try:
                self.torch.mps.empty_cache()
            except RuntimeError:
                pass

    def close(self) -> None:
        """Release model, prototypes, and cached device allocations."""

        self.model = None
        self.prototypes = None
        self._empty_cache()
