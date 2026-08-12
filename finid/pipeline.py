"""Bounded fin detection and identification pipeline."""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image
from findetection_core import partition_detections

from .models import (
    DetectionModel,
    IdentificationModel,
    IdentifierOutOfMemory,
    IdentifierRuntime,
)
from .reporting import ReportMetadata, is_generated_report, report_filename, write_reports
from .storage import ResultStore


JPEG_EXTENSIONS = {".jpg", ".jpeg"}
LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class BatchRecommendation:
    """Recommended detector and identifier batches for available memory."""

    memory_gib: float
    detector_batch: int
    identifier_batch: int


def total_memory_bytes() -> int:
    """Return physical memory in bytes, or a conservative fallback."""

    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 16 * 1024**3


def recommended_batches(memory_bytes: int | None = None) -> BatchRecommendation:
    """Choose bounded batch sizes for the available memory.

    Parameters:
        memory_bytes: Physical memory in bytes; detected when omitted.

    Returns:
        Detector and identifier batch recommendations.
    """

    gib = (memory_bytes if memory_bytes is not None else total_memory_bytes()) / 1024**3
    if gib <= 16.5:
        return BatchRecommendation(gib, 2, 8)
    if gib <= 32.5:
        return BatchRecommendation(gib, 4, 16)
    if gib <= 64.5:
        return BatchRecommendation(gib, 8, 32)
    return BatchRecommendation(gib, 12, 64)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configure a fin-identification run over one image tree.

    Parameters:
        input_dir: Root containing JPEG images.
        detector: Detection model descriptor.
        identifier: Identification model descriptor.
        threshold: Minimum accepted identification score.
        detector_confidence: Minimum accepted detection confidence.
        detector_image_size: Square detector input size in pixels.
        detector_batch_size: Maximum images per detection batch.
        identifier_batch_size: Maximum crops per identification batch.
        crop_padding: Fractional padding added around selected detections.
        detector_fp16: Whether to request half-precision MPS detection.
        max_detections: Maximum detections retained per source image.
        selected_class_ids: Classes passed to identification, or ``None`` for all.
    """

    input_dir: Path
    detector: DetectionModel
    identifier: IdentificationModel
    threshold: float = 0.5
    detector_confidence: float = 0.25
    detector_image_size: int = 1280
    detector_batch_size: int = 2
    identifier_batch_size: int = 8
    crop_padding: float = 0.0
    detector_fp16: bool = True
    max_detections: int = 20
    selected_class_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir).expanduser().resolve())
        if not 0 <= self.threshold <= 1:
            raise ValueError("Identification threshold must be between 0 and 1.")
        if not 0 <= self.detector_confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1.")
        if self.detector_image_size < 32:
            raise ValueError("Detector image size must be at least 32.")
        if self.detector_batch_size < 1 or self.identifier_batch_size < 1:
            raise ValueError("Batch sizes must be at least 1.")
        if self.crop_padding < 0:
            raise ValueError("Crop padding cannot be negative.")
        if self.selected_class_ids is not None:
            selected_ids = tuple(
                dict.fromkeys(int(value) for value in self.selected_class_ids)
            )
            if not selected_ids:
                raise ValueError("Select at least one object class to identify.")
            object.__setattr__(self, "selected_class_ids", selected_ids)


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """Summarize completed work, reports, performance, and errors."""

    completed: bool
    stopped: bool
    processed: int
    total: int
    report_count: int
    root_report: Path
    elapsed_seconds: float
    detector_batch_size: int
    identifier_batch_size: int
    error: str | None = None


def _walk_directories(root: Path) -> Iterator[tuple[Path, list[str]]]:
    for current, child_dirs, filenames in os.walk(root, followlinks=False):
        current_path = Path(current).resolve()
        child_dirs[:] = sorted(
            name
            for name in child_dirs
            if not (Path(current) / name).is_symlink()
        )
        yield current_path, sorted(filenames, key=lambda value: (value.casefold(), value))


def inventory_tree(
    root: Path,
    store: ResultStore,
    *,
    scan_progress: Callable[[int, int], None] | None = None,
    yield_every: int = 250,
) -> int:
    """Inventory supported files and directories into a result store.

    Parameters:
        root: Root directory to scan.
        store: Destination for discovered paths.
        scan_progress: Optional callback receiving file and directory counts.
        yield_every: Number of files between brief scheduler yields.

    Returns:
        Number of JPEG images discovered.
    """

    root = root.resolve()
    scanned_files = 0
    scanned_directories = 0
    for directory, filenames in _walk_directories(root):
        scanned_directories += 1
        parent = directory.parent if directory != root else None
        store.add_directory(directory, parent)
        for filename in filenames:
            scanned_files += 1
            path = directory / filename
            if filename == ".DS_Store" or is_generated_report(path):
                pass
            elif path.is_symlink():
                store.add_skipped(directory, filename)
            elif path.suffix.lower() in JPEG_EXTENSIONS:
                store.add_image(path)
            else:
                store.add_skipped(directory, filename)
            if yield_every > 0 and scanned_files % yield_every == 0:
                if scan_progress is not None:
                    scan_progress(scanned_files, scanned_directories)
                # Directory scans are Python/SQLite heavy. Yielding briefly
                # keeps Tk's main thread responsive on very large trees.
                time.sleep(0.001)
        if scan_progress is not None:
            scan_progress(scanned_files, scanned_directories)
    store.finish_inventory()
    return store.total_images()


def iter_jpegs(root: Path) -> Iterator[Path]:
    """Yield non-symlink JPEG paths below a root in deterministic order."""

    for directory, filenames in _walk_directories(root.resolve()):
        for filename in filenames:
            path = directory / filename
            if not path.is_symlink() and path.suffix.lower() in JPEG_EXTENSIONS:
                yield path


def preflight_writable(directories: Iterator[Path]) -> None:
    """Raise when reports cannot be written to any supplied directory."""

    unwritable: list[Path] = []
    unwritable_count = 0
    for path in directories:
        report = path / report_filename(path)
        if not os.access(path, os.W_OK) or (
            report.exists() and not os.access(report, os.W_OK)
        ):
            unwritable_count += 1
            if len(unwritable) < 10:
                unwritable.append(path)
    if unwritable_count:
        preview = "\n".join(f"• {path}" for path in unwritable)
        remainder = unwritable_count - len(unwritable)
        if remainder:
            preview += f"\n• …and {remainder} more"
        raise PermissionError(
            "Reports cannot be written in these folders:\n" + preview
        )


def _result_image(result: Any) -> Image.Image:
    payload = getattr(result, "image", None)
    if isinstance(payload, Image.Image):
        return payload.convert("RGB").copy()
    if payload is not None and hasattr(payload, "shape"):
        if len(payload.shape) != 3 or payload.shape[2] < 3:
            raise ValueError(f"Unsupported decoded image shape: {payload.shape}")
        return Image.fromarray(payload[:, :, :3][:, :, ::-1].copy(), mode="RGB")
    with Image.open(result.path) as source:
        return source.convert("RGB").copy()


def _crop_box(
    image: Image.Image,
    box: Any,
    padding: float,
) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    x1, y1, x2, y2 = (float(value) for value in box.xyxy)
    pad_x = max(0.0, x2 - x1) * padding
    pad_y = max(0.0, y2 - y1) * padding
    left = max(0, min(image.width, math.floor(x1 - pad_x)))
    top = max(0, min(image.height, math.floor(y1 - pad_y)))
    right = max(0, min(image.width, math.ceil(x2 + pad_x)))
    bottom = max(0, min(image.height, math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    coordinates = left, top, right, bottom
    return image.crop(coordinates), coordinates


def _process_result(
    result: Any,
    identifier: Any,
    threshold: float,
    crop_padding: float,
    selected_class_ids: tuple[int, ...] | None,
    detector_confidence: float,
) -> tuple[int, list[dict[str, object]]]:
    """Crop and identify accepted detections from selected model classes.

    Parameters:
        result: Normalized result returned by FinDetection Core.
        identifier: Loaded crop-identification runtime.
        threshold: Minimum accepted identification score.
        crop_padding: Fractional padding around each selected detection.
        selected_class_ids: Classes allowed to produce identification crops.
        detector_confidence: Minimum accepted detection confidence.

    Returns:
        Selected detection count and accepted identification records.
    """

    partitions = partition_detections(
        result.boxes,
        selected_class_ids,
        detector_confidence,
    )
    selected_boxes = partitions.accepted_selected
    if not selected_boxes:
        return 0, []
    image = _result_image(result)
    crops: list[Image.Image] = []
    crop_rows: list[tuple[Any, tuple[int, int, int, int]]] = []
    try:
        for box in selected_boxes:
            cropped = _crop_box(image, box, crop_padding)
            if cropped is None:
                continue
            crop, coordinates = cropped
            crops.append(crop)
            crop_rows.append((box, coordinates))
        if not crops:
            return len(selected_boxes), []
        predictions = identifier.predict(crops)
        accepted: list[dict[str, object]] = []
        for prediction, (box, coordinates) in zip(predictions, crop_rows):
            if prediction.score < threshold:
                continue
            left, top, right, bottom = coordinates
            accepted.append(
                {
                    "identity": prediction.identity,
                    "score": prediction.score,
                    "score_type": prediction.score_type,
                    "detection_confidence": float(box.confidence),
                    "x1": left,
                    "y1": top,
                    "x2": right,
                    "y2": bottom,
                }
            )
        return len(selected_boxes), accepted
    finally:
        for crop in crops:
            crop.close()
        image.close()


def run_pipeline(
    config: PipelineConfig,
    *,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    runtime: Any | None = None,
    identifier_runtime: Any | None = None,
    probe: Callable[[], tuple[bool, str]] | None = None,
) -> PipelineSummary:
    """Run bounded local detection/identification and always write reports.

    Parameters:
        config: Validated pipeline configuration.
        log: Optional user-facing log callback.
        progress: Optional progress callback.
        stop_event: Optional cooperative cancellation event.
        runtime: Optional injected detector runtime.
        identifier_runtime: Optional injected identification runtime.
        probe: Optional injected device readiness probe.

    Returns:
        Summary of completed work and any error.
    """

    emit_log = log or (lambda _message: None)
    emit_progress = progress or (lambda _current, _total, _label: None)
    stop_event = stop_event or threading.Event()
    started = time.monotonic()
    completed = False
    stopped = False
    error: str | None = None
    report_count = 0
    total = 0
    processed = 0
    detector_batch = config.detector_batch_size
    identifier_batch = config.identifier_batch_size
    root_report = config.input_dir / report_filename(config.input_dir)
    detector_runtime = runtime
    active_identifier = identifier_runtime
    store = ResultStore()
    try:
        if not config.input_dir.is_dir():
            raise ValueError(f"Input folder does not exist: {config.input_dir}")
        emit_log("Scanning folders and JPEG filenames…")
        total = inventory_tree(
            config.input_dir,
            store,
            scan_progress=lambda files, directories: emit_progress(
                files,
                0,
                f"Scanning folders… {files:,} files in {directories:,} folders",
            ),
        )
        preflight_writable((Path(row["path"]) for row in store.directories()))
        directory_count = store.directory_count()
        emit_log(f"Found {total} JPEG images in {directory_count} folders.")
        if total == 0:
            completed = True
            emit_log("No JPEG images were found; status reports will still be created.")
        else:
            if probe is None:
                from findetection_core import probe_runtime

                probe = probe_runtime
            available, detail = probe()
            if not available:
                raise RuntimeError(detail)
            emit_log(detail)

            if detector_runtime is None:
                from findetection_core import MPSModelRuntime

                detector_runtime = MPSModelRuntime()
            from findetection_core import (
                MPSDetectorConfig,
                MPSPredictionStopped,
            )

            detector_config = MPSDetectorConfig(
                model_path=config.detector.path,
                image_size=config.detector_image_size,
                batch_size=config.detector_batch_size,
                confidence=config.detector_confidence,
                max_detections=config.max_detections,
                use_fp16=config.detector_fp16,
                source_window_size=config.detector_batch_size,
            )
            detector_runtime.load(detector_config, log=emit_log)
            device_name = str(getattr(detector_runtime, "device", "mps"))
            if active_identifier is None:
                active_identifier = IdentifierRuntime(
                    config.identifier,
                    config.identifier_batch_size,
                    log=emit_log,
                    device_name=device_name,
                    prefer_fp16=device_name == "mps",
                )
            emit_log(
                f"Starting {device_name.upper()} inference with detector batch "
                f"{config.detector_batch_size} "
                f"and identifier batch {config.identifier_batch_size}."
            )

            try:
                results = detector_runtime.predict_paths(
                    iter_jpegs(config.input_dir),
                    stop_event=stop_event,
                )
                for result in results:
                    if stop_event.is_set():
                        stopped = True
                        break
                    image_path = Path(result.path).resolve()
                    try:
                        detected, accepted = _process_result(
                            result,
                            active_identifier,
                            config.threshold,
                            config.crop_padding,
                            config.selected_class_ids,
                            config.detector_confidence,
                        )
                        store.record_result(image_path, detected, accepted)
                    except IdentifierOutOfMemory:
                        raise
                    except Exception as exc:
                        store.record_failure(image_path, str(exc))
                        emit_log(f"Could not process {image_path.name}: {exc}")
                    processed = store.processed_images()
                    elapsed = time.monotonic() - started
                    rate = processed / elapsed if elapsed else 0.0
                    remaining = max(0, total - processed)
                    eta = remaining / rate if rate else 0.0
                    emit_progress(
                        processed,
                        total,
                        f"{processed}/{total} · {rate:.2f} images/s · about {eta / 60:.1f} min left",
                    )
            except MPSPredictionStopped:
                stopped = True

            if stop_event.is_set():
                stopped = True
            if not stopped:
                missing = store.mark_pending_failed("Image could not be decoded by the detector.")
                if missing:
                    emit_log(f"{missing} JPEG images could not be decoded.")
                processed = store.processed_images()
                completed = True
            detector_batch = int(
                getattr(detector_runtime, "effective_batch_size", config.detector_batch_size)
            )
            identifier_batch = int(
                getattr(active_identifier, "effective_batch_size", config.identifier_batch_size)
            )
    except Exception as exc:
        error = str(exc)
        emit_log(f"Pipeline stopped with an error: {exc}")
    finally:
        try:
            if detector_runtime is not None:
                try:
                    detector_runtime.synchronize()
                    memory = detector_runtime.memory_description()
                    if memory:
                        emit_log(memory)
                except Exception:
                    pass
                detector_runtime.close()
            if active_identifier is not None:
                active_identifier.close()
        finally:
            elapsed = time.monotonic() - started
            processed = store.processed_images()
            throughput = processed / elapsed if elapsed else 0.0
            message = (
                error
                if error
                else (
                    "The run was stopped. Results shown are for completed images only."
                    if stopped
                    else ""
                )
            )
            try:
                report_count = write_reports(
                    store,
                    ReportMetadata(
                        generated_at=datetime.now().astimezone(),
                        completed=completed,
                        detector_name=config.detector.name,
                        identifier_name=config.identifier.name,
                        threshold=config.threshold,
                        score_label=config.identifier.score_label,
                        elapsed_seconds=elapsed,
                        throughput=throughput,
                        message=message,
                    ),
                )
                emit_log(f"Wrote {report_count} folder reports.")
            except Exception as report_exc:
                if error:
                    error = f"{error}; report error: {report_exc}"
                else:
                    error = f"Could not write reports: {report_exc}"
                completed = False
                emit_log(error)
            store.close()

    return PipelineSummary(
        completed=completed,
        stopped=stopped,
        processed=processed,
        total=total,
        report_count=report_count,
        root_report=root_report,
        elapsed_seconds=time.monotonic() - started,
        detector_batch_size=detector_batch,
        identifier_batch_size=identifier_batch,
        error=error,
    )
