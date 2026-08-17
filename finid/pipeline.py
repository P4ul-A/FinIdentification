"""Encounter-aware, disk-backed fin detection and image clustering pipeline."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from PIL import Image

from .models import DetectionModel, IdentificationModel, IdentifierRuntime
from .reporting import (
    ReportMetadata,
    is_generated_report_assets,
    report_filename,
    write_reports,
)
from .storage import ResultStore


JPEG_EXTENSIONS = {".jpg", ".jpeg"}
DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")
ENCOUNTER_PREFIX = re.compile(r"^ENCOUNTER", re.IGNORECASE)
MANAGEMENT_MARKER = ".finid-managed"
LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str], None]
PROGRESS_SCALE = 1000


def _phase_progress(
    emit: ProgressCallback,
    start: int,
    end: int,
) -> ProgressCallback:
    """Map phase-local progress onto one monotonic whole-run scale.

    Parameters:
        emit: Whole-run progress callback.
        start: Inclusive starting unit on the whole-run scale.
        end: Inclusive ending unit on the whole-run scale.

    Returns:
        A callback accepting current units, total units, and a status label.
    """

    def publish(current: int, total: int, label: str) -> None:
        fraction = min(1.0, max(0.0, current / total)) if total > 0 else 0.0
        overall = round(start + (end - start) * fraction)
        emit(overall, PROGRESS_SCALE, label)

    return publish


@dataclass(frozen=True, slots=True)
class BatchRecommendation:
    """Recommended detector and identifier batches for available memory."""

    memory_gib: float
    detector_batch: int
    identifier_batch: int


@dataclass(frozen=True, slots=True)
class Encounter:
    """One source encounter and its lazy recursively owned JPEG collection."""

    root: Path
    relative_path: Path
    images: "EncounterImages"


class EncounterImages(Collection[Path]):
    """Re-iterable, constant-memory view of one encounter's source JPEGs."""

    def __init__(self, root: Path, encounter_roots: Collection[Path]) -> None:
        self.root = root
        self.encounter_roots = (
            encounter_roots
            if isinstance(encounter_roots, frozenset)
            else frozenset(encounter_roots)
        )

    def __iter__(self) -> Iterator[Path]:
        """Yield recursively owned JPEGs without retaining their paths."""
        for path in _iter_files(self.root):
            if path.suffix.lower() in JPEG_EXTENSIONS and _owner(path, self.encounter_roots) == self.root:
                yield path

    def __len__(self) -> int:
        """Return the image count using a constant-memory scan."""
        return sum(1 for _path in self)

    def __contains__(self, value: object) -> bool:
        """Return whether a path is a JPEG owned by this encounter."""
        if not isinstance(value, Path):
            return False
        path = value.expanduser().resolve()
        return (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in JPEG_EXTENSIONS
            and _owner(path, self.encounter_roots) == self.root
        )


def total_memory_bytes() -> int:
    """Return physical memory in bytes, or a conservative fallback."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 16 * 1024**3


def recommended_batches(memory_bytes: int | None = None) -> BatchRecommendation:
    """Choose bounded batch sizes for the available memory."""
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
    """Configure encounter discovery, inference, reporting, and clustering.

    Parameters:
        input_dir: Source image tree.
        detector: Detection model descriptor.
        identifier: Identification model descriptor.
        threshold: Minimum identification score.
        detector_confidence: Fin/FinSaddle confidence threshold.
        eye_confidence: Eye confidence threshold.
        clustering: Whether original JPEGs are copied into category folders.
        output_root: Empty or app-managed clustering output root.
        detector_classes: Detector class metadata with ``class_id`` and ``name``.
        selected_class_ids: FinSaddle classes eligible for identification.

    Returns:
        A validated immutable configuration.
    """

    input_dir: Path
    detector: DetectionModel
    identifier: IdentificationModel
    threshold: float = 0.5
    detector_confidence: float = 0.25
    eye_confidence: float = 0.25
    clustering: bool = False
    output_root: Path | None = None
    detector_classes: tuple[Any, ...] = ()
    detector_image_size: int = 1280
    detector_batch_size: int = 2
    identifier_batch_size: int = 8
    crop_padding: float = 0.0
    detector_fp16: bool = True
    max_detections: int = 20
    selected_class_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir).expanduser().resolve())
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).expanduser().resolve())
        for label, value in (("Identification", self.threshold), ("Fin/FinSaddle", self.detector_confidence), ("Eye", self.eye_confidence)):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} confidence must be between 0 and 1.")
        if self.clustering and self.output_root is None:
            raise ValueError("Choose an output folder when clustering is enabled.")
        if self.detector_image_size < 32:
            raise ValueError("Detector image size must be at least 32.")
        if self.detector_batch_size < 1 or self.identifier_batch_size < 1:
            raise ValueError("Batch sizes must be at least 1.")
        if self.crop_padding < 0:
            raise ValueError("Crop padding cannot be negative.")
        if self.selected_class_ids is not None:
            values = tuple(dict.fromkeys(int(value) for value in self.selected_class_ids))
            if not values:
                raise ValueError("Select at least one object class to identify.")
            object.__setattr__(self, "selected_class_ids", values)


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """Summarize run totals, encounter reports, output, and errors."""

    completed: bool
    stopped: bool
    processed: int
    total: int
    encounter_count: int
    skipped_undated_count: int
    clustered_image_count: int
    report_count: int
    reports_root: Path
    root_report: Path
    elapsed_seconds: float
    detector_batch_size: int
    identifier_batch_size: int
    error: str | None = None


def _iter_tree(root: Path) -> Iterator[tuple[Path, Path | None]]:
    """Yield directory markers and files with memory proportional to tree depth."""
    root = root.resolve()
    stack: list[tuple[Path, os.ScandirIterator[str]]] = []
    yield root, None
    stack.append((root, os.scandir(root)))
    try:
        while stack:
            directory, entries = stack[-1]
            try:
                entry = next(entries)
            except StopIteration:
                entries.close()
                stack.pop()
                continue
            try:
                if entry.is_symlink():
                    continue
                path = Path(entry.path).resolve()
                if entry.is_dir(follow_symlinks=False):
                    if is_generated_report_assets(path):
                        continue
                    yield path, None
                    stack.append((path, os.scandir(path)))
                elif entry.is_file(follow_symlinks=False):
                    yield directory, path
            except OSError:
                continue
    finally:
        for _directory, entries in stack:
            entries.close()


def _iter_files(root: Path) -> Iterator[Path]:
    """Yield regular files recursively without materializing directory contents."""
    for _directory, path in _iter_tree(root):
        if path is not None:
            yield path


def _resolve_encounter_roots(root: Path, store: ResultStore) -> list[Path]:
    """Resolve encounter roots from the disk-backed single-pass inventory."""
    base_roots: list[Path] = []
    for row in store.scanned_directories():
        directory = Path(str(row["path"]))
        if not DATE_PREFIX.match(str(row["name"])):
            continue
        explicit = [
            Path(str(child["path"]))
            for child in store.scanned_children(directory)
            if ENCOUNTER_PREFIX.match(str(child["name"]))
        ]
        child_count = int(row["child_count"])
        if not int(row["direct_jpeg_count"]) and child_count and len(explicit) == child_count:
            base_roots.extend(explicit)
        else:
            base_roots.append(directory)

    groups = [
        group
        for group in store.group_sibling_directories()
        if any(group == base or base in group.parents for base in base_roots)
    ]
    candidates = set([*base_roots, *groups])
    owned_counts = dict.fromkeys(candidates, 0)
    for row in store.scanned_images():
        owner = _owner(Path(str(row["path"])), candidates)
        if owner is not None:
            owned_counts[owner] += 1
    roots = [
        candidate
        for candidate in candidates
        if owned_counts[candidate] > 0
        or not any(group != candidate and candidate in group.parents for group in groups)
    ]
    return sorted(roots, key=lambda path: (len(path.parts), str(path).casefold(), str(path)))


def _owner(path: Path, roots: Collection[Path]) -> Path | None:
    """Return the deepest encounter root containing a path."""
    candidate = path if path.is_dir() else path.parent
    while True:
        if candidate in roots:
            return candidate
        if candidate == candidate.parent:
            return None
        candidate = candidate.parent


def _mirrored_relative(root: Path, encounter: Path) -> Path:
    """Return a non-empty encounter path beneath an output root."""
    relative = encounter.relative_to(root)
    return Path(encounter.name) if relative == Path(".") else relative


def discover_encounters(root: Path) -> tuple[Encounter, ...]:
    """Return encounter roots and recursively owned source JPEG sets.

    Parameters:
        root: Selected source tree or a dated encounter root.

    Returns:
        Deterministically ordered encounters. Undated JPEGs are omitted.
    """
    root = Path(root).expanduser().resolve()
    with ResultStore() as store:
        inventory_tree(root, store)
        encounter_roots = [Path(str(row["path"])) for row in store.encounters()]
    roots = frozenset(encounter_roots)
    return tuple(
        Encounter(
            encounter,
            _mirrored_relative(root, encounter),
            EncounterImages(encounter, roots),
        )
        for encounter in encounter_roots
    )


def inventory_tree(
    root: Path, store: ResultStore, *,
    scan_progress: Callable[[int, int], None] | None = None, yield_every: int = 250,
) -> int:
    """Inventory encounters and JPEG ownership directly into SQLite."""
    root = root.resolve()
    files = directories = 0
    for directory, path in _iter_tree(root):
        if path is None:
            directories += 1
            store.add_scanned_directory(
                directory,
                None if directory == root else directory.parent,
            )
            if scan_progress:
                scan_progress(files, directories)
            continue
        files += 1
        if path.suffix.lower() in JPEG_EXTENSIONS:
            store.add_scanned_image(path)
        if yield_every > 0 and files % yield_every == 0:
            if scan_progress:
                scan_progress(files, directories)
            time.sleep(0.001)
    if scan_progress:
        scan_progress(files, directories)
    store.finish_inventory()
    encounter_roots = _resolve_encounter_roots(root, store)
    roots = set(encounter_roots)
    for encounter in encounter_roots:
        store.add_encounter(encounter, _mirrored_relative(root, encounter))
    for row in store.scanned_images():
        path = Path(str(row["path"]))
        encounter = _owner(path, roots)
        if encounter is None:
            store.add_skipped_undated(path)
        else:
            store.add_image(path, encounter)
    store.discard_scan_state()
    return store.total_images()


def iter_jpegs(root: Path) -> Iterator[Path]:
    """Yield JPEGs deterministically using a disk-backed path sort."""
    descriptor, database_name = tempfile.mkstemp(prefix="finid-paths-", suffix=".sqlite3")
    os.close(descriptor)
    connection = sqlite3.connect(database_name)
    try:
        connection.executescript(
            """PRAGMA temp_store=FILE;
               CREATE TABLE paths(path TEXT PRIMARY KEY);"""
        )
        for path in _iter_files(root.resolve()):
            if path.suffix.lower() in JPEG_EXTENSIONS:
                connection.execute("INSERT INTO paths(path) VALUES (?)", (str(path),))
        connection.commit()
        for row in connection.execute("SELECT path FROM paths ORDER BY path COLLATE NOCASE, path"):
            yield Path(row[0])
    finally:
        connection.close()
        Path(database_name).unlink(missing_ok=True)


def _class_map(config: PipelineConfig) -> dict[int, str]:
    """Normalize detector class metadata to numeric ID and machine name."""
    mapping: dict[int, str] = {}
    for item in config.detector_classes:
        class_id = int(getattr(item, "class_id", getattr(item, "id", -1)))
        name = str(
            getattr(
                item,
                "name",
                getattr(item, "raw_name", getattr(item, "model_name", "")),
            )
        )
        mapping[class_id] = name
    if not mapping and config.selected_class_ids is not None:
        mapping.update({class_id: "fin_left" for class_id in config.selected_class_ids})
    return mapping


def _kind_side(name: str) -> tuple[str, str | None]:
    """Normalize one model class into fin, FinSaddle, eye, or other and side."""
    normalized = name.replace("-", "_").replace(" ", "_").casefold()
    side = "LEFT" if normalized.endswith("_left") else "RIGHT" if normalized.endswith("_right") else None
    base = normalized.rsplit("_", 1)[0] if side else normalized
    if base in {"finsaddle", "fin_saddle", "saddle"}:
        return "finsaddle", side
    if base == "fin":
        return "fin", side
    if base == "eye":
        return "eye", side
    return "other", side


def detector_has_finsaddle_classes(classes: Sequence[Any]) -> bool:
    """Return whether detector metadata contains a FinSaddle class.

    Parameters:
        classes: Detector class metadata entries.

    Returns:
        ``True`` when at least one class is recognized as FinSaddle or saddle.
    """
    for item in classes:
        name = str(
            getattr(
                item,
                "name",
                getattr(item, "raw_name", getattr(item, "model_name", "")),
            )
        )
        if _kind_side(name)[0] == "finsaddle":
            return True
    return False


def _result_image(result: Any) -> Image.Image:
    """Return an RGB PIL image from a normalized detector result."""
    payload = getattr(result, "image", None)
    if isinstance(payload, Image.Image):
        return payload.convert("RGB").copy()
    if payload is not None and hasattr(payload, "shape"):
        return Image.fromarray(payload[:, :, :3][:, :, ::-1].copy(), mode="RGB")
    with Image.open(result.path) as source:
        return source.convert("RGB").copy()


def _crop_box(image: Image.Image, values: Sequence[float], padding: float) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    """Crop a padded box and return its clamped coordinates."""
    x1, y1, x2, y2 = (float(value) for value in values)
    pad_x, pad_y = max(0.0, x2 - x1) * padding, max(0.0, y2 - y1) * padding
    coordinates = (
        max(0, min(image.width, math.floor(x1 - pad_x))),
        max(0, min(image.height, math.floor(y1 - pad_y))),
        max(0, min(image.width, math.ceil(x2 + pad_x))),
        max(0, min(image.height, math.ceil(y2 + pad_y))),
    )
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        return None
    return image.crop(coordinates), coordinates


def _winning_side(sides: Sequence[str | None]) -> str | None:
    """Choose LEFT on a bilateral conflict, otherwise the qualifying side."""
    return "LEFT" if "LEFT" in sides else "RIGHT" if "RIGHT" in sides else None


@dataclass(slots=True, eq=False)
class _PreparedImage:
    """Hold one image's bounded state until its queued crops are identified."""

    path: Path
    detections: list[dict[str, object]]
    pending_crops: int
    identities: list[dict[str, object]] = field(default_factory=list)


def _prepare_result(
    result: Any,
    config: PipelineConfig,
    names: dict[int, str],
    selected: set[int],
) -> tuple[_PreparedImage, list[tuple[Image.Image, int]]]:
    """Extract FinSaddle crops without running identification yet."""
    detections: list[dict[str, object]] = []
    candidate_indices: list[int] = []
    for box in result.boxes:
        class_id = int(box.class_id)
        name = names.get(class_id, f"class_{class_id}")
        kind, side = _kind_side(name)
        confidence = float(box.confidence)
        x1, y1, x2, y2 = (int(round(float(value))) for value in box.xyxy)
        is_candidate = (
            class_id in selected
            and kind == "finsaddle"
            and confidence >= config.detector_confidence
        )
        detections.append(
            {
                "class_id": class_id,
                "class_name": name,
                "kind": kind,
                "side": side,
                "confidence": confidence,
                "selected": is_candidate,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
        if is_candidate:
            candidate_indices.append(len(detections) - 1)
    crops: list[tuple[Image.Image, int]] = []
    if candidate_indices:
        image = _result_image(result)
        try:
            for index in candidate_indices:
                item = detections[index]
                cropped = _crop_box(
                    image,
                    (item["x1"], item["y1"], item["x2"], item["y2"]),
                    config.crop_padding,
                )
                if cropped is not None:
                    crops.append((cropped[0], index))
        except Exception:
            for crop, _index in crops:
                crop.close()
            raise
        finally:
            image.close()
    return _PreparedImage(Path(result.path).resolve(), detections, len(crops)), crops


def _classification(
    state: _PreparedImage,
    config: PipelineConfig,
) -> tuple[str, str | None]:
    """Return the single category and LEFT-preferring side for an image."""
    if state.identities:
        sides = [
            state.detections[int(item["detection_index"])]["side"]
            for item in state.identities
        ]
        return "IDed", _winning_side(sides)
    saddle_sides = [
        item["side"]
        for item in state.detections
        if item["kind"] == "finsaddle"
        and float(item["confidence"]) >= config.detector_confidence
    ]
    if saddle_sides:
        return "FinSaddle", _winning_side(saddle_sides)
    eye_sides = [
        item["side"]
        for item in state.detections
        if item["kind"] == "eye"
        and float(item["confidence"]) >= config.eye_confidence
    ]
    if eye_sides:
        return "Eyes", _winning_side(eye_sides)
    return "Rest", None


class _IdentificationBatcher:
    """Batch FinSaddle crops across images while keeping a strict RAM bound."""

    def __init__(self, identifier: Any, threshold: float, batch_size: int) -> None:
        self.identifier = identifier
        self.threshold = threshold
        self.batch_size = batch_size
        self.queue: deque[tuple[_PreparedImage, Image.Image, int]] = deque()
        self.states: list[_PreparedImage] = []

    def add(
        self,
        state: _PreparedImage,
        crops: Sequence[tuple[Image.Image, int]],
    ) -> list[_PreparedImage]:
        """Queue one image and return images completed by full batches."""
        self.states.append(state)
        self.queue.extend((state, crop, index) for crop, index in crops)
        while len(self.queue) >= self.batch_size:
            self._flush(self.batch_size)
        return self._take_ready()

    def finish(self) -> list[_PreparedImage]:
        """Identify the final short batch and return all completed images."""
        if self.queue:
            self._flush(len(self.queue))
        return self._take_ready()

    def _flush(self, count: int) -> None:
        """Identify and close the next bounded crop batch."""
        items = [self.queue.popleft() for _index in range(count)]
        crops = [item[1] for item in items]
        try:
            predictions = self.identifier.predict(crops)
            if len(predictions) != len(items):
                raise ValueError(
                    "Identifier returned a different number of predictions than crops."
                )
            for prediction, (state, _crop, detection_index) in zip(
                predictions, items
            ):
                if prediction.score >= self.threshold:
                    state.identities.append(
                        {
                            "identity": prediction.identity,
                            "score": prediction.score,
                            "score_type": prediction.score_type,
                            "detection_index": detection_index,
                        }
                    )
                state.pending_crops -= 1
        finally:
            for crop in crops:
                crop.close()

    def _take_ready(self) -> list[_PreparedImage]:
        """Remove and return states with no outstanding predictions."""
        ready = [state for state in self.states if state.pending_crops == 0]
        if ready:
            ready_ids = {id(state) for state in ready}
            self.states = [state for state in self.states if id(state) not in ready_ids]
        return ready

    def close(self) -> None:
        """Close queued crops after cancellation or an exception."""
        while self.queue:
            _state, crop, _index = self.queue.popleft()
            crop.close()
        self.states.clear()


def _validate_output(output_root: Path) -> None:
    """Refuse non-empty output not carrying the app management marker."""
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Output path is not a folder: {output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not (output_root / MANAGEMENT_MARKER).is_file():
        raise ValueError("Existing output is not app-managed. Choose an empty folder or an output containing .finid-managed.")
    output_root.parent.mkdir(parents=True, exist_ok=True)


def _preflight_source_reports(store: ResultStore) -> None:
    """Raise before inference when an encounter report cannot be replaced."""
    unwritable: list[Path] = []
    for row in store.encounters():
        encounter = Path(str(row["path"]))
        report = encounter / report_filename(encounter)
        if not os.access(encounter, os.W_OK) or (report.exists() and not os.access(report, os.W_OK)):
            unwritable.append(encounter)
            if len(unwritable) == 10:
                break
    if unwritable:
        raise PermissionError(
            "Encounter reports cannot be written in:\n"
            + "\n".join(f"• {path}" for path in unwritable)
        )


def _safe_identity(value: str) -> str:
    """Return an identity safe for use in a copied filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "Unknown"


def _cluster_to_staging(
    config: PipelineConfig,
    store: ResultStore,
    stage: Path,
    progress: ProgressCallback | None = None,
) -> None:
    """Copy completed sources into staging with image-level progress.

    Parameters:
        config: Active pipeline configuration.
        store: Disk-backed run results.
        stage: Temporary output directory to populate.
        progress: Optional phase-local progress callback.

    Returns:
        None.
    """
    (stage / MANAGEMENT_MARKER).write_text("FinIdentification managed output\n", encoding="utf-8")
    copy_total = store.processed_images()
    copied = 0
    if progress is not None:
        progress(0, copy_total, "Creating staged encounter folders…")
    for encounter_row in store.encounters():
        encounter = Path(str(encounter_row["path"]))
        encounter_output = stage / Path(str(encounter_row["relative_path"]))
        for relative in ("LEFT/IDed", "LEFT/FinSaddle", "LEFT/Eyes", "RIGHT/IDed", "RIGHT/FinSaddle", "RIGHT/Eyes", "Rest"):
            (encounter_output / relative).mkdir(parents=True, exist_ok=True)
        for row in store.encounter_images(encounter):
            category = str(row["cluster_category"])
            side = row["cluster_side"]
            destination = encounter_output / (Path(str(side)) / category if side else Path(category))
            identities = list(store.identities(int(row["id"])))
            prefix = "_".join(_safe_identity(str(item["identity"])) for item in identities)
            original = str(row["filename"])
            candidate = f"{prefix}_{original}" if prefix else original
            if (destination / candidate).exists():
                source_suffix = hashlib.sha1(str(row["relative_path"]).encode("utf-8")).hexdigest()[:8]
                source_name = Path(candidate)
                candidate = f"{source_name.stem}_{source_suffix}{source_name.suffix}"
            shutil.copy2(str(row["path"]), destination / candidate)
            store.set_copied_filename(int(row["id"]), candidate)
            copied += 1
            if progress is not None and (copied == copy_total or copied % 25 == 0):
                progress(
                    copied,
                    copy_total,
                    f"Copying clustered images… {copied:,}/{copy_total:,}",
                )
        store.set_encounter_output(encounter, encounter_output)
    store.commit()
    if progress is not None:
        progress(copy_total, copy_total, "Cluster staging complete.")


def _commit_staging(stage: Path, output_root: Path) -> None:
    """Atomically replace an empty or managed output with completed staging."""
    had_output = output_root.exists()
    backup_container: Path | None = None
    backup: Path | None = None
    try:
        if had_output:
            backup_container = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_root.name}.finid-backup-",
                    dir=output_root.parent,
                )
            )
            backup = backup_container / "previous"
            os.replace(output_root, backup)
        os.replace(stage, output_root)
    except Exception:
        if backup is not None and backup.exists() and not output_root.exists():
            os.replace(backup, output_root)
        if backup_container is not None and backup_container.exists():
            shutil.rmtree(backup_container, ignore_errors=True)
        raise
    if backup_container is not None and backup_container.exists():
        shutil.rmtree(backup_container, ignore_errors=True)


def run_pipeline(
    config: PipelineConfig, *, log: LogCallback | None = None,
    progress: ProgressCallback | None = None, stop_event: threading.Event | None = None,
    runtime: Any | None = None, identifier_runtime: Any | None = None,
    probe: Callable[[], tuple[bool, str]] | None = None,
) -> PipelineSummary:
    """Run encounter-aware inference, optional clustering, and reporting."""
    emit_log = log or (lambda _message: None)
    emit_progress = progress or (lambda _current, _total, _label: None)
    stop_event = stop_event or threading.Event()
    started = time.monotonic()
    completed = stopped = False
    error: str | None = None
    report_count = total = processed = clustered = 0
    encounter_count = skipped_undated = 0
    processing_failures = 0
    detector_batch, identifier_batch = config.detector_batch_size, config.identifier_batch_size
    reports_root = config.output_root if config.clustering and config.output_root else config.input_dir
    root_report = reports_root
    detector_runtime, active_identifier = runtime, identifier_runtime
    stage: Path | None = None
    output_validated = False
    store = ResultStore()
    inference_start = 80
    inference_end = 780 if config.clustering else 840
    report_start = 900 if config.clustering else inference_end
    inference_progress = _phase_progress(
        emit_progress,
        inference_start,
        inference_end,
    )
    try:
        if not config.input_dir.is_dir():
            raise ValueError(f"Input folder does not exist: {config.input_dir}")
        if config.clustering and config.output_root is not None:
            if config.output_root == config.input_dir or config.output_root in config.input_dir.parents or config.input_dir in config.output_root.parents:
                raise ValueError("Input and clustering output folders must not overlap.")
            _validate_output(config.output_root)
            output_validated = True
        emit_log("Discovering dated encounters and JPEG filenames…")
        total = inventory_tree(config.input_dir, store, scan_progress=lambda files, dirs: emit_progress(files, 0, f"Scanning… {files:,} files in {dirs:,} folders"))
        encounter_count, skipped_undated = store.encounter_count(), store.skipped_undated_count()
        emit_progress(50, PROGRESS_SCALE, "Inventory complete; resolving run setup…")
        if config.clustering and not detector_has_finsaddle_classes(
            config.detector_classes
        ):
            emit_log(
                "Warning: clustering is enabled, but the detector model has no "
                "FinSaddle/saddle classes. IDed and FinSaddle will remain empty; "
                "plain fin detections will be classified as Rest."
            )
        if not config.clustering:
            _preflight_source_reports(store)
        emit_log(f"Found {encounter_count} encounters with {total} JPEG images.")
        if skipped_undated:
            examples = ", ".join(str(path) for path in store.skipped_undated_paths(5))
            emit_log(f"Skipped {skipped_undated} JPEGs without a dated encounter ancestor. Examples: {examples}")
        if total == 0:
            completed = True
        else:
            emit_progress(60, PROGRESS_SCALE, "Loading inference runtimes…")
            if probe is None:
                from findetection_core import probe_runtime
                probe = probe_runtime
            available, detail = probe()
            if not available:
                raise RuntimeError(detail)
            if detector_runtime is None:
                from findetection_core import MPSModelRuntime
                detector_runtime = MPSModelRuntime()
            from findetection_core import MPSDetectorConfig, MPSPredictionStopped
            detector_runtime.load(MPSDetectorConfig(
                model_path=config.detector.path, image_size=config.detector_image_size,
                batch_size=config.detector_batch_size,
                confidence=min(config.detector_confidence, config.eye_confidence),
                max_detections=config.max_detections, use_fp16=config.detector_fp16,
                source_window_size=config.detector_batch_size,
            ), log=emit_log)
            device_name = str(getattr(detector_runtime, "device", "mps"))
            class_names = _class_map(config)
            selected_classes = set(config.selected_class_ids or class_names)
            needs_identification = any(
                class_id in selected_classes
                and _kind_side(class_name)[0] == "finsaddle"
                for class_id, class_name in class_names.items()
            )
            if active_identifier is None and needs_identification:
                active_identifier = IdentifierRuntime(config.identifier, config.identifier_batch_size, log=emit_log, device_name=device_name, prefer_fp16=device_name == "mps")
            batcher = _IdentificationBatcher(
                active_identifier,
                config.threshold,
                config.identifier_batch_size,
            )
            inference_started = time.monotonic()
            inference_progress(0, total, "Starting detection and identification…")

            def record_ready(states: Sequence[_PreparedImage]) -> None:
                """Persist completed batched states and publish bounded progress."""
                nonlocal processed
                for state in states:
                    category, side = _classification(state, config)
                    store.record_result(
                        state.path,
                        state.detections,
                        state.identities,
                        category,
                        side,
                    )
                    processed += 1
                    elapsed = time.monotonic() - inference_started
                    rate = processed / elapsed if elapsed else 0.0
                    eta = (total - processed) / rate if rate else 0.0
                    inference_progress(
                        processed,
                        total,
                        f"Inference {processed}/{total} · {rate:.2f} images/s · "
                        f"this-step ETA about {eta / 60:.1f} min",
                    )

            try:
                try:
                    for result in detector_runtime.predict_paths(
                        store.pending_paths(), stop_event=stop_event
                    ):
                        if stop_event.is_set():
                            stopped = True
                            break
                        path = Path(result.path).resolve()
                        try:
                            state, crops = _prepare_result(
                                result,
                                config,
                                class_names,
                                selected_classes,
                            )
                        except Exception as exc:
                            store.record_failure(path, str(exc))
                            processing_failures += 1
                            if processing_failures <= 10:
                                emit_log(f"Could not process {path}: {exc}")
                            elif processing_failures == 11:
                                emit_log(
                                    "Additional image failures are being recorded in "
                                    "encounter reports without individual activity-log entries."
                                )
                            processed += 1
                            inference_progress(
                                processed,
                                total,
                                f"{processed}/{total} · processing failed image recorded",
                            )
                            continue
                        record_ready(batcher.add(state, crops))
                except MPSPredictionStopped:
                    stopped = True
                record_ready(batcher.finish())
            finally:
                batcher.close()
            stopped = stopped or stop_event.is_set()
            if not stopped:
                missing = store.mark_pending_failed("Image could not be decoded by the detector.")
                if missing:
                    emit_log(f"{missing} JPEG images could not be decoded.")
                    processed += missing
                completed = True
                inference_progress(
                    processed,
                    total,
                    f"Inference complete · {processed:,}/{total:,} images",
                )
            if processing_failures > 10:
                emit_log(
                    f"Recorded {processing_failures} per-image processing failures in SQLite."
                )
            detector_batch = int(getattr(detector_runtime, "effective_batch_size", detector_batch))
            identifier_batch = int(getattr(active_identifier, "effective_batch_size", identifier_batch))
    except Exception as exc:
        error = str(exc)
        emit_log(f"Pipeline stopped with an error: {exc}")
    finally:
        if detector_runtime is not None:
            try:
                detector_runtime.synchronize()
            except Exception:
                pass
            detector_runtime.close()
        if active_identifier is not None:
            active_identifier.close()
        elapsed = time.monotonic() - started
        processed = store.processed_images()
        message = error or ("The run was stopped. Results shown are for completed images only." if stopped else "")
        try:
            if config.clustering and config.output_root is not None and output_validated:
                stage = Path(tempfile.mkdtemp(prefix=f".{config.output_root.name}.finid-staging-", dir=config.output_root.parent))
                _cluster_to_staging(
                    config,
                    store,
                    stage,
                    _phase_progress(emit_progress, inference_end, 900),
                )
            else:
                emit_progress(
                    report_start,
                    PROGRESS_SCALE,
                    "Preparing source encounter report locations…",
                )
                for encounter_row in store.encounters():
                    store.set_encounter_output(Path(str(encounter_row["path"])), Path(str(encounter_row["path"])))
                store.commit()
            emit_log("Writing encounter reports and large-gallery thumbnails…")
            report_count = write_reports(
                store,
                ReportMetadata(
                    generated_at=datetime.now().astimezone(), completed=completed,
                    detector_name=config.detector.name, identifier_name=config.identifier.name,
                    threshold=config.threshold, score_label=config.identifier.score_label,
                    elapsed_seconds=elapsed, throughput=processed / elapsed if elapsed else 0.0,
                    fin_confidence=config.detector_confidence, eye_confidence=config.eye_confidence,
                    message=message,
                ),
                progress=_phase_progress(emit_progress, report_start, 990),
            )
            if stage is not None and config.output_root is not None:
                emit_progress(995, PROGRESS_SCALE, "Publishing completed output atomically…")
                _commit_staging(stage, config.output_root)
                stage = None
                clustered = store.clustered_images()
            first = next(store.encounters(), None)
            if first is not None:
                final_encounter_root = reports_root / Path(str(first["relative_path"])) if config.clustering else Path(str(first["path"]))
                root_report = final_encounter_root / report_filename(Path(str(first["path"])))
            emit_log(f"Wrote {report_count} encounter reports.")
            if error is None:
                emit_progress(
                    PROGRESS_SCALE,
                    PROGRESS_SCALE,
                    "Partial run finalized." if stopped else "Run complete.",
                )
        except Exception as report_exc:
            if stage is not None and stage.exists():
                shutil.rmtree(stage)
            error = f"{error}; report/output error: {report_exc}" if error else f"Could not write reports/output: {report_exc}"
            completed = False
            emit_log(error)
        store.close()
    return PipelineSummary(completed, stopped, processed, total, encounter_count,
        skipped_undated, clustered, report_count, reports_root, root_report,
        time.monotonic() - started, detector_batch, identifier_batch, error)
