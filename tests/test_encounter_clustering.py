from __future__ import annotations

import re
import tempfile
import threading
import tracemalloc
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from findetection_core import DetectionClass

import finid.pipeline as pipeline_module
from finid.models import (
    DetectionModel,
    IdentificationModel,
    IdentityCandidate,
    IdentityPrediction,
)
from finid.pipeline import (
    MANAGEMENT_MARKER,
    PipelineConfig,
    discover_encounters,
    inventory_tree,
    run_pipeline,
)
from finid.reporting import report_filename
from finid.storage import ResultStore


def detection(
    class_id: int,
    confidence: float = 0.9,
    coordinates: tuple[int, int, int, int] = (2, 2, 30, 30),
) -> object:
    """Build a normalized fake detector box.

    Parameters:
        class_id: Detector class ID.
        confidence: Detector score.
        coordinates: Pixel bounding box.

    Returns:
        A box-shaped test double.
    """
    return SimpleNamespace(class_id=class_id, confidence=confidence, xyxy=coordinates)


class Detector:
    """Small detector runtime that yields configured boxes by filename."""

    def __init__(self, boxes: dict[str, list[object]]) -> None:
        self.boxes = boxes
        self.effective_batch_size = 2

    def load(self, _config: object, log: object = None) -> None:
        """Accept a detector configuration."""

    def predict_paths(self, paths: object, stop_event: threading.Event | None = None):
        """Yield one result per source path."""
        for path in paths:
            if stop_event is not None and stop_event.is_set():
                from findetection_core import MPSPredictionStopped

                raise MPSPredictionStopped()
            yield SimpleNamespace(path=path, boxes=self.boxes.get(Path(path).name, ()), image=None)

    def synchronize(self) -> None:
        """Synchronize the fake runtime."""

    def close(self) -> None:
        """Close the fake runtime."""


class Identifier:
    """Return deterministic scores and distinct identities across crop calls."""

    def __init__(
        self,
        scores: list[float],
        identities: list[str] | None = None,
    ) -> None:
        """Initialize deterministic identifier output.

        Parameters:
            scores: Prediction scores returned in crop order.
            identities: Optional identity names returned in crop order.

        Returns:
            None.
        """
        self.scores = scores
        self.identity_names = identities
        self.index = 0
        self.effective_batch_size = 8
        self.batch_sizes: list[int] = []

    def predict(self, crops: object) -> list[IdentityPrediction]:
        """Return one prediction for each crop."""
        self.batch_sizes.append(len(crops))
        predictions: list[IdentityPrediction] = []
        for _crop in crops:
            score = self.scores[self.index]
            identity = (
                self.identity_names[self.index]
                if self.identity_names is not None
                else f"NKW-{self.index + 1:03d}"
            )
            self.index += 1
            predictions.append(IdentityPrediction(identity, score, "probability"))
        return predictions

    def close(self) -> None:
        """Close the fake identifier."""


class RankedIdentifier:
    """Return prebuilt predictions with ranked identity candidates."""

    def __init__(self, predictions: list[IdentityPrediction]) -> None:
        """Initialize ordered fake predictions.

        Parameters:
            predictions: Predictions returned in crop order.

        Returns:
            None.
        """
        self.predictions = predictions
        self.offset = 0
        self.effective_batch_size = 8

    def predict(self, crops: object) -> list[IdentityPrediction]:
        """Return one configured prediction for every supplied crop.

        Parameters:
            crops: Crop batch whose length determines the returned slice.

        Returns:
            Configured predictions for the current batch.
        """
        count = len(crops)
        output = self.predictions[self.offset : self.offset + count]
        self.offset += count
        return output

    def close(self) -> None:
        """Close the fake identifier."""


def ranked_prediction(
    values: tuple[tuple[str, float], ...],
) -> IdentityPrediction:
    """Build a prediction containing ordered candidates.

    Parameters:
        values: Identity and score pairs in descending rank order.

    Returns:
        Best prediction carrying all supplied candidates.
    """
    candidates = tuple(
        IdentityCandidate(identity, score, "probability")
        for identity, score in values
    )
    best = candidates[0]
    return IdentityPrediction(
        best.identity,
        best.score,
        best.score_type,
        candidates,
    )


CLASSES = (
    DetectionClass(0, "fin_left", "Fin Left"),
    DetectionClass(1, "fin_right", "Fin Right"),
    DetectionClass(2, "finSaddle_left", "FinSaddle Left"),
    DetectionClass(3, "finSaddle_right", "FinSaddle Right"),
    DetectionClass(4, "eye_left", "Eye Left"),
    DetectionClass(5, "eye_right", "Eye Right"),
)


def models() -> tuple[DetectionModel, IdentificationModel]:
    """Return lightweight model descriptors for pipeline tests."""
    return (
        DetectionModel("Detector", Path("detector.pt")),
        IdentificationModel("Identifier", Path("identifier.pt"), "resnet", "probability"),
    )


def save_jpeg(path: Path) -> None:
    """Create a small valid JPEG and all parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 35), "white").save(path)


def save_jpeg_at(path: Path, captured: datetime) -> None:
    """Create a JPEG with an EXIF original capture timestamp.

    Parameters:
        path: Destination JPEG path.
        captured: Naive capture time, including optional microseconds.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[36867] = captured.strftime("%Y:%m:%d %H:%M:%S")
    if captured.microsecond:
        exif[37521] = f"{captured.microsecond:06d}".rstrip("0")
    Image.new("RGB", (40, 35), "white").save(path, exif=exif)


class EncounterDiscoveryTests(unittest.TestCase):
    def test_nested_dates_explicit_splits_and_undated_omission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2_All"
            save_jpeg(root / "2026-02-01 Trip" / "GROUP1" / "Trip2" / "RIGHT" / "a.jpg")
            save_jpeg(root / "2026-02-02 Day" / "ENCOUNTER1" / "Camera A" / "b.jpg")
            save_jpeg(root / "2026-02-02 Day" / "ENCOUNTER2" / "Camera B" / "c.jpg")
            save_jpeg(root / "year" / "undated.jpg")

            encounters = discover_encounters(root)

            self.assertEqual(
                [item.root.name for item in encounters],
                ["2026-02-01 Trip", "ENCOUNTER1", "ENCOUNTER2"],
            )
            self.assertEqual([len(item.images) for item in encounters], [1, 1, 1])
            self.assertEqual(
                encounters[1].relative_path,
                Path("2026-02-02 Day/ENCOUNTER1"),
            )

    def test_selected_dated_root_is_one_encounter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-03-04 Encounter"
            save_jpeg(root / "Camera" / "one.jpg")
            encounters = discover_encounters(root)
            self.assertEqual(len(encounters), 1)
            self.assertEqual(encounters[0].relative_path, Path(root.name))

    def test_sibling_groups_are_independent_encounters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            dated = root / "2026-03-05 Grouped"
            save_jpeg(dated / "Trip" / "GROUP1" / "Camera" / "one.jpg")
            save_jpeg(dated / "Trip" / "GROUP2" / "Camera" / "two.jpg")

            encounters = discover_encounters(root)

            self.assertEqual([item.root.name for item in encounters], ["GROUP1", "GROUP2"])
            self.assertEqual([len(item.images) for item in encounters], [1, 1])
            self.assertEqual(
                [item.relative_path for item in encounters],
                [
                    Path("2026-03-05 Grouped/Trip/GROUP1"),
                    Path("2026-03-05 Grouped/Trip/GROUP2"),
                ],
            )
            with ResultStore() as store:
                self.assertEqual(inventory_tree(root, store), 2)
                self.assertEqual(store.encounter_count(), 2)
                self.assertEqual(
                    [int(row["image_count"]) for row in store.encounters()],
                    [1, 1],
                )

    def test_inventory_memory_does_not_scale_with_image_path_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-03-06 Large"
            root.mkdir()
            for index in range(12_000):
                (root / f"{index:05d}.jpg").touch()
            with ResultStore() as store:
                tracemalloc.start()
                total = inventory_tree(root, store, yield_every=0)
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

            self.assertEqual(total, 12_000)
            self.assertLess(peak, 12 * 1024 * 1024)

    def test_inventory_traverses_the_filesystem_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-03-07 Single Pass"
            save_jpeg(root / "Camera A" / "one.jpg")
            save_jpeg(root / "Camera B" / "two.jpg")
            original = pipeline_module._iter_tree
            with (
                ResultStore() as store,
                patch("finid.pipeline._iter_tree", wraps=original) as traversal,
            ):
                self.assertEqual(inventory_tree(root, store), 2)
            self.assertEqual(traversal.call_count, 1)

    def test_inventory_skips_managed_report_thumbnails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-03-08 Report Assets"
            save_jpeg(root / "source.jpg")
            assets = root / f"FinID_{root.name}_assets"
            save_jpeg(assets / "thumbs" / "thumbnail.jpg")
            (assets / ".finid-report-assets").write_text(
                "managed",
                encoding="utf-8",
            )
            with ResultStore() as store:
                self.assertEqual(inventory_tree(root, store), 1)


class ClusteringTests(unittest.TestCase):
    def test_inplace_clusters_are_managed_and_excluded_from_reruns(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            encounter = Path(temporary) / "2026-03-28 In Place"
            source_image = encounter / "one.jpg"
            save_jpeg(source_image)
            config = PipelineConfig(
                encounter,
                detector_model,
                identifier_model,
                clustering=True,
                inplace_clustering=True,
                detector_classes=CLASSES,
                selected_class_ids=(0,),
            )

            first = run_pipeline(
                config,
                runtime=Detector({}),
                identifier_runtime=Identifier([]),
                probe=lambda: (True, "ready"),
            )
            second = run_pipeline(
                config,
                runtime=Detector({}),
                identifier_runtime=Identifier([]),
                probe=lambda: (True, "ready"),
            )

            clusters = encounter / f"FinID_{encounter.name}_clusters"
            self.assertTrue(first.completed)
            self.assertTrue(second.completed)
            self.assertEqual(first.total, 1)
            self.assertEqual(second.total, 1)
            self.assertEqual(second.clustered_image_count, 1)
            self.assertTrue(source_image.is_file())
            self.assertTrue((clusters / MANAGEMENT_MARKER).is_file())
            self.assertTrue((clusters / "Rest/one.jpg").is_file())
            self.assertTrue((clusters / report_filename(encounter)).is_file())
            self.assertEqual(
                second.root_report,
                (clusters / report_filename(encounter)).resolve(),
            )

    def test_right_identification_toggle_preserves_right_classification(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-03-29 Left Only"
            for name in ("left.jpg", "right-eye.jpg", "right-fin.jpg", "right-saddle.jpg"):
                save_jpeg(encounter / name)
            output = Path(temporary) / "output"
            boxes = {
                "left.jpg": [detection(2)],
                "right-eye.jpg": [detection(5)],
                "right-fin.jpg": [detection(1)],
                "right-saddle.jpg": [detection(3)],
            }
            identifier = Identifier([0.9])

            summary = run_pipeline(
                PipelineConfig(
                    source,
                    detector_model,
                    identifier_model,
                    exclude_right_identification=True,
                    clustering=True,
                    output_root=output,
                    detector_classes=CLASSES,
                    selected_class_ids=(0, 1, 2, 3),
                ),
                runtime=Detector(boxes),
                identifier_runtime=identifier,
                probe=lambda: (True, "ready"),
            )

            target = output / encounter.name
            self.assertTrue(summary.completed)
            self.assertEqual(identifier.batch_sizes, [1])
            self.assertTrue((target / "LEFT/IDed/NKW-001_left.jpg").is_file())
            self.assertFalse((target / "RIGHT/IDed").exists())
            self.assertTrue((target / "RIGHT/Eyes/right-eye.jpg").is_file())
            self.assertTrue(
                (target / "RIGHT/FinSaddle/right-saddle.jpg").is_file()
            )
            self.assertTrue((target / "Rest/right-fin.jpg").is_file())
            report = target / report_filename(encounter)
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("RIGHT-side identification disabled", report_text)
            self.assertIn("finSaddle_right", report_text)
            self.assertIn("eye_right", report_text)

    def test_runtime_cleanup_failure_does_not_discard_completed_results(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "2026-03-30 Cleanup"
            save_jpeg(source / "one.jpg")
            detector = Detector({})
            detector.close = lambda: (_ for _ in ()).throw(
                RuntimeError("close failed")
            )
            messages: list[str] = []

            summary = run_pipeline(
                PipelineConfig(
                    source,
                    detector_model,
                    identifier_model,
                    detector_classes=CLASSES,
                    selected_class_ids=(0,),
                ),
                runtime=detector,
                identifier_runtime=Identifier([]),
                probe=lambda: (True, "ready"),
                log=messages.append,
            )

            self.assertTrue(summary.completed)
            self.assertIsNone(summary.error)
            self.assertTrue(
                any("could not fully close detector runtime" in item for item in messages)
            )

    def test_progress_is_monotonic_across_every_pipeline_phase(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-03-31 Progress"
            for index in range(3):
                save_jpeg(encounter / f"{index}.jpg")
            output = Path(temporary) / "output"
            updates: list[tuple[int, int, str]] = []

            summary = run_pipeline(
                PipelineConfig(
                    source,
                    detector_model,
                    identifier_model,
                    clustering=True,
                    output_root=output,
                    detector_classes=CLASSES,
                    selected_class_ids=(2,),
                ),
                runtime=Detector({}),
                identifier_runtime=Identifier([]),
                probe=lambda: (True, "ready"),
                progress=lambda current, total, label: updates.append(
                    (current, total, label)
                ),
            )

            determinate = [
                update
                for update in updates
                if update[1] == pipeline_module.PROGRESS_SCALE
            ]
            values = [update[0] for update in determinate]
            labels = [update[2] for update in determinate]
            self.assertTrue(summary.completed)
            self.assertEqual(values, sorted(values))
            self.assertEqual(determinate[-1], (1000, 1000, "Run complete."))
            self.assertTrue(
                any("Copying clustered images" in label for label in labels)
            )
            self.assertTrue(
                any("Writing reports" in label for label in labels)
            )
            self.assertTrue(
                any("Publishing completed output" in label for label in labels)
            )
            self.assertTrue(
                any("this-step ETA" in label for label in labels)
            )

    def test_identifier_batches_finsaddle_crops_across_images(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "2026-04-01 Batched"
            boxes: dict[str, list[object]] = {}
            for index in range(10):
                name = f"{index:02d}.jpg"
                save_jpeg(source / name)
                boxes[name] = [detection(2)]
            identifier = Identifier([0.9] * 10)

            summary = run_pipeline(
                PipelineConfig(
                    source,
                    detector_model,
                    identifier_model,
                    detector_classes=CLASSES,
                    selected_class_ids=(2,),
                    identifier_batch_size=4,
                ),
                runtime=Detector(boxes),
                identifier_runtime=identifier,
                probe=lambda: (True, "ready"),
            )

            self.assertTrue(summary.completed)
            self.assertEqual(identifier.batch_sizes, [4, 4, 2])

    def test_priority_sides_multi_id_and_report_locations(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-04-05 Survey"
            for name in ("both.jpg", "eye.jpg", "id.jpg", "multi.jpg", "none.jpg", "plain.jpg", "saddle.jpg"):
                save_jpeg(encounter / "GROUP1" / name)
            (encounter / "GROUP1" / "broken.jpg").write_bytes(b"not a JPEG")
            output = Path(temporary) / "clusters"
            boxes = {
                "both.jpg": [detection(2, 0.25), detection(3, 0.25)],
                "broken.jpg": [detection(2)],
                "eye.jpg": [detection(5, 0.25)],
                "id.jpg": [detection(3), detection(5)],
                "multi.jpg": [detection(2), detection(3)],
                "plain.jpg": [detection(0)],
                "saddle.jpg": [detection(3, 0.25)],
            }
            identifier = Identifier([0.1, 0.1, 0.5, 0.95, 0.9, 0.2])
            summary = run_pipeline(
                PipelineConfig(
                    source, detector_model, identifier_model,
                    clustering=True, output_root=output,
                    detector_classes=CLASSES, selected_class_ids=(0, 1, 2, 3),
                ),
                runtime=Detector(boxes),
                identifier_runtime=identifier,
                probe=lambda: (True, "test device ready"),
            )

            target = output / encounter.name
            self.assertTrue(summary.completed)
            self.assertEqual(summary.encounter_count, 1)
            self.assertEqual(summary.clustered_image_count, 8)
            self.assertEqual(summary.report_count, 1)
            self.assertEqual(identifier.batch_sizes, [6])
            self.assertTrue((target / "LEFT/FinSaddle/both.jpg").is_file())
            self.assertTrue((target / "RIGHT/Eyes/eye.jpg").is_file())
            self.assertTrue((target / "RIGHT/IDed/NKW-003_EYE_id.jpg").is_file())
            self.assertTrue((target / "LEFT/IDed/NKW-004_NKW-005_multi.jpg").is_file())
            self.assertTrue((target / "Rest/plain.jpg").is_file())
            self.assertTrue((target / "Rest/broken.jpg").is_file())
            self.assertTrue((target / "RIGHT/FinSaddle/saddle.jpg").is_file())
            reports = list(output.rglob("FinID_*.html"))
            self.assertEqual(reports, [target / report_filename(encounter)])
            text = reports[0].read_text(encoding="utf-8")
            self.assertEqual(text.count("NKW-004_NKW-005_multi.jpg</strong>"), 1)
            self.assertIn("Original:", text)
            self.assertIn("Detections:", text)
            self.assertIn(
                '<span style="color:#0369a1;font-weight:700">0.250</span>',
                text,
            )
            self.assertIn(
                '<span style="color:#b45309;font-weight:700">0.250</span>',
                text,
            )
            self.assertIn("Problem:", text)

    def test_rejected_finsaddles_report_three_candidates_per_box(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            encounter = Path(temporary) / "2026-04-05 Rejected Candidates"
            save_jpeg(encounter / "saddles.jpg")
            identifier = RankedIdentifier(
                [
                    ranked_prediction(
                        (("NKW-101", 0.49), ("NKW-102", 0.41), ("NKW-103", 0.33))
                    ),
                    ranked_prediction(
                        (("NKW-201", 0.48), ("NKW-202", 0.40), ("NKW-203", 0.32))
                    ),
                ]
            )

            summary = run_pipeline(
                PipelineConfig(
                    encounter,
                    detector_model,
                    identifier_model,
                    threshold=0.5,
                    detector_classes=CLASSES,
                    selected_class_ids=(2, 3),
                ),
                runtime=Detector(
                    {
                        "saddles.jpg": [
                            detection(2, 0.9),
                            detection(3, 0.8),
                        ]
                    }
                ),
                identifier_runtime=identifier,
                probe=lambda: (True, "ready"),
            )

            self.assertTrue(summary.completed)
            report_text = (encounter / report_filename(encounter)).read_text(
                encoding="utf-8"
            )
            self.assertIn("Destination: FinSaddle", report_text)
            self.assertIn("<strong>Candidates:</strong>", report_text)
            for identity in ("NKW-101", "NKW-102", "NKW-103"):
                self.assertIn(
                    f'<span style="color:#0369a1;font-weight:700">{identity}</span>',
                    report_text,
                )
            for identity in ("NKW-201", "NKW-202", "NKW-203"):
                self.assertIn(
                    f'<span style="color:#b45309;font-weight:700">{identity}</span>',
                    report_text,
                )
            self.assertNotIn("NKW-204", report_text)

    def test_burst_eye_inherits_all_identities_and_left_wins(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-04-06 Burst IDs"
            camera = encounter / "Camera A"
            captured = datetime(2026, 4, 6, 12, 0, 0)
            names = (
                "a-left-saddle.jpg",
                "b-right-saddle.jpg",
                "c-left-saddle.jpg",
                "d-right-eye.jpg",
            )
            for index, name in enumerate(names):
                save_jpeg_at(camera / name, captured + timedelta(milliseconds=400 * index))
            output = Path(temporary) / "output"
            boxes = {
                "a-left-saddle.jpg": [detection(2)],
                "b-right-saddle.jpg": [detection(3)],
                "c-left-saddle.jpg": [detection(2)],
                "d-right-eye.jpg": [detection(5)],
            }
            identifier = Identifier(
                [0.7, 0.8, 0.95],
                ["NKW-001", "NKW-002", "NKW-001"],
            )

            summary = run_pipeline(
                PipelineConfig(
                    source,
                    detector_model,
                    identifier_model,
                    clustering=True,
                    output_root=output,
                    detector_classes=CLASSES,
                    selected_class_ids=(2, 3),
                ),
                runtime=Detector(boxes),
                identifier_runtime=identifier,
                probe=lambda: (True, "ready"),
            )

            target = output / encounter.name
            linked = target / "LEFT/IDed/NKW-001_NKW-002_EYE_d-right-eye.jpg"
            self.assertTrue(summary.completed)
            self.assertTrue(linked.is_file())
            self.assertFalse((target / "RIGHT/Eyes/d-right-eye.jpg").exists())
            report_text = (target / report_filename(encounter)).read_text(
                encoding="utf-8"
            )
            self.assertIn("NKW-001 0.950", report_text)
            self.assertIn("NKW-002 0.800", report_text)
            self.assertIn("2 orcas total · 4 images", report_text)

    def test_burst_boundaries_use_chained_exif_times_and_source_folder(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-04-07 Burst Boundaries"
            camera_a = encounter / "Camera A"
            camera_b = encounter / "Camera B"
            captured = datetime(2026, 4, 7, 12, 0, 0)
            timed = {
                "a-saddle.jpg": captured,
                "b-exact.jpg": captured + timedelta(seconds=2),
                "c-chained.jpg": captured + timedelta(seconds=4),
                "d-outside.jpg": captured + timedelta(seconds=6, milliseconds=1),
            }
            for name, capture_time in timed.items():
                save_jpeg_at(camera_a / name, capture_time)
            save_jpeg(camera_a / "e-missing.jpg")
            save_jpeg_at(camera_b / "f-other-camera.jpg", captured + timedelta(seconds=1))
            output = Path(temporary) / "output"
            boxes = {
                "a-saddle.jpg": [detection(2)],
                "b-exact.jpg": [detection(4)],
                "c-chained.jpg": [detection(4)],
                "d-outside.jpg": [detection(4)],
                "e-missing.jpg": [detection(4)],
                "f-other-camera.jpg": [detection(4)],
            }

            summary = run_pipeline(
                PipelineConfig(
                    source,
                    detector_model,
                    identifier_model,
                    clustering=True,
                    output_root=output,
                    detector_classes=CLASSES,
                    selected_class_ids=(2,),
                ),
                runtime=Detector(boxes),
                identifier_runtime=Identifier([0.9]),
                probe=lambda: (True, "ready"),
            )

            target = output / encounter.name
            self.assertTrue(summary.completed)
            self.assertTrue((target / "LEFT/IDed/NKW-001_EYE_b-exact.jpg").is_file())
            self.assertTrue((target / "LEFT/IDed/NKW-001_EYE_c-chained.jpg").is_file())
            self.assertTrue((target / "LEFT/Eyes/d-outside.jpg").is_file())
            self.assertTrue((target / "LEFT/Eyes/e-missing.jpg").is_file())
            self.assertTrue((target / "LEFT/Eyes/f-other-camera.jpg").is_file())

    def test_identified_saddle_wins_and_supplies_right_destination(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-04-07 Right Burst"
            camera = encounter / "Camera A"
            captured = datetime(2026, 4, 7, 13, 0, 0)
            for index, name in enumerate(
                (
                    "a-left-unidentified.jpg",
                    "b-right-identified.jpg",
                    "c-left-eye.jpg",
                    "d-low-eye.jpg",
                )
            ):
                save_jpeg_at(camera / name, captured + timedelta(milliseconds=index * 400))
            output = Path(temporary) / "output"
            boxes = {
                "a-left-unidentified.jpg": [detection(2)],
                "b-right-identified.jpg": [detection(3)],
                "c-left-eye.jpg": [detection(4)],
                "d-low-eye.jpg": [detection(4, 0.2)],
            }

            summary = run_pipeline(
                PipelineConfig(
                    source,
                    detector_model,
                    identifier_model,
                    threshold=0.5,
                    eye_confidence=0.5,
                    clustering=True,
                    output_root=output,
                    detector_classes=CLASSES,
                    selected_class_ids=(2, 3),
                ),
                runtime=Detector(boxes),
                identifier_runtime=Identifier([0.1, 0.9]),
                probe=lambda: (True, "ready"),
            )

            target = output / encounter.name
            self.assertTrue(summary.completed)
            self.assertTrue((target / "RIGHT/IDed/NKW-002_EYE_c-left-eye.jpg").is_file())
            self.assertTrue((target / "Rest/d-low-eye.jpg").is_file())

    def test_disabled_right_policy_keeps_right_eyes_but_links_left_eyes(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-04-08 Right Policy"
            captured = datetime(2026, 4, 8, 12, 0, 0)
            camera_a = encounter / "Camera A"
            camera_b = encounter / "Camera B"
            for index, name in enumerate(
                ("a-right-saddle.jpg", "b-left-eye.jpg", "c-right-eye.jpg")
            ):
                save_jpeg_at(camera_a / name, captured + timedelta(milliseconds=index * 500))
            for index, name in enumerate(
                ("d-left-saddle.jpg", "e-left-eye.jpg", "f-right-eye.jpg")
            ):
                save_jpeg_at(camera_b / name, captured + timedelta(milliseconds=index * 500))
            output = Path(temporary) / "output"
            boxes = {
                "a-right-saddle.jpg": [detection(3)],
                "b-left-eye.jpg": [detection(4)],
                "c-right-eye.jpg": [detection(5)],
                "d-left-saddle.jpg": [detection(2)],
                "e-left-eye.jpg": [detection(4)],
                "f-right-eye.jpg": [detection(5)],
            }

            summary = run_pipeline(
                PipelineConfig(
                    source,
                    detector_model,
                    identifier_model,
                    exclude_right_identification=True,
                    clustering=True,
                    output_root=output,
                    detector_classes=CLASSES,
                    selected_class_ids=(2, 3),
                ),
                runtime=Detector(boxes),
                identifier_runtime=Identifier([0.9]),
                probe=lambda: (True, "ready"),
            )

            target = output / encounter.name
            self.assertTrue(summary.completed)
            self.assertFalse((target / "RIGHT/IDed").exists())
            self.assertTrue((target / "RIGHT/FinSaddle/b-left-eye.jpg").is_file())
            self.assertTrue((target / "RIGHT/Eyes/c-right-eye.jpg").is_file())
            self.assertTrue((target / "LEFT/IDed/NKW-001_EYE_e-left-eye.jpg").is_file())
            self.assertTrue((target / "RIGHT/Eyes/f-right-eye.jpg").is_file())

    def test_disabled_only_right_candidates_do_not_load_identifier(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            encounter = Path(temporary) / "2026-04-09 Right Only"
            save_jpeg_at(
                encounter / "right-saddle.jpg",
                datetime(2026, 4, 9, 12, 0, 0),
            )
            config = PipelineConfig(
                encounter,
                detector_model,
                identifier_model,
                exclude_right_identification=True,
                detector_classes=CLASSES,
                selected_class_ids=(3,),
            )

            with patch(
                "finid.pipeline.IdentifierRuntime",
                side_effect=AssertionError("identifier should not load"),
            ):
                summary = run_pipeline(
                    config,
                    runtime=Detector({"right-saddle.jpg": [detection(3)]}),
                    probe=lambda: (True, "ready"),
                )

            self.assertTrue(summary.completed)
            self.assertIsNone(summary.error)

    def test_undated_skip_duplicate_names_and_managed_rerun(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-05-06 Duplicate"
            save_jpeg(encounter / "Camera A" / "same.jpg")
            save_jpeg(encounter / "Camera B" / "same.jpg")
            save_jpeg(source / "loose.jpg")
            output = Path(temporary) / "output"
            config = PipelineConfig(
                source, detector_model, identifier_model, clustering=True,
                output_root=output, detector_classes=CLASSES, selected_class_ids=(0,),
            )

            first = run_pipeline(config, runtime=Detector({}), identifier_runtime=Identifier([]), probe=lambda: (True, "ready"))
            first_names = sorted(path.name for path in (output / encounter.name / "Rest").glob("*.jpg"))
            second = run_pipeline(config, runtime=Detector({}), identifier_runtime=Identifier([]), probe=lambda: (True, "ready"))
            second_names = sorted(path.name for path in (output / encounter.name / "Rest").glob("*.jpg"))

            self.assertEqual(first.skipped_undated_count, 1)
            self.assertEqual(first_names, second_names)
            self.assertEqual(len(first_names), 2)
            self.assertTrue(any(re.fullmatch(r"same_[0-9a-f]{8}\.jpg", name) for name in first_names))
            self.assertTrue((output / MANAGEMENT_MARKER).is_file())
            self.assertTrue(second.completed)

    def test_unmanaged_output_refusal_and_staging_rollback(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-06-07 Rollback"
            save_jpeg(encounter / "one.jpg")
            output = Path(temporary) / "output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("original", encoding="utf-8")
            config = PipelineConfig(source, detector_model, identifier_model, clustering=True, output_root=output, detector_classes=CLASSES, selected_class_ids=(0,))

            refused = run_pipeline(config, runtime=Detector({}), identifier_runtime=Identifier([]), probe=lambda: (True, "ready"))
            self.assertIn("not app-managed", refused.error or "")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")

            (output / MANAGEMENT_MARKER).write_text("managed", encoding="utf-8")
            with patch("finid.pipeline.shutil.copy2", side_effect=OSError("copy failed")):
                failed = run_pipeline(config, runtime=Detector({}), identifier_runtime=Identifier([]), probe=lambda: (True, "ready"))
            self.assertIn("copy failed", failed.error or "")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")
            self.assertFalse(any(output.parent.glob(f".{output.name}.finid-staging-*")))

    def test_partial_stop_commits_a_partial_encounter_report(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-07-08 Partial"
            save_jpeg(encounter / "one.jpg")
            output = Path(temporary) / "output"
            stop = threading.Event()
            stop.set()

            summary = run_pipeline(
                PipelineConfig(
                    source, detector_model, identifier_model, clustering=True,
                    output_root=output, detector_classes=CLASSES,
                    selected_class_ids=(0,),
                ),
                runtime=Detector({}),
                identifier_runtime=Identifier([]),
                probe=lambda: (True, "ready"),
                stop_event=stop,
            )

            report = output / encounter.name / report_filename(encounter)
            self.assertTrue(summary.stopped)
            self.assertEqual(summary.clustered_image_count, 0)
            self.assertTrue(report.is_file())
            self.assertIn("Partial", report.read_text(encoding="utf-8"))

    def test_large_report_embeds_paged_data_without_asset_directory(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            encounter = source / "2026-07-09 Virtual"
            for index in range(5):
                save_jpeg(encounter / f"{index}.jpg")
            output = Path(temporary) / "output"

            with (
                patch("finid.reporting.VIRTUALIZE_THRESHOLD", 2),
                patch("finid.reporting.REPORT_PAGE_SIZE", 2),
            ):
                summary = run_pipeline(
                    PipelineConfig(
                        source,
                        detector_model,
                        identifier_model,
                        clustering=True,
                        output_root=output,
                        detector_classes=CLASSES,
                        selected_class_ids=(2,),
                    ),
                    runtime=Detector(
                        {
                            "0.jpg": [detection(2)],
                            "1.jpg": [detection(2)],
                        }
                    ),
                    identifier_runtime=RankedIdentifier(
                        [
                            ranked_prediction(
                                (("NKW-001", 0.9), ("NKW-002", 0.8), ("NKW-003", 0.7))
                            ),
                            ranked_prediction(
                                (("NKW-101", 0.4), ("NKW-102", 0.3), ("NKW-103", 0.2))
                            ),
                        ]
                    ),
                    probe=lambda: (True, "ready"),
                )

            report = output / encounter.name / report_filename(encounter)
            assets = output / encounter.name / f"FinID_{encounter.name}_assets"
            self.assertTrue(summary.completed)
            text = report.read_text(encoding="utf-8")
            self.assertIn("FINID_SECTIONS", text)
            self.assertIn('type="application/json" id="finid-page-', text)
            self.assertIn('"pages":["finid-page-', text)
            self.assertIn(
                '"identities":[["NKW-001",0.9,"probability","#0369a1"]]',
                text,
            )
            self.assertIn(
                '"candidates":[["NKW-101",0.4,"probability","#0369a1"],'
                '["NKW-102",0.3,"probability","#0369a1"],'
                '["NKW-103",0.2,"probability","#0369a1"]]',
                text,
            )
            self.assertIn("if(value[3])identity.style.cssText", text)
            self.assertFalse(assets.exists())

    def test_report_removes_legacy_managed_asset_directory(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            encounter = Path(temporary) / "2026-07-10 Legacy Assets"
            save_jpeg(encounter / "one.jpg")
            assets = encounter / f"FinID_{encounter.name}_assets"
            save_jpeg(assets / "thumbs" / "thumbnail.jpg")
            (assets / ".finid-report-assets").write_text(
                "managed",
                encoding="utf-8",
            )

            summary = run_pipeline(
                PipelineConfig(
                    encounter,
                    detector_model,
                    identifier_model,
                    detector_classes=CLASSES,
                    selected_class_ids=(0,),
                ),
                runtime=Detector({}),
                identifier_runtime=Identifier([]),
                probe=lambda: (True, "ready"),
            )

            self.assertTrue(summary.completed)
            self.assertFalse(assets.exists())
            self.assertTrue((encounter / report_filename(encounter)).is_file())

    def test_report_setup_failure_removes_temporary_report(self) -> None:
        detector_model, identifier_model = models()
        with tempfile.TemporaryDirectory() as temporary:
            encounter = Path(temporary) / "2026-07-10 Asset Cleanup"
            save_jpeg(encounter / "one.jpg")
            original_mkstemp = tempfile.mkstemp

            def fail_report_file(*args: object, **kwargs: object) -> tuple[int, str]:
                """Fail HTML staging while allowing SQLite temporary files.

                Parameters:
                    args: Positional ``mkstemp`` arguments.
                    kwargs: Keyword ``mkstemp`` arguments.

                Returns:
                    A descriptor and path for non-report temporary files.
                """
                if kwargs.get("suffix") == ".tmp":
                    raise OSError("temporary report failed")
                return original_mkstemp(*args, **kwargs)

            with (
                patch("finid.reporting.VIRTUALIZE_THRESHOLD", 0),
                patch(
                    "finid.reporting.tempfile.mkstemp",
                    side_effect=fail_report_file,
                ),
            ):
                summary = run_pipeline(
                    PipelineConfig(
                        encounter,
                        detector_model,
                        identifier_model,
                        detector_classes=CLASSES,
                        selected_class_ids=(0,),
                    ),
                    runtime=Detector({}),
                    identifier_runtime=Identifier([]),
                    probe=lambda: (True, "ready"),
                )

            self.assertIn("temporary report failed", summary.error or "")
            self.assertFalse(
                any(encounter.glob(f".FinID_{encounter.name}_assets.*"))
            )


if __name__ == "__main__":
    unittest.main()
