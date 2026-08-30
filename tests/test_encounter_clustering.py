from __future__ import annotations

import re
import tempfile
import threading
import tracemalloc
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from findetection_core import DetectionClass

import finid.pipeline as pipeline_module
from finid.models import DetectionModel, IdentificationModel, IdentityPrediction
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

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.index = 0
        self.effective_batch_size = 8
        self.batch_sizes: list[int] = []

    def predict(self, crops: object) -> list[IdentityPrediction]:
        """Return one prediction for each crop."""
        self.batch_sizes.append(len(crops))
        predictions: list[IdentityPrediction] = []
        for _crop in crops:
            score = self.scores[self.index]
            self.index += 1
            predictions.append(IdentityPrediction(f"NKW-{self.index:03d}", score, "probability"))
        return predictions

    def close(self) -> None:
        """Close the fake identifier."""


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
                any("reports and thumbnails" in label for label in labels)
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
                "id.jpg": [detection(3)],
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
            self.assertTrue((target / "RIGHT/IDed/NKW-003_id.jpg").is_file())
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
            self.assertIn("Problem:", text)

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

    def test_large_report_uses_paged_chunks_and_thumbnails(self) -> None:
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
                    runtime=Detector({}),
                    identifier_runtime=Identifier([]),
                    probe=lambda: (True, "ready"),
                )

            report = output / encounter.name / report_filename(encounter)
            assets = output / encounter.name / f"FinID_{encounter.name}_assets"
            self.assertTrue(summary.completed)
            self.assertIn("FINID_SECTIONS", report.read_text(encoding="utf-8"))
            self.assertEqual(len(list((assets / "thumbs").glob("*.jpg"))), 5)
            self.assertEqual(len(list(assets.glob("section-*.js"))), 3)
            self.assertTrue((assets / ".finid-report-assets").is_file())

    def test_report_setup_failure_removes_asset_staging_directory(self) -> None:
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
