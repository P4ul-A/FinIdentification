from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
from PIL import Image

from findetection_core import DetectionClass

from finid.models import (
    DetectionModel,
    IdentificationModel,
    IdentityPrediction,
    IdentifierOutOfMemory,
    IdentifierRuntime,
    discover_identification_models,
)
from finid.pipeline import (
    PipelineConfig,
    discover_encounters,
    inventory_tree,
    iter_jpegs,
    recommended_batches,
    run_pipeline,
)
from finid.reporting import ReportMetadata, report_filename, write_reports
from finid.storage import ResultStore
from finid_app import FinIdentificationApp


class FakeDetectorRuntime:
    def __init__(self, boxes_by_name: dict[str, list[object]] | None = None) -> None:
        self.boxes_by_name = boxes_by_name or {}
        self.effective_batch_size = 2
        self.loaded = False

    def load(self, _config: object, log: object = None) -> None:
        self.loaded = True

    def predict_paths(self, paths: object, stop_event: threading.Event | None = None):
        for path in paths:
            if stop_event is not None and stop_event.is_set():
                from findetection_core import MPSPredictionStopped

                raise MPSPredictionStopped()
            yield SimpleNamespace(
                path=path,
                boxes=tuple(self.boxes_by_name.get(path.name, [])),
                image=None,
            )

    def synchronize(self) -> None:
        pass

    def memory_description(self) -> str:
        return ""

    def close(self) -> None:
        pass


class FakeIdentifierRuntime:
    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = scores or [0.9]
        self.offset = 0
        self.effective_batch_size = 8

    def predict(self, crops: object) -> list[IdentityPrediction]:
        output = []
        for index, _crop in enumerate(crops):
            score_index = min(self.offset, len(self.scores) - 1)
            output.append(
                IdentityPrediction(
                    identity=f"NKW-{self.offset + 1:03d}",
                    score=self.scores[score_index],
                    score_type="cosine similarity",
                )
            )
            self.offset += 1
        return output

    def close(self) -> None:
        pass


def box(
    xyxy: tuple[float, float, float, float],
    confidence: float = 0.9,
    class_id: int = 0,
) -> object:
    return SimpleNamespace(xyxy=xyxy, confidence=confidence, class_id=class_id)


def descriptors() -> tuple[DetectionModel, IdentificationModel]:
    return (
        DetectionModel("Orca", Path("detector.pt")),
        IdentificationModel(
            "ArcFace",
            Path("identifier.pt"),
            "arcface",
            "cosine similarity",
        ),
    )


class BatchRecommendationTests(unittest.TestCase):
    def test_memory_tiers(self) -> None:
        self.assertEqual(recommended_batches(16 * 1024**3).detector_batch, 2)
        self.assertEqual(recommended_batches(16 * 1024**3).identifier_batch, 8)
        self.assertEqual(recommended_batches(24 * 1024**3).detector_batch, 4)
        self.assertEqual(recommended_batches(48 * 1024**3).identifier_batch, 32)
        self.assertEqual(recommended_batches(96 * 1024**3).detector_batch, 12)

    def test_identifier_halves_batch_and_retries_mps_oom(self) -> None:
        runtime = IdentifierRuntime.__new__(IdentifierRuntime)
        runtime.effective_batch_size = 8
        runtime.log_messages = []
        runtime.log = runtime.log_messages.append
        runtime._empty_cache = lambda: None
        attempted: list[int] = []

        def attempt(crops: object) -> list[IdentityPrediction]:
            attempted.append(len(crops))
            if len(crops) > 2:
                raise RuntimeError("MPS backend out of memory")
            return [
                IdentityPrediction("NKW-001", 0.9, "probability")
                for _crop in crops
            ]

        runtime._predict_attempt = attempt
        predictions = runtime.predict([object() for _index in range(6)])
        self.assertEqual(len(predictions), 6)
        self.assertEqual(runtime.effective_batch_size, 2)
        self.assertEqual(attempted[:3], [6, 4, 2])
        self.assertTrue(any("retrying" in message for message in runtime.log_messages))

    def test_identifier_batch_one_oom_is_actionable(self) -> None:
        runtime = IdentifierRuntime.__new__(IdentifierRuntime)
        runtime.effective_batch_size = 1
        runtime.log = lambda _message: None
        runtime._empty_cache = lambda: None
        runtime._predict_attempt = lambda _crops: (_ for _ in ()).throw(
            RuntimeError("MPS allocation failed")
        )
        with self.assertRaisesRegex(IdentifierOutOfMemory, "batch size 1"):
            runtime.predict([object()])


class AppInputTests(unittest.TestCase):
    def test_typed_model_choices_accept_label_filename_and_path(self) -> None:
        detector, identifier = descriptors()
        detector_choices = {detector.name: detector}
        identifier_choices = {identifier.name: identifier}
        self.assertIs(
            FinIdentificationApp._typed_model_choice("orca", detector_choices),
            detector,
        )
        self.assertIs(
            FinIdentificationApp._typed_model_choice(
                "IDENTIFIER.PT", identifier_choices
            ),
            identifier,
        )
        self.assertIs(
            FinIdentificationApp._typed_model_choice(
                str(identifier.path), identifier_choices
            ),
            identifier,
        )
        self.assertIsNone(
            FinIdentificationApp._typed_model_choice("missing", detector_choices)
        )

    def test_app_applies_classes_in_order_and_selects_all(self) -> None:
        app = FinIdentificationApp.__new__(FinIdentificationApp)
        path = Path("detector.pt")
        classes = (
            DetectionClass(0, "fin_left", "Fin Left"),
            DetectionClass(2, "eye_right", "Eye Right"),
        )
        app.class_cache = {}
        app.current_classes = ()
        app.hardware_available = False
        app.object_list = Mock()
        app._active_detector_path = lambda: path
        app._update_class_controls = Mock()
        app._set_ready_state = Mock()

        app._apply_classes(path, classes)

        self.assertEqual(app.class_cache[path], classes)
        self.assertEqual(app.current_classes, classes)
        app.object_list.insert.assert_has_calls(
            [call("end", "Fin Left"), call("end", "Eye Right")]
        )
        app.object_list.selection_set.assert_called_once_with(0, "end")

    def test_selected_class_ids_preserve_list_order(self) -> None:
        app = FinIdentificationApp.__new__(FinIdentificationApp)
        app.current_classes = (
            DetectionClass(2, "fin_left", "Fin Left"),
            DetectionClass(7, "fin_right", "Fin Right"),
        )
        app.object_list = SimpleNamespace(curselection=lambda: (0, 1))

        self.assertEqual(app._selected_class_ids(), (2, 7))

    def test_clustering_without_saddle_classes_requires_confirmation(self) -> None:
        app = FinIdentificationApp.__new__(FinIdentificationApp)
        config = SimpleNamespace(
            clustering=True,
            detector_classes=(DetectionClass(0, "fin_left", "Fin Left"),),
        )

        with patch("finid_app.messagebox.askokcancel", return_value=False) as warning:
            self.assertFalse(app._confirm_clustering_compatibility(config))

        warning.assert_called_once()
        self.assertIn("IDed and FinSaddle", warning.call_args.args[1])

    def test_saddle_capable_model_does_not_show_clustering_warning(self) -> None:
        app = FinIdentificationApp.__new__(FinIdentificationApp)
        config = SimpleNamespace(
            clustering=True,
            detector_classes=(
                DetectionClass(2, "finSaddle_right", "FinSaddle Right"),
            ),
        )

        with patch("finid_app.messagebox.askokcancel") as warning:
            self.assertTrue(app._confirm_clustering_compatibility(config))

        warning.assert_not_called()


class DiscoveryTests(unittest.TestCase):
    def test_discovers_arcface_and_resnet_and_skips_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arcface = root / "orca.pt"
            gallery = root / "orca.gallery.pt"
            resnet = root / "closed.pt"
            for path in (arcface, gallery, resnet):
                path.touch()
            mapping = {"A": 0, "B": 1}
            payloads = {
                arcface: {
                    "class_to_idx": mapping,
                    "embedding_dim": 4,
                    "model_state_dict": {},
                },
                gallery: {
                    "class_to_idx": mapping,
                    "identities": ["A", "B"],
                    "prototypes": torch.randn(2, 4),
                },
                resnet: {
                    "class_to_index": mapping,
                    "model_name": "resnet18",
                    "model_state_dict": {},
                },
            }
            models, warnings = discover_identification_models(
                root, loader=lambda path: payloads[path]
            )
            self.assertEqual([model.kind for model in models], ["resnet", "arcface"])
            self.assertFalse(warnings)
            self.assertEqual(models[1].gallery_path, gallery.resolve())

    def test_reports_missing_gallery_and_unsupported_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arcface = root / "orca.pt"
            unknown = root / "unknown.pt"
            arcface.touch()
            unknown.touch()
            payloads = {
                arcface: {
                    "class_to_idx": {"A": 0},
                    "embedding_dim": 4,
                    "model_state_dict": {},
                },
                unknown: {"weights": {}},
            }
            models, warnings = discover_identification_models(
                root, loader=lambda path: payloads[path]
            )
            self.assertFalse(models)
            self.assertTrue(any("missing" in warning for warning in warnings))
            self.assertTrue(any("unsupported" in warning for warning in warnings))


class InventoryAndReportTests(unittest.TestCase):
    def test_large_inventory_is_disk_backed_and_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-01-01 Large"
            root.mkdir()
            for index in range(2000):
                (root / f"{index:05d}.jpg").touch()
            with ResultStore() as store:
                self.assertEqual(inventory_tree(root, store), 2000)
                iterator = iter_jpegs(root)
                self.assertNotIsInstance(iterator, list)
                self.assertEqual(next(iterator).name, "00000.jpg")
                self.assertEqual(store.total_images(), 2000)

    def test_inventory_reports_cooperative_scan_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-01-01 Scan"
            (root / "Child").mkdir(parents=True)
            for index in range(5):
                (root / f"{index}.jpg").touch()
            updates: list[tuple[int, int]] = []
            with ResultStore() as store:
                total = inventory_tree(
                    root,
                    store,
                    scan_progress=lambda files, directories: updates.append(
                        (files, directories)
                    ),
                    yield_every=2,
                )
            self.assertEqual(total, 5)
            self.assertIn((2, 1), updates)
            self.assertEqual(updates[-1], (5, 2))

    def test_inventory_is_recursive_and_ignores_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-01-01 Trip"
            child = root / "Day 1"
            child.mkdir(parents=True)
            (root / ".DS_Store").write_bytes(b"metadata")
            (root / "notes.txt").write_text("note", encoding="utf-8")
            (root / report_filename(root)).write_text("old", encoding="utf-8")
            Image.new("RGB", (20, 20)).save(root / "A.JPG")
            Image.new("RGB", (20, 20)).save(child / "B.jpeg")
            symlink = root / "linked"
            try:
                symlink.symlink_to(child, target_is_directory=True)
            except OSError:
                symlink = None

            with ResultStore() as store:
                total = inventory_tree(root, store)
                self.assertEqual(total, 2)
                self.assertEqual(store.encounter_count(), 1)
                self.assertEqual([path.name for path in iter_jpegs(root)], ["A.JPG", "B.jpeg"])

    def test_report_name_content_navigation_and_no_image_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-01-01 Tur med øre"
            child = root / "Day #1"
            child.mkdir(parents=True)
            source = child / "orca one.jpg"
            Image.new("RGB", (40, 30), "navy").save(source)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            with ResultStore() as store:
                inventory_tree(root, store)
                store.record_result(
                    source,
                    [
                        {
                            "class_id": 0,
                            "class_name": "fin_left",
                            "side": "LEFT",
                            "confidence": 0.88,
                            "selected": True,
                            "x1": 1,
                            "y1": 2,
                            "x2": 30,
                            "y2": 20,
                        },
                        {
                            "class_id": 1,
                            "class_name": "fin_right",
                            "side": "RIGHT",
                            "confidence": 0.84,
                            "selected": True,
                            "x1": 20,
                            "y1": 10,
                            "x2": 38,
                            "y2": 28,
                        },
                    ],
                    [
                        {
                            "identity": "NKW-001 & friend",
                            "score": 0.91,
                            "score_type": "cosine similarity",
                            "detection_confidence": 0.88,
                            "x1": 1,
                            "y1": 2,
                            "x2": 30,
                            "y2": 20,
                            "detection_index": 0,
                        },
                        {
                            "identity": "NKW-002",
                            "score": 0.87,
                            "score_type": "cosine similarity",
                            "detection_confidence": 0.84,
                            "x1": 20,
                            "y1": 10,
                            "x2": 38,
                            "y2": 28,
                            "detection_index": 1,
                        }
                    ],
                    "IDed",
                    "LEFT",
                )
                count = write_reports(
                    store,
                    ReportMetadata(
                        generated_at=__import__("datetime").datetime.now().astimezone(),
                        completed=True,
                        detector_name="Orca",
                        identifier_name="ArcFace",
                        threshold=0.8,
                        score_label="cosine similarity",
                        elapsed_seconds=2,
                        throughput=0.5,
                    ),
                )
            self.assertEqual(count, 1)
            root_report = root / "FinID_2026-01-01 Tur med øre.html"
            child_report = root_report
            self.assertTrue(root_report.is_file())
            text = child_report.read_text(encoding="utf-8")
            self.assertIn("NKW-001 &amp; friend", text)
            self.assertIn("NKW-002", text)
            self.assertIn("orca%20one.jpg", text)
            self.assertIn("Original:", text)
            self.assertEqual(text.count('class="box"'), 2)
            self.assertIn("--box-color:#0369a1", text)
            self.assertIn("--box-color:#b45309", text)
            self.assertNotIn("box (1, 2)", text)
            self.assertNotIn("Detection 88.0%", text)
            self.assertNotIn("0 0 0 1px", text)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(
                sorted(path.name for path in child.iterdir()),
                ["orca one.jpg"],
            )


class PipelineTests(unittest.TestCase):
    def test_pipeline_uses_selected_class_and_writes_accepted_only_reports(self) -> None:
        detector, identifier = descriptors()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-01-01 Encounter"
            child = root / "Camera A"
            child.mkdir(parents=True)
            accepted = child / "accepted.jpg"
            rejected = child / "rejected.jpeg"
            Image.new("RGB", (100, 80), "white").save(accepted)
            Image.new("RGB", (100, 80), "gray").save(rejected)
            (child / "video.mov").write_bytes(b"not a jpeg")
            runtime = FakeDetectorRuntime(
                {
                    "accepted.jpg": [
                        box((10, 10, 70, 60), 0.95, 0),
                        box((1, 1, 5, 5), 0.99, 1),
                    ],
                    "rejected.jpeg": [box((5, 5, 60, 50), 0.8, 0)],
                }
            )
            summary = run_pipeline(
                PipelineConfig(
                    root,
                    detector,
                    identifier,
                    threshold=0.8,
                    detector_batch_size=2,
                    identifier_batch_size=8,
                    detector_classes=(
                        DetectionClass(0, "finSaddle_left", "FinSaddle Left"),
                        DetectionClass(1, "eye_right", "Eye Right"),
                    ),
                    selected_class_ids=(0,),
                ),
                runtime=runtime,
                identifier_runtime=FakeIdentifierRuntime([0.9, 0.4]),
                probe=lambda: (True, "MPS test ready"),
            )
            self.assertTrue(summary.completed)
            self.assertEqual(summary.processed, 2)
            self.assertEqual(summary.report_count, 1)
            report = root / report_filename(root)
            text = report.read_text(encoding="utf-8")
            self.assertIn("accepted.jpg", text)
            self.assertIn("rejected.jpeg", text)
            self.assertIn("Destination: FinSaddle", text)

    def test_pipeline_can_use_a_nonzero_model_class(self) -> None:
        detector, identifier = descriptors()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-01-01 Encounter"
            root.mkdir()
            source = root / "objects.jpg"
            Image.new("RGB", (100, 80), "white").save(source)
            runtime = FakeDetectorRuntime(
                {
                    source.name: [
                        box((10, 10, 60, 60), 0.95, 0),
                        box((70, 10, 90, 30), 0.90, 1),
                    ]
                }
            )

            summary = run_pipeline(
                PipelineConfig(
                    root,
                    detector,
                    identifier,
                    detector_classes=(
                        DetectionClass(0, "fin_left", "Fin Left"),
                        DetectionClass(1, "finSaddle_right", "FinSaddle Right"),
                    ),
                    selected_class_ids=(1,),
                ),
                runtime=runtime,
                identifier_runtime=FakeIdentifierRuntime([0.9]),
                probe=lambda: (True, "MPS test ready"),
            )

            self.assertTrue(summary.completed)
            text = (root / report_filename(root)).read_text(encoding="utf-8")
            self.assertIn("finSaddle_right 0.900", text)
            self.assertIn("left:70.0000%", text)

    def test_pipeline_config_rejects_an_empty_explicit_selection(self) -> None:
        detector, identifier = descriptors()

        with self.assertRaisesRegex(ValueError, "Select at least one"):
            PipelineConfig(
                Path("."),
                detector,
                identifier,
                selected_class_ids=(),
            )

    def test_empty_tree_still_gets_reports_without_loading_models(self) -> None:
        detector, identifier = descriptors()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-01-01 Empty"
            (root / "Child").mkdir(parents=True)
            summary = run_pipeline(
                PipelineConfig(root, detector, identifier),
                runtime=FakeDetectorRuntime(),
                identifier_runtime=FakeIdentifierRuntime(),
                probe=lambda: (_ for _ in ()).throw(AssertionError("MPS should not be probed")),
            )
            self.assertTrue(summary.completed)
            self.assertEqual(summary.total, 0)
            self.assertEqual(summary.report_count, 1)

    def test_pre_stopped_run_writes_partial_reports(self) -> None:
        detector, identifier = descriptors()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026-01-01 Stopped"
            root.mkdir()
            Image.new("RGB", (40, 40)).save(root / "one.jpg")
            event = threading.Event()
            event.set()
            summary = run_pipeline(
                PipelineConfig(root, detector, identifier),
                runtime=FakeDetectorRuntime(),
                identifier_runtime=FakeIdentifierRuntime(),
                probe=lambda: (True, "MPS test ready"),
                stop_event=event,
            )
            self.assertTrue(summary.stopped)
            self.assertFalse(summary.completed)
            text = (root / report_filename(root)).read_text(encoding="utf-8")
            self.assertIn("Partial", text)
            self.assertIn("completed images only", text)


if __name__ == "__main__":
    unittest.main()
