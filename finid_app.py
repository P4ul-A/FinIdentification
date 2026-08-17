"""Tk desktop application for local fin identification."""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from findetection_core import DetectionClass, load_detection_classes

from finid.models import (
    DetectionModel,
    IdentificationModel,
    discover_detection_models,
    discover_identification_models,
)
from finid.pipeline import (
    PipelineConfig,
    PipelineSummary,
    detector_has_finsaddle_classes,
    recommended_batches,
    run_pipeline,
)


APP_DIR = Path(__file__).resolve().parent
DETECTION_MODELS_DIR = APP_DIR / "model_recognition"
IDENTIFICATION_MODELS_DIR = APP_DIR / "model_identification"
LOGO_PATH = APP_DIR / "assets" / "logo_orca.png"
LOGO_FRAME_SIZE = (166, 124)
LOGO_SUBSAMPLE = 2


class FinIdentificationApp:
    """Coordinate model discovery, controls, and background identification."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fin Identification")
        self.root.geometry("1060x790")
        self.root.minsize(920, 690)
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.latest_progress: tuple[int, int, str] | None = None
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.running = False
        self.hardware_available: bool | None = None
        self.detection_models: dict[str, DetectionModel] = {}
        self.identification_models: dict[str, IdentificationModel] = {}
        self.class_cache: dict[Path, tuple[DetectionClass, ...]] = {}
        self.current_classes: tuple[DetectionClass, ...] = ()
        self.last_report: Path | None = None
        self.last_reports_root: Path | None = None
        self.last_report_count = 0
        recommendation = recommended_batches()

        pictures = Path.home() / "Pictures"
        self.input_var = tk.StringVar(value=str(pictures if pictures.is_dir() else Path.home()))
        self.output_var = tk.StringVar(value="")
        self.clustering_var = tk.BooleanVar(value=False)
        self.detector_var = tk.StringVar()
        self.identifier_var = tk.StringVar()
        self.threshold_var = tk.DoubleVar(value=0.7)
        self.threshold_help_var = tk.StringVar(value="Minimum identification score")
        self.detector_confidence_var = tk.DoubleVar(value=0.5)
        self.eye_confidence_var = tk.DoubleVar(value=0.5)
        self.image_size_var = tk.IntVar(value=1280)
        self.detector_batch_var = tk.IntVar(value=recommendation.detector_batch)
        self.identifier_batch_var = tk.IntVar(value=recommendation.identifier_batch)
        self.crop_padding_var = tk.DoubleVar(value=0.0)
        self.fp16_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Loading models…")
        self.hardware_status_var = tk.StringVar(value="Checking MPS…")
        self.progress_text_var = tk.StringVar(value="Waiting to start")
        self.progress_value = tk.DoubleVar(value=0)
        self.advanced_visible = False

        self._styles()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll_events)
        threading.Thread(target=self._discover_models, daemon=True).start()
        threading.Thread(target=self._probe_hardware, daemon=True).start()

    def _styles(self) -> None:
        self.colors = {
            "bg": "#f4f7fb",
            "panel": "#ffffff",
            "text": "#172033",
            "muted": "#64748b",
            "accent": "#2563eb",
            "success": "#15803d",
            "danger": "#b91c1c",
            "log_bg": "#111827",
            "log_fg": "#d1d5db",
        }
        self.root.configure(bg=self.colors["bg"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure("Panel.TFrame", background=self.colors["panel"])
        style.configure(
            "Header.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Helvetica", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["muted"],
            font=("Helvetica", 12),
        )
        style.configure(
            "Section.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Helvetica", 13, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
        )
        style.configure(
            "Muted.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
        )
        for widget_style in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(
                widget_style,
                fieldbackground=self.colors["panel"],
                foreground=self.colors["text"],
                insertcolor=self.colors["text"],
                selectbackground=self.colors["accent"],
                selectforeground="#ffffff",
            )
            style.map(
                widget_style,
                fieldbackground=[
                    ("disabled", "#e2e8f0"),
                    ("readonly", self.colors["panel"]),
                ],
                foreground=[
                    ("disabled", self.colors["muted"]),
                    ("readonly", self.colors["text"]),
                ],
            )
        style.configure("Accent.TButton", padding=(18, 9), font=("Helvetica", 11, "bold"))
        style.configure("Tool.TButton", padding=(10, 6))

    def _build(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=(24, 20))
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(3, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        header = ttk.Frame(shell, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Fin Identification", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Choose detector objects to crop, then find confident individual matches",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w")
        logo_frame = tk.Frame(
            header,
            width=LOGO_FRAME_SIZE[0],
            height=LOGO_FRAME_SIZE[1],
            bg=self.colors["bg"],
            borderwidth=0,
            highlightthickness=0,
        )
        logo_frame.grid(
            row=0,
            column=1,
            rowspan=2,
            sticky="ne",
            padx=(18, 0),
        )
        logo_frame.grid_propagate(False)
        try:
            self.logo_photo = tk.PhotoImage(file=str(LOGO_PATH)).subsample(
                LOGO_SUBSAMPLE,
                LOGO_SUBSAMPLE,
            )
            tk.Label(
                logo_frame,
                image=self.logo_photo,
                bg=self.colors["bg"],
                borderwidth=0,
                highlightthickness=0,
            ).place(relx=1.0, rely=0.0, anchor="ne")
        except tk.TclError:
            pass

        setup = ttk.Frame(shell, style="Panel.TFrame", padding=18)
        setup.grid(row=1, column=0, sticky="ew")
        setup.columnconfigure(1, weight=1)
        ttk.Label(setup, text="Set up the run", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        self._path_row(setup, 1)
        ttk.Label(setup, text="Fin recognition model", style="Body.TLabel").grid(
            row=3, column=0, sticky="w", pady=7
        )
        self.detector_combo = ttk.Combobox(
            setup, textvariable=self.detector_var, state="disabled"
        )
        self.detector_combo.grid(row=3, column=1, columnspan=2, sticky="ew", pady=7)
        self.detector_combo.bind("<<ComboboxSelected>>", self._detector_changed)
        self.detector_combo.bind("<KeyRelease>", self._detector_changed)
        ttk.Label(setup, text="Detector objects (FinSaddle IDs)", style="Body.TLabel").grid(
            row=4, column=0, sticky="nw", pady=7
        )
        object_frame = ttk.Frame(setup, style="Panel.TFrame")
        object_frame.grid(row=4, column=1, columnspan=2, sticky="ew", pady=7)
        object_frame.columnconfigure(0, weight=1)
        self.object_list = tk.Listbox(
            object_frame,
            height=4,
            selectmode=tk.EXTENDED,
            exportselection=False,
            state="disabled",
        )
        self.object_list.grid(row=0, column=0, rowspan=2, sticky="ew")
        scrollbar = ttk.Scrollbar(
            object_frame,
            orient="vertical",
            command=self.object_list.yview,
        )
        scrollbar.grid(row=0, column=1, rowspan=2, sticky="ns")
        self.object_list.configure(yscrollcommand=scrollbar.set)
        self.select_all_button = ttk.Button(
            object_frame,
            text="Select all",
            command=self._select_all_objects,
            state="disabled",
            style="Tool.TButton",
        )
        self.select_all_button.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        self.clear_button = ttk.Button(
            object_frame,
            text="Clear",
            command=self._clear_objects,
            state="disabled",
            style="Tool.TButton",
        )
        self.clear_button.grid(row=1, column=2, sticky="ew", padx=(8, 0))
        ttk.Label(setup, text="Fin identification model", style="Body.TLabel").grid(
            row=5, column=0, sticky="w", pady=7
        )
        self.identifier_combo = ttk.Combobox(
            setup, textvariable=self.identifier_var, state="disabled"
        )
        self.identifier_combo.grid(row=5, column=1, columnspan=2, sticky="ew", pady=7)
        self.identifier_combo.bind("<<ComboboxSelected>>", self._identifier_changed)
        self.identifier_combo.bind("<KeyRelease>", self._identifier_changed)
        ttk.Label(setup, text="Good identification threshold", style="Body.TLabel").grid(
            row=6, column=0, sticky="w", pady=7
        )
        threshold = ttk.Spinbox(
            setup,
            from_=0,
            to=1,
            increment=0.01,
            textvariable=self.threshold_var,
            width=10,
        )
        threshold.grid(row=6, column=1, sticky="w", pady=7)
        ttk.Label(setup, textvariable=self.threshold_help_var, style="Muted.TLabel").grid(
            row=6, column=2, sticky="w", padx=(10, 0)
        )

        self.advanced_button = ttk.Button(
            setup,
            text="Show advanced settings ▾",
            command=self._toggle_advanced,
            style="Tool.TButton",
        )
        self.advanced_button.grid(row=7, column=0, columnspan=3, sticky="w", pady=(9, 0))
        self.advanced = ttk.Frame(setup, style="Panel.TFrame")
        self._build_advanced(self.advanced)

        controls = ttk.Frame(shell, style="App.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=14)
        controls.columnconfigure(2, weight=1)
        self.start_button = ttk.Button(
            controls,
            text="Start identification",
            command=self.start_or_stop,
            style="Accent.TButton",
            state="disabled",
        )
        self.start_button.grid(row=0, column=0, rowspan=2, sticky="w")
        self.open_button = ttk.Button(
            controls,
            text="Open reports",
            command=self.open_report,
            state="disabled",
            style="Tool.TButton",
        )
        self.open_button.grid(row=0, column=1, padx=12, sticky="w")
        ttk.Label(controls, textvariable=self.status_var, style="Subtitle.TLabel").grid(
            row=0, column=2, padx=12, sticky="w"
        )
        self.hardware_status_label = ttk.Label(
            controls,
            textvariable=self.hardware_status_var,
            style="Subtitle.TLabel",
        )
        self.hardware_status_label.grid(
            row=0, column=3, sticky="e"
        )
        self.progress_bar = ttk.Progressbar(
            controls,
            variable=self.progress_value,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.grid(
            row=1,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=(12, 0),
            pady=(7, 0),
        )
        ttk.Label(controls, textvariable=self.progress_text_var, style="Subtitle.TLabel").grid(
            row=2, column=1, columnspan=3, sticky="e"
        )

        log_panel = ttk.Frame(shell, style="Panel.TFrame", padding=16)
        log_panel.grid(row=3, column=0, sticky="nsew")
        log_panel.columnconfigure(0, weight=1)
        log_panel.rowconfigure(1, weight=1)
        ttk.Label(log_panel, text="Activity", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.log_box = scrolledtext.ScrolledText(
            log_panel,
            height=9,
            wrap="word",
            state="disabled",
            bg=self.colors["log_bg"],
            fg=self.colors["log_fg"],
            insertbackground="white",
            font=("Menlo", 11),
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log_box.grid(row=1, column=0, sticky="nsew")

    def _path_row(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, text="Folder of JPEG images", style="Body.TLabel").grid(
            row=row, column=0, sticky="w", pady=7
        )
        self.input_entry = ttk.Entry(parent, textvariable=self.input_var)
        self.input_entry.grid(
            row=row, column=1, sticky="ew", pady=7, padx=(0, 8)
        )
        ttk.Button(
            parent, text="Browse…", command=self._browse, style="Tool.TButton"
        ).grid(row=row, column=2, pady=7)

        ttk.Checkbutton(
            parent,
            text="Copy images into encounter clusters",
            variable=self.clustering_var,
            command=self._sync_output_controls,
        ).grid(row=row + 1, column=0, sticky="w", pady=7)
        self.output_entry = ttk.Entry(parent, textvariable=self.output_var, state="disabled")
        self.output_entry.grid(row=row + 1, column=1, sticky="ew", pady=7, padx=(0, 8))
        self.output_button = ttk.Button(
            parent, text="Output…", command=self._browse_output,
            style="Tool.TButton", state="disabled",
        )
        self.output_button.grid(row=row + 1, column=2, pady=7)

    def _build_advanced(self, parent: ttk.Frame) -> None:
        labels = (
            ("Fin / FinSaddle confidence", self.detector_confidence_var, 0.01, 0, 1),
            ("Eye confidence", self.eye_confidence_var, 0.01, 0, 1),
            ("Detector image size", self.image_size_var, 32, 32, 4096),
            ("Detector batch", self.detector_batch_var, 1, 1, 128),
            ("Identifier batch", self.identifier_batch_var, 1, 1, 256),
            ("Crop padding", self.crop_padding_var, 0.01, 0, 1),
        )
        for row, (label, variable, increment, start, end) in enumerate(labels):
            ttk.Label(parent, text=label, style="Body.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=5
            )
            ttk.Spinbox(
                parent,
                from_=start,
                to=end,
                increment=increment,
                textvariable=variable,
                width=12,
            ).grid(row=row, column=1, sticky="w", pady=5)
        ttk.Checkbutton(
            parent,
            text="Use FP16 for faster, lower-memory fin detection",
            variable=self.fp16_var,
        ).grid(row=len(labels), column=0, columnspan=2, sticky="w", pady=5)

    def _toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced.grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 0))
            self.advanced_button.configure(text="Hide advanced settings ▴")
        else:
            self.advanced.grid_remove()
            self.advanced_button.configure(text="Show advanced settings ▾")

    def _browse(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose the folder containing JPEG images",
            initialdir=self.input_var.get() or str(Path.home()),
        )
        if selected:
            self.input_var.set(selected)

    def _browse_output(self) -> None:
        """Choose the root used for clustered encounter output.

        Returns:
            None.
        """
        selected = filedialog.askdirectory(
            title="Choose an empty or FinIdentification-managed output folder",
            initialdir=self.output_var.get() or str(Path.home()),
        )
        if selected:
            self.output_var.set(selected)

    def _sync_output_controls(self) -> None:
        """Enable output controls only when clustering is selected.

        Returns:
            None.
        """
        state = "normal" if self.clustering_var.get() and not self.running else "disabled"
        self.output_entry.configure(state=state)
        self.output_button.configure(state=state)

    def _discover_models(self) -> None:
        detectors, detector_warnings = discover_detection_models(DETECTION_MODELS_DIR)
        identifiers, identifier_warnings = discover_identification_models(
            IDENTIFICATION_MODELS_DIR
        )
        self.events.put(("models", (detectors, identifiers)))
        for warning in detector_warnings + identifier_warnings:
            self.events.put(("log", f"Model warning: {warning}"))

    def _probe_hardware(self) -> None:
        try:
            from findetection_core import probe_runtime

            available, detail = probe_runtime()
        except Exception as exc:
            available, detail = False, f"Hardware check failed: {exc}"
        self.events.put(("hardware", (available, detail)))

    def _identifier_changed(self, _event: object | None = None) -> None:
        descriptor = self._typed_model_choice(
            self.identifier_var.get(),
            self.identification_models,
        )
        if descriptor:
            self.threshold_help_var.set(
                f"Minimum {descriptor.score_label}; higher is stricter"
            )
        else:
            self.threshold_help_var.set("Type or choose an available model name")

    def _detector_changed(self, _event: object | None = None) -> None:
        """Refresh selectable classes after the detector choice changes.

        Parameters:
            _event: Optional Tk selection or key event.

        Returns:
            None.
        """

        self._refresh_classes()

    def _refresh_classes(self) -> None:
        """Load the active detector's classes without blocking Tk.

        Returns:
            None.
        """

        detector = self._typed_model_choice(
            self.detector_var.get(),
            self.detection_models,
        )
        self.current_classes = ()
        self.status_var.set("Reading detector object classes…")
        self.object_list.configure(state="normal")
        self.object_list.delete(0, tk.END)
        self.object_list.insert(tk.END, "Loading model classes…")
        self._update_class_controls()
        if detector is None:
            self._apply_class_error(None, "Choose an available detection model.")
            return
        cached = self.class_cache.get(detector.path)
        if cached is not None:
            self._apply_classes(detector.path, cached)
            return

        def load() -> None:
            """Read class metadata and publish the result to Tk.

            Returns:
                None.
            """

            try:
                classes = load_detection_classes(detector.path)
            except Exception as exc:
                self.events.put(("class_error", (detector.path, str(exc))))
                return
            self.events.put(("classes", (detector.path, classes)))

        threading.Thread(target=load, daemon=True).start()

    def _active_detector_path(self) -> Path | None:
        """Return the selected detector path when the entry is valid.

        Returns:
            Selected detector path, or ``None``.
        """

        detector = self._typed_model_choice(
            self.detector_var.get(),
            self.detection_models,
        )
        return detector.path if detector is not None else None

    def _apply_classes(
        self,
        model_path: Path,
        classes: tuple[DetectionClass, ...],
    ) -> None:
        """Display cached or newly loaded classes and select all.

        Parameters:
            model_path: Model whose metadata was loaded.
            classes: Ordered model class entries.

        Returns:
            None.
        """

        self.class_cache[model_path] = classes
        if model_path != self._active_detector_path():
            return
        self.current_classes = classes
        self.object_list.configure(state="normal")
        self.object_list.delete(0, tk.END)
        for item in classes:
            self.object_list.insert(tk.END, item.display_name)
        self._select_all_objects()
        self._update_class_controls()
        if self.hardware_available and self.identification_models:
            self.status_var.set("Ready")
        self._set_ready_state()

    def _apply_class_error(self, model_path: Path | None, message: str) -> None:
        """Display a class-metadata error for the active detector.

        Parameters:
            model_path: Model whose metadata failed, if known.
            message: User-readable error detail.

        Returns:
            None.
        """

        if model_path is not None and model_path != self._active_detector_path():
            return
        self.current_classes = ()
        self.object_list.configure(state="normal")
        self.object_list.delete(0, tk.END)
        self.object_list.insert(tk.END, "Could not read model classes")
        self._update_class_controls()
        self.log(f"Model class warning: {message}")
        self.status_var.set("Could not read detector object classes")
        self._set_ready_state()

    def _select_all_objects(self) -> None:
        """Select every available detector class.

        Returns:
            None.
        """

        if self.current_classes:
            self.object_list.selection_set(0, tk.END)

    def _clear_objects(self) -> None:
        """Clear every selected detector class.

        Returns:
            None.
        """

        self.object_list.selection_clear(0, tk.END)

    def _selected_class_ids(self) -> tuple[int, ...]:
        """Return selected class IDs in model display order.

        Returns:
            Selected numeric model class IDs.
        """

        return tuple(
            self.current_classes[index].class_id
            for index in self.object_list.curselection()
            if index < len(self.current_classes)
        )

    def _update_class_controls(self) -> None:
        """Synchronize class-list controls with load and run state.

        Returns:
            None.
        """

        enabled = bool(self.current_classes) and not self.running
        state = "normal" if enabled else "disabled"
        self.object_list.configure(state=state)
        self.select_all_button.configure(state=state)
        self.clear_button.configure(state=state)

    @staticmethod
    def _typed_model_choice(text: str, choices: dict[str, Any]) -> Any | None:
        """Resolve typed display names, filenames, or full checkpoint paths."""

        value = text.strip()
        if not value:
            return None
        if value in choices:
            return choices[value]
        folded = value.casefold()
        for label, descriptor in choices.items():
            path = descriptor.path
            if folded in {
                label.casefold(),
                path.name.casefold(),
                str(path).casefold(),
            }:
                return descriptor
        return None

    def _set_ready_state(self) -> None:
        ready = bool(
            self.hardware_available
            and self.detection_models
            and self.identification_models
            and self.current_classes
        )
        if not self.running:
            self.start_button.configure(state="normal" if ready else "disabled")

    def start_or_stop(self) -> None:
        """Start a validated run or request cooperative cancellation."""

        if self.running:
            self.stop_event.set()
            self.start_button.configure(text="Stopping after current batch…", state="disabled")
            self.status_var.set("Stopping safely…")
            return
        try:
            config = self._config()
        except (OSError, ValueError, tk.TclError) as exc:
            messagebox.showerror("Check the settings", str(exc))
            return
        if not self._confirm_clustering_compatibility(config):
            return
        self.running = True
        self.stop_event.clear()
        self.last_report = None
        self.open_button.configure(state="disabled")
        self.progress_value.set(0)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(12)
        self.progress_text_var.set("Scanning folders…")
        self.status_var.set("Scanning…")
        self.start_button.configure(text="Stop safely", state="normal")
        self._set_inputs_enabled(False)
        # Paint the scanning state before the worker begins importing/loading
        # models or walking a very large directory tree.
        self.root.update_idletasks()
        self.worker = threading.Thread(
            target=self._run,
            args=(config,),
            daemon=True,
        )
        self.worker.start()

    def _confirm_clustering_compatibility(self, config: PipelineConfig) -> bool:
        """Warn before clustering with a detector that has no saddle classes.

        Parameters:
            config: Validated configuration about to be started.

        Returns:
            ``True`` when the run may proceed, or ``False`` when cancelled.
        """
        if not config.clustering or detector_has_finsaddle_classes(
            config.detector_classes
        ):
            return True
        return bool(
            messagebox.askokcancel(
                "No FinSaddle classes in detector",
                "Clustering is enabled, but this detector model has no "
                "FinSaddle or saddle classes.\n\n"
                "The IDed and FinSaddle folders will remain empty. Plain fin "
                "detections will go to Rest; qualifying eyes can still go to "
                "Eyes.\n\nContinue with this run?",
                icon="warning",
            )
        )

    def _config(self) -> PipelineConfig:
        input_text = self.input_var.get().strip()
        if not input_text:
            raise ValueError("Choose a folder of JPEG images.")
        input_dir = Path(input_text).expanduser().resolve()
        if not input_dir.is_dir():
            raise ValueError("The chosen JPEG folder does not exist.")
        detector = self._typed_model_choice(
            self.detector_var.get(),
            self.detection_models,
        )
        identifier = self._typed_model_choice(
            self.identifier_var.get(),
            self.identification_models,
        )
        if detector is None:
            raise ValueError(
                "Type or choose a valid fin-recognition model name or filename."
            )
        if identifier is None:
            raise ValueError(
                "Type or choose a valid fin-identification model name or filename."
            )
        if not self.current_classes:
            raise ValueError("Wait for the detector's object list to finish loading.")
        selected_class_ids = self._selected_class_ids()
        if not selected_class_ids:
            raise ValueError("Select at least one object class to identify.")
        return PipelineConfig(
            input_dir=input_dir,
            detector=detector,
            identifier=identifier,
            threshold=float(self.threshold_var.get()),
            detector_confidence=float(self.detector_confidence_var.get()),
            eye_confidence=float(self.eye_confidence_var.get()),
            clustering=bool(self.clustering_var.get()),
            output_root=(
                Path(self.output_var.get()).expanduser().resolve()
                if self.output_var.get().strip()
                else None
            ),
            detector_classes=self.current_classes,
            detector_image_size=int(self.image_size_var.get()),
            detector_batch_size=int(self.detector_batch_var.get()),
            identifier_batch_size=int(self.identifier_batch_var.get()),
            crop_padding=float(self.crop_padding_var.get()),
            detector_fp16=bool(self.fp16_var.get()),
            selected_class_ids=selected_class_ids,
        )

    def _set_inputs_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.detector_combo.configure(state=state)
        self.identifier_combo.configure(state=state)
        self._update_class_controls()
        self._sync_output_controls()

    def _run(self, config: PipelineConfig) -> None:
        summary = run_pipeline(
            config,
            log=lambda message: self.events.put(("log", message)),
            progress=self._publish_progress,
            stop_event=self.stop_event,
        )
        self.events.put(("finished", summary))

    def _publish_progress(self, current: int, total: int, label: str) -> None:
        # Assignment is atomic under CPython. Keeping only the newest progress
        # update prevents a fast pipeline from starving Tk with stale events.
        self.latest_progress = (current, total, label)

    def _poll_events(self) -> None:
        handled = 0
        try:
            while handled < 100:
                name, payload = self.events.get_nowait()
                handled += 1
                if name == "log":
                    self.log(str(payload))
                elif name == "models":
                    detectors, identifiers = payload
                    self.detection_models = {model.name: model for model in detectors}
                    self.identification_models = {model.name: model for model in identifiers}
                    self.detector_combo.configure(values=list(self.detection_models))
                    self.identifier_combo.configure(values=list(self.identification_models))
                    if detectors:
                        self.detector_var.set(detectors[0].name)
                        self._refresh_classes()
                    if identifiers:
                        self.identifier_var.set(identifiers[0].name)
                        self._identifier_changed()
                    self.detector_combo.configure(state="normal" if detectors else "disabled")
                    self.identifier_combo.configure(state="normal" if identifiers else "disabled")
                    self.log(
                        f"Found {len(detectors)} fin-recognition and "
                        f"{len(identifiers)} identification models."
                    )
                    self._set_ready_state()
                elif name == "classes":
                    model_path, classes = payload
                    self._apply_classes(model_path, classes)
                elif name == "class_error":
                    model_path, message = payload
                    self._apply_class_error(model_path, message)
                elif name == "hardware":
                    available, detail = payload
                    self.hardware_available = bool(available)
                    self.hardware_status_var.set(detail)
                    self.hardware_status_label.configure(
                        foreground=(
                            self.colors["success"]
                            if available
                            else self.colors["danger"]
                        )
                    )
                    self.log(detail)
                    if (
                        available
                        and self.detection_models
                        and self.identification_models
                        and self.current_classes
                    ):
                        self.status_var.set("Ready")
                    elif not available:
                        self.status_var.set("MPS unavailable")
                    self._set_ready_state()
                elif name == "finished":
                    self._finished(payload)
        except queue.Empty:
            pass
        latest_progress = self.latest_progress
        if latest_progress is not None:
            self.latest_progress = None
            current, total, label = latest_progress
            if total:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self.progress_value.set(current / total * 100)
                if self.running:
                    folded = label.casefold()
                    if "publish" in folded:
                        self.status_var.set("Publishing output…")
                    elif (
                        "copying" in folded
                        or "staged" in folded
                        or "cluster staging" in folded
                    ):
                        self.status_var.set("Finalizing: copying clusters…")
                    elif "report" in folded or "thumbnail" in folded:
                        self.status_var.set("Finalizing: writing reports…")
                    elif "inventory" in folded or "loading" in folded:
                        self.status_var.set("Preparing inference…")
                    else:
                        self.status_var.set("Detecting and identifying…")
            self.progress_text_var.set(label)
        self.root.after(100, self._poll_events)

    def _finished(self, summary: PipelineSummary) -> None:
        self.running = False
        self.worker = None
        self.latest_progress = None
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self._set_inputs_enabled(True)
        self.start_button.configure(text="Start identification", state="normal")
        self.last_report = summary.root_report if summary.root_report.is_file() else None
        self.last_reports_root = summary.reports_root if summary.report_count else None
        self.last_report_count = summary.report_count
        self.open_button.configure(state="normal" if self.last_reports_root else "disabled")
        self.open_button.configure(
            text=(
                "Open report"
                if summary.report_count == 1
                else "Open reports/output folder"
            )
        )
        self.progress_value.set(100)
        if summary.error:
            self.status_var.set("Finished with a problem")
            messagebox.showerror(
                "Fin identification stopped",
                f"{summary.error}\n\nAny reports that could be written contain partial results.",
            )
        elif summary.stopped:
            self.status_var.set("Stopped safely; partial reports are ready")
        else:
            self.status_var.set("Complete; reports are ready")
        self.progress_text_var.set(
            f"{summary.processed}/{summary.total} images · "
            f"{summary.encounter_count} encounters · {summary.report_count} reports · "
            f"{summary.skipped_undated_count} undated skipped · "
            f"batches {summary.detector_batch_size}/{summary.identifier_batch_size}"
        )

    def log(self, message: str) -> None:
        """Append a user-facing message to the activity log."""

        self.log_box.configure(state="normal")
        self.log_box.insert("end", message.rstrip() + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def open_report(self) -> None:
        """Open one report or the reports/output folder for multiple encounters."""

        if (
            self.last_report_count == 1
            and self.last_report
            and self.last_report.is_file()
        ):
            webbrowser.open(self.last_report.as_uri())
        elif self.last_reports_root and self.last_reports_root.exists():
            webbrowser.open(self.last_reports_root.as_uri())

    def on_close(self) -> None:
        """Close immediately or stop active work before closing."""

        if self.running:
            if not messagebox.askyesno(
                "Stop the current run?",
                "The app will stop after the current batch and write partial reports.",
            ):
                return
            self.stop_event.set()
            self.root.after(250, self._wait_then_close)
        else:
            self.root.destroy()

    def _wait_then_close(self) -> None:
        if self.worker and self.worker.is_alive():
            self.root.after(250, self._wait_then_close)
        else:
            self.root.destroy()


def main() -> None:
    """Create and run the Fin Identification desktop application."""

    root = tk.Tk()
    FinIdentificationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
