"""Generate navigable HTML reports for fin-identification results."""

from __future__ import annotations

import html
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from .storage import ResultStore


REPORT_PREFIX = "FinID_"
BOX_COLORS = (
    "#0369a1",
    "#b45309",
    "#be185d",
    "#6d28d9",
    "#0f766e",
    "#b91c1c",
    "#4d7c0f",
    "#334155",
)


def report_filename(directory: Path) -> str:
    """Return the generated report filename for a directory."""

    folder_name = directory.name or "root"
    return f"{REPORT_PREFIX}{folder_name}.html"


def is_generated_report(path: Path) -> bool:
    """Return whether a path has the generated report naming scheme."""

    return (
        path.suffix.lower() == ".html"
        and path.name.startswith(REPORT_PREFIX)
    )


@dataclass(frozen=True)
class ReportMetadata:
    """Store run metadata displayed in every generated report."""

    generated_at: datetime
    completed: bool
    detector_name: str
    identifier_name: str
    threshold: float
    score_label: str
    elapsed_seconds: float
    throughput: float
    message: str = ""


def _write_header(handle: object, directory: Path, metadata: ReportMetadata) -> None:
    title = f"Fin identifications — {directory.name or directory}"
    status = "Complete" if metadata.completed else "Partial"
    status_class = "complete" if metadata.completed else "partial"
    handle.write(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dce4ef;--panel:#fff;--bg:#f4f7fb;--blue:#2563eb;--green:#15803d;--amber:#b45309}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px 20px 60px}} h1{{margin:.2rem 0;font-size:clamp(1.8rem,4vw,2.7rem)}} h2{{margin-top:2rem}}
.eyebrow{{color:var(--blue);font-weight:700}} .muted{{color:var(--muted)}} .status{{display:inline-block;padding:.3rem .7rem;border-radius:999px;font-weight:700}}
.complete{{background:#dcfce7;color:var(--green)}} .partial{{background:#ffedd5;color:var(--amber)}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:22px 0}}
.metric,.card,.notice{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 18px #2538580d}}
.metric{{padding:16px}} .metric strong{{display:block;font-size:1.6rem}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}
.card{{overflow:hidden}} .image-wrap{{display:block;position:relative;background:#e8eef6}} .card img{{display:block;width:100%;height:auto}}
.box{{position:absolute;border:3px solid var(--box-color);box-shadow:0 2px 8px #0009;pointer-events:none}}
.identity-name{{font-weight:800}} .color-key{{display:inline-block;width:.75rem;height:.75rem;border-radius:3px;margin-right:.35rem}}
.fin{{padding:10px 0;border-top:1px solid var(--line)}} .score{{font-weight:700;color:var(--green)}} .notice{{padding:16px;margin:14px 0}}
a{{color:var(--blue)}} code{{overflow-wrap:anywhere}} ul{{padding-left:1.3rem}}
</style>
</head>
<body><main>
<div class="eyebrow">Fin Identification Report</div>
<h1>{html.escape(directory.name or str(directory))}</h1>
<p><span class="status {status_class}">{status}</span> <span class="muted">Generated {html.escape(metadata.generated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))}</span></p>
<p class="muted">Detector: {html.escape(metadata.detector_name)} · Identifier: {html.escape(metadata.identifier_name)} · Acceptance: {html.escape(metadata.score_label)} ≥ {metadata.threshold:.3f}</p>
"""
    )
    if metadata.message:
        handle.write(f'<div class="notice">{html.escape(metadata.message)}</div>\n')


def _relative_report_link(source: Path, target: Path) -> str:
    relative = os.path.relpath(target / report_filename(target), source)
    return quote(relative, safe="/")


def write_reports(store: ResultStore, metadata: ReportMetadata) -> int:
    """Stream all directory reports from SQLite and atomically replace pages."""

    written = 0
    for row in store.directories():
        directory = Path(row["path"])
        report_path = directory / report_filename(directory)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                _write_header(handle, directory, metadata)
                handle.write('<section class="summary">\n')
                metrics = (
                    ("JPEGs", row["jpeg_count"]),
                    ("Processed", row["processed_count"]),
                    ("Selected objects", row["detected_fins"]),
                    ("Accepted images", row["accepted_images"]),
                    ("Rejected images", row["rejected_images"]),
                    ("Failures", row["failure_count"]),
                    ("Skipped files", row["skipped_count"]),
                    ("Images/second", f"{metadata.throughput:.2f}"),
                )
                for label, value in metrics:
                    handle.write(
                        f'<div class="metric"><strong>{html.escape(str(value))}</strong>{html.escape(label)}</div>\n'
                    )
                handle.write("</section>\n")

                parent_value = row["parent_path"]
                child_iterator = store.child_directories(directory)
                first_child = next(child_iterator, None)
                has_parent = bool(
                    parent_value and store.has_directory(Path(parent_value))
                )
                if has_parent or first_child is not None:
                    handle.write("<nav><strong>Folder reports:</strong> ")
                    separator = ""
                if parent_value and store.has_directory(Path(parent_value)):
                    parent = Path(parent_value)
                    handle.write(
                        f'{separator}<a href="{html.escape(_relative_report_link(directory, parent), quote=True)}">← {html.escape(parent.name)}</a>'
                    )
                    separator = " · "
                if first_child is not None:
                    handle.write(
                        f'{separator}<a href="{html.escape(_relative_report_link(directory, first_child), quote=True)}">{html.escape(first_child.name)} →</a>'
                    )
                    separator = " · "
                    for child in child_iterator:
                        handle.write(
                            f'{separator}<a href="{html.escape(_relative_report_link(directory, child), quote=True)}">{html.escape(child.name)} →</a>'
                        )
                if has_parent or first_child is not None:
                    handle.write("</nav>\n")

                handle.write("<h2>Accepted identifications</h2>\n")
                accepted_any = False
                handle.write('<section class="cards">\n')
                for image_row in store.accepted_images(directory):
                    accepted_any = True
                    filename = str(image_row["filename"])
                    source = quote(filename, safe="")
                    fins = list(store.accepted_fins(int(image_row["id"])))
                    identity_colors: dict[str, str] = {}
                    for fin in fins:
                        identity = str(fin["identity"])
                        if identity not in identity_colors:
                            identity_colors[identity] = BOX_COLORS[
                                len(identity_colors) % len(BOX_COLORS)
                            ]
                    image_width = 0
                    image_height = 0
                    try:
                        with Image.open(directory / filename) as source_image:
                            image_width, image_height = source_image.size
                    except OSError:
                        pass
                    handle.write(
                        f'<article class="card"><a class="image-wrap" href="{html.escape(source, quote=True)}">'
                        f'<img loading="lazy" src="{html.escape(source, quote=True)}" alt="{html.escape(filename, quote=True)}">'
                    )
                    if image_width > 0 and image_height > 0:
                        for fin in fins:
                            left = max(0.0, min(100.0, float(fin["x1"]) / image_width * 100))
                            top = max(0.0, min(100.0, float(fin["y1"]) / image_height * 100))
                            right = max(left, min(100.0, float(fin["x2"]) / image_width * 100))
                            bottom = max(top, min(100.0, float(fin["y2"]) / image_height * 100))
                            identity = str(fin["identity"])
                            color = identity_colors[identity]
                            handle.write(
                                '<span class="box" '
                                f'style="left:{left:.4f}%;top:{top:.4f}%;'
                                f'width:{right - left:.4f}%;height:{bottom - top:.4f}%;'
                                f'--box-color:{color};"></span>'
                            )
                    handle.write(
                        f'</a><div class="body"><strong>{html.escape(filename)}</strong>\n'
                    )
                    for fin in fins:
                        identity = str(fin["identity"])
                        color = identity_colors[identity]
                        handle.write(
                            '<div class="fin">'
                            f'<div><span class="color-key" style="background:{color}"></span>'
                            f'<span class="identity-name" style="color:{color}">{html.escape(identity)}</span> '
                            f'<span class="score">{float(fin["score"]):.3f} {html.escape(str(fin["score_type"]))}</span></div>'
                            "</div>\n"
                        )
                    handle.write("</div></article>\n")
                handle.write("</section>\n")
                if not accepted_any:
                    handle.write('<div class="notice">No identifications met the selected threshold in this folder.</div>\n')

                if row["skipped_count"]:
                    handle.write("<h2>Skipped non-JPEG files</h2><div class=\"notice\"><ul>\n")
                    for filename in store.skipped_files(directory):
                        handle.write(f"<li>{html.escape(filename)}</li>\n")
                    handle.write("</ul></div>\n")
                if row["failure_count"]:
                    handle.write("<h2>Processing problems</h2><div class=\"notice\"><ul>\n")
                    for failure in store.failures(directory):
                        handle.write(
                            f"<li><strong>{html.escape(str(failure['filename']))}</strong>: "
                            f"{html.escape(str(failure['message']))}</li>\n"
                        )
                    handle.write("</ul></div>\n")
                handle.write(
                    f'<p class="muted">Elapsed time: {metadata.elapsed_seconds:.1f} seconds. Source images were not copied or modified.</p>'
                    "</main></body></html>\n"
                )
            os.replace(temporary_name, report_path)
            written += 1
        except Exception:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
            raise
    return written
