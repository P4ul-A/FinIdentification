"""Stream one HTML report per encounter from SQLite run state."""

from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote

from PIL import Image

from .storage import ResultStore


REPORT_PREFIX = "FinID_"
BOX_COLORS = ("#0369a1", "#b45309", "#be185d", "#6d28d9", "#0f766e", "#b91c1c")
VIRTUALIZE_THRESHOLD = 500
REPORT_PAGE_SIZE = 100
ASSET_MARKER = ".finid-report-assets"


def report_filename(directory: Path) -> str:
    """Return the report filename for an encounter root."""
    return f"{REPORT_PREFIX}{directory.name or 'root'}.html"


def is_generated_report(path: Path) -> bool:
    """Return whether a path uses the generated report naming scheme."""
    return path.suffix.lower() == ".html" and path.name.startswith(REPORT_PREFIX)


def is_generated_report_assets(path: Path) -> bool:
    """Return whether a directory is a managed generated-report asset tree."""
    return (
        path.is_dir()
        and path.name.startswith(REPORT_PREFIX)
        and path.name.endswith("_assets")
        and (path / ASSET_MARKER).is_file()
    )


@dataclass(frozen=True)
class ReportMetadata:
    """Run metadata displayed in every encounter report."""

    generated_at: datetime
    completed: bool
    detector_name: str
    identifier_name: str
    threshold: float
    score_label: str
    elapsed_seconds: float
    throughput: float
    fin_confidence: float = 0.25
    eye_confidence: float = 0.25
    exclude_right_identification: bool = False
    message: str = ""


def _header(
    handle: object,
    encounter: Path,
    metadata: ReportMetadata,
    counts: dict[str, int],
    identity_count: int,
) -> None:
    """Write report header and encounter totals.

    Parameters:
        handle: Open report output stream.
        encounter: Source encounter represented by the report.
        metadata: Run metadata displayed above the summary.
        counts: Image counts by cluster category.
        identity_count: Distinct accepted orcas among IDed images.

    Returns:
        None.
    """
    title = f"Fin identifications — {encounter.name}"
    status = "Complete" if metadata.completed else "Partial"
    status_class = "complete" if metadata.completed else "partial"
    handle.write(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dce4ef;--panel:#fff;--bg:#f4f7fb;--blue:#2563eb;--green:#15803d;--amber:#b45309}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1280px;margin:auto;padding:30px 20px 60px}}h1{{margin:.2rem 0}}h2{{margin-top:2rem}}h3{{margin-top:1.5rem}}
.muted{{color:var(--muted)}}.status{{display:inline-block;padding:.3rem .7rem;border-radius:999px;font-weight:700}}.complete{{background:#dcfce7;color:var(--green)}}.partial{{background:#ffedd5;color:var(--amber)}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:20px 0}}.metric,.card,.notice{{background:#fff;border:1px solid var(--line);border-radius:12px}}.metric{{padding:14px}}.metric strong{{display:block;font-size:1.5rem}}.metric small{{display:block;color:var(--muted);margin-top:2px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px}}.compact{{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}}.card{{overflow:hidden}}.image-wrap{{display:block;position:relative;background:#e8eef6}}.card img{{display:block;width:100%;height:auto}}.body{{padding:12px}}.box{{position:absolute;border:3px solid var(--box-color);box-shadow:0 2px 7px #0008;pointer-events:none}}.detail{{border-top:1px solid var(--line);padding-top:7px;margin-top:7px}}.notice{{padding:14px}}code{{overflow-wrap:anywhere}}a{{color:var(--blue)}}
.pager{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:18px 0}}.pager select,.pager button{{font:inherit;padding:8px 10px}}.pager button:disabled{{opacity:.45}}
</style></head><body><main><div style="color:var(--blue);font-weight:700">Fin Identification Report</div>
<h1>{html.escape(encounter.name)}</h1><p><span class="status {status_class}">{status}</span> <span class="muted">Generated {html.escape(metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S %Z'))}</span></p>
<p class="muted">Detector: {html.escape(metadata.detector_name)} · Identifier: {html.escape(metadata.identifier_name)} · Identification ≥ {metadata.threshold:.3f} · Fin/FinSaddle ≥ {metadata.fin_confidence:.3f} · Eyes ≥ {metadata.eye_confidence:.3f}</p>""")
    if metadata.exclude_right_identification:
        handle.write(
            '<div class="notice"><strong>RIGHT-side identification disabled:'
            "</strong> RIGHT FinSaddle crops were not sent for identification. "
            "RIGHT FinSaddle and eye detections were still classified, but "
            "RIGHT eye images were not linked to camera-burst saddle results.</div>"
        )
    if metadata.message:
        handle.write(f'<div class="notice">{html.escape(metadata.message)}</div>')
    handle.write('<section class="summary">')
    orca_label = "orca" if identity_count == 1 else "orcas"
    image_label = "image" if counts["IDed"] == 1 else "images"
    for label in ("Total", "IDed", "FinSaddle", "Eyes", "Rest", "Failures"):
        detail = (
            f'<small>{identity_count} {orca_label} total · '
            f'{counts["IDed"]} {image_label}</small>'
            if label == "IDed"
            else ""
        )
        handle.write(
            f'<div class="metric"><strong>{counts[label]}</strong>{label}{detail}</div>'
        )
    handle.write("</section>")


def _image_href(row: object, report_root: Path) -> str:
    """Return a report-relative URL for a source or copied image."""
    copied = row["copied_filename"]
    if copied:
        category = str(row["cluster_category"])
        side = row["cluster_side"]
        relative = Path(str(side)) / category / str(copied) if side else Path(category) / str(copied)
        return quote(relative.as_posix(), safe="/")
    return quote(os.path.relpath(str(row["path"]), report_root), safe="/")


def _report_details(
    store: ResultStore,
    row: object,
) -> tuple[list[object], list[object], list[object]]:
    """Load the report details needed for one image.

    Parameters:
        store: Disk-backed run results.
        row: Source-image database row.

    Returns:
        Accepted identities, rejected candidates, and raw detections. Ranked
        candidate queries are limited to FinSaddle cards.
    """
    image_id = int(row["id"])
    identities = (
        list(store.identities(image_id))
        if row["cluster_category"] == "IDed"
        else []
    )
    candidates = (
        list(store.identity_candidates(image_id))
        if row["cluster_category"] == "FinSaddle"
        else []
    )
    return identities, candidates, list(store.detections(image_id))


def _detection_colors(detections: Iterable[object]) -> dict[int, str]:
    """Map stored detection IDs to their report overlay colors.

    Parameters:
        detections: Detections in the same order used to draw overlays.

    Returns:
        Detection database IDs mapped to their corresponding box colors.
    """
    return {
        int(detection["id"]): BOX_COLORS[index % len(BOX_COLORS)]
        for index, detection in enumerate(detections)
    }


def _identity_html(identity: object, detection_colors: dict[int, str]) -> str:
    """Render one identity with its associated detection-box color.

    Parameters:
        identity: Stored identity row, optionally linked to a detection.
        detection_colors: Detection IDs mapped to report colors.

    Returns:
        Escaped identity details with a colored identity label when associated.
    """
    label = html.escape(str(identity["identity"]))
    detection_id = identity["detection_id"]
    color = (
        detection_colors.get(int(detection_id))
        if detection_id is not None
        else None
    )
    if color is not None:
        label = f'<span style="color:{color};font-weight:700">{label}</span>'
    return (
        f'{label} {float(identity["score"]):.3f} '
        f'{html.escape(str(identity["score_type"]))}'
    )


def _card(
    handle: object,
    store: ResultStore,
    row: object,
    report_root: Path,
    advance: Callable[[], None] | None = None,
) -> None:
    """Write one image card with assignment, scores, and overlays."""
    identities, candidates, detections = _report_details(store, row)
    href = _image_href(row, report_root)
    width = height = 0
    if detections:
        try:
            with Image.open(Path(str(row["path"]))) as image:
                width, height = image.size
        except OSError:
            pass
    handle.write(f'<article class="card"><a class="image-wrap" href="{html.escape(href, quote=True)}"><img loading="lazy" src="{html.escape(href, quote=True)}" alt="{html.escape(str(row["filename"]), quote=True)}">')
    if width and height:
        for index, detection in enumerate(detections):
            left = max(0.0, min(100.0, float(detection["x1"]) / width * 100))
            top = max(0.0, min(100.0, float(detection["y1"]) / height * 100))
            right = max(left, min(100.0, float(detection["x2"]) / width * 100))
            bottom = max(top, min(100.0, float(detection["y2"]) / height * 100))
            handle.write(f'<span class="box" style="left:{left:.4f}%;top:{top:.4f}%;width:{right-left:.4f}%;height:{bottom-top:.4f}%;--box-color:{BOX_COLORS[index % len(BOX_COLORS)]}"></span>')
    shown_name = str(row["copied_filename"] or row["filename"])
    handle.write(f'</a><div class="body"><strong>{html.escape(shown_name)}</strong><div class="muted">Original: <code>{html.escape(str(row["relative_path"]))}</code></div><div>Destination: {html.escape(str(row["cluster_category"]))} · Side: {html.escape(str(row["cluster_side"] or "—"))}</div>')
    if identities:
        detection_colors = _detection_colors(detections)
        handle.write('<div class="detail"><strong>Identities:</strong> ' + " · ".join(
            _identity_html(item, detection_colors) for item in identities
        ) + "</div>")
    if candidates:
        detection_colors = _detection_colors(detections)
        handle.write('<div class="detail"><strong>Candidates:</strong> ' + " · ".join(
            _identity_html(item, detection_colors) for item in candidates
        ) + "</div>")
    if detections:
        labels = []
        for index, item in enumerate(detections):
            color = BOX_COLORS[index % len(BOX_COLORS)]
            labels.append(
                f'{html.escape(str(item["class_name"]))} '
                f'<span style="color:{color};font-weight:700">'
                f'{float(item["confidence"]):.3f}</span>'
            )
        handle.write(
            '<div class="detail"><strong>Detections:</strong> '
            + " · ".join(labels)
            + "</div>"
        )
    if row["failure_message"]:
        handle.write(f'<div class="detail">Problem: {html.escape(str(row["failure_message"]))}</div>')
    handle.write("</div></article>")
    if advance is not None:
        advance()


def _section(
    handle: object,
    title: str,
    rows: Iterable[object],
    count: int,
    store: ResultStore,
    root: Path,
    compact: bool = False,
    advance: Callable[[], None] | None = None,
) -> None:
    """Write one category section."""
    handle.write(f'<h2>{html.escape(title)} <span class="muted">({count})</span></h2>')
    if not count:
        handle.write('<div class="notice">No images in this category.</div>')
        return
    handle.write(f'<section class="cards{" compact" if compact else ""}">')
    for row in rows:
        _card(handle, store, row, root, advance)
    handle.write("</section>")


def _asset_directory_name(encounter: Path) -> str:
    """Return the managed asset directory name for an encounter report."""
    return f"{REPORT_PREFIX}{encounter.name}_assets"


def _image_dimensions(source: Path) -> tuple[int, int] | None:
    """Return image dimensions when the source can be opened.

    Parameters:
        source: Image whose dimensions should be read.

    Returns:
        The width and height, or ``None`` when the image cannot be opened.
    """
    try:
        with Image.open(source) as image:
            return image.size
    except OSError:
        return None


def _virtual_card_data(
    store: ResultStore,
    row: object,
    report_root: Path,
) -> dict[str, object]:
    """Build serializable card data for a paged report.

    Parameters:
        store: Disk-backed run results.
        row: Source-image database row.
        report_root: Directory containing the generated report.

    Returns:
        JSON-compatible card content, including colored detection overlays.
    """
    identities, candidates, detections = _report_details(store, row)
    detection_colors = _detection_colors(detections)
    href = _image_href(row, report_root)
    dimensions = _image_dimensions(Path(str(row["path"])))
    overlays: list[dict[str, object]] = []
    if dimensions is not None:
        width, height = dimensions
        for index, detection in enumerate(detections):
            left = max(0.0, min(100.0, float(detection["x1"]) / width * 100))
            top = max(0.0, min(100.0, float(detection["y1"]) / height * 100))
            right = max(left, min(100.0, float(detection["x2"]) / width * 100))
            bottom = max(top, min(100.0, float(detection["y2"]) / height * 100))
            overlays.append(
                {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                    "color": BOX_COLORS[index % len(BOX_COLORS)],
                }
            )
    return {
        "href": href,
        "src": href,
        "filename": str(row["copied_filename"] or row["filename"]),
        "relative_path": str(row["relative_path"]),
        "category": str(row["cluster_category"]),
        "side": str(row["cluster_side"] or "—"),
        "identities": [
            [
                str(item["identity"]),
                float(item["score"]),
                str(item["score_type"]),
                detection_colors.get(int(item["detection_id"]))
                if item["detection_id"] is not None
                else None,
            ]
            for item in identities
        ],
        "candidates": [
            [
                str(item["identity"]),
                float(item["score"]),
                str(item["score_type"]),
                detection_colors.get(int(item["detection_id"])),
            ]
            for item in candidates
        ],
        "detections": [
            [
                str(item["class_name"]),
                float(item["confidence"]),
                BOX_COLORS[index % len(BOX_COLORS)],
            ]
            for index, item in enumerate(detections)
        ],
        "overlays": overlays,
        "failure": str(row["failure_message"] or ""),
    }


def _write_virtual_section(
    handle: object,
    store: ResultStore,
    rows: Iterable[object],
    section_index: int,
    report_root: Path,
    advance: Callable[[], None],
) -> list[str]:
    """Build one gallery section as bounded in-report pages.

    Parameters:
        handle: Open report output stream.
        store: Disk-backed run results.
        rows: Images belonging to the section.
        section_index: Stable index used for embedded page IDs.
        report_root: Directory containing the generated report.
        advance: Callback invoked after each card is built.

    Returns:
        IDs of embedded JSON page elements.
    """
    page_ids: list[str] = []
    cards: list[dict[str, object]] = []

    def write_page() -> None:
        """Write the current bounded card page into the report."""
        page_id = f"finid-page-{section_index}-{len(page_ids)}"
        payload = json.dumps(cards, ensure_ascii=True, separators=(",", ":"))
        safe_payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        handle.write(
            f'<script type="application/json" id="{page_id}">{safe_payload}</script>'
        )
        page_ids.append(page_id)

    for row in rows:
        cards.append(_virtual_card_data(store, row, report_root))
        advance()
        if len(cards) == REPORT_PAGE_SIZE:
            write_page()
            cards = []
    if cards:
        write_page()
    return page_ids


def _write_virtual_body(
    handle: object,
    store: ResultStore,
    encounter: Path,
    report_root: Path,
    counts: dict[str, int],
    advance: Callable[[], None],
) -> None:
    """Write a self-contained paged gallery and its card data."""
    sections: list[dict[str, object]] = []
    section_index = 0
    for group in store.primary_identity_groups(encounter):
        identity = str(group["identity"])
        pages = _write_virtual_section(
            handle,
            store,
            store.identified_images(encounter, identity),
            section_index,
            report_root,
            advance,
        )
        sections.append(
            {
                "title": f"Identified — {identity}",
                "count": int(group["image_count"]),
                "pages": pages,
            }
        )
        section_index += 1
    for category in ("FinSaddle", "Eyes", "Rest"):
        pages = _write_virtual_section(
            handle,
            store,
            store.encounter_images(encounter, category),
            section_index,
            report_root,
            advance,
        )
        sections.append(
            {"title": category, "count": counts[category], "pages": pages}
        )
        section_index += 1
    handle.write(
        '<h2>Encounter gallery</h2><div class="pager">'
        '<label>Section <select id="finid-section"></select></label>'
        '<button id="finid-prev" type="button">← Previous</button>'
        '<span id="finid-page" class="muted"></span>'
        '<button id="finid-next" type="button">Next →</button></div>'
        '<div id="finid-loading" class="notice">Loading gallery page…</div>'
        '<section id="finid-cards" class="cards"></section>'
    )
    handle.write("<script>const FINID_SECTIONS=")
    manifest = json.dumps(sections, ensure_ascii=True, separators=(",", ":"))
    handle.write(
        manifest.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    handle.write(
        r''';
const sectionSelect=document.getElementById("finid-section"),cards=document.getElementById("finid-cards"),pageLabel=document.getElementById("finid-page"),loading=document.getElementById("finid-loading"),previous=document.getElementById("finid-prev"),next=document.getElementById("finid-next");let sectionIndex=0,pageIndex=0;
function node(tag,text,cls){const value=document.createElement(tag);if(text!==undefined)value.textContent=text;if(cls)value.className=cls;return value}
function renderRanked(body,label,items){if(!items.length)return;const detail=node("div",undefined,"detail");detail.append(node("strong",`${label}: `));items.forEach((value,index)=>{if(index)detail.append(" · ");const identity=node("span",value[0]);if(value[3])identity.style.cssText=`color:${value[3]};font-weight:700`;detail.append(identity,` ${value[1].toFixed(3)} ${value[2]}`)});body.append(detail)}
function render(items){cards.replaceChildren();for(const item of items){const card=node("article",undefined,"card"),link=node("a",undefined,"image-wrap"),image=node("img");link.href=item.href;image.src=item.src;image.alt=item.filename;image.loading="lazy";link.append(image);for(const box of item.overlays){const overlay=node("span",undefined,"box");overlay.style.cssText=`left:${box.left}%;top:${box.top}%;width:${box.width}%;height:${box.height}%;--box-color:${box.color}`;link.append(overlay)}card.append(link);const body=node("div",undefined,"body");body.append(node("strong",item.filename),node("div",`Original: ${item.relative_path}`,"muted"),node("div",`Destination: ${item.category} · Side: ${item.side}`));renderRanked(body,"Identities",item.identities);renderRanked(body,"Candidates",item.candidates);if(item.detections.length){const detail=node("div",undefined,"detail");detail.append(node("strong","Detections: "));item.detections.forEach((value,index)=>{if(index)detail.append(" · ");detail.append(`${value[0]} `);const score=node("span",value[1].toFixed(3));score.style.cssText=`color:${value[2]};font-weight:700`;detail.append(score)});body.append(detail)}if(item.failure)body.append(node("div",`Problem: ${item.failure}`,"detail"));card.append(body);cards.append(card)}loading.hidden=true}
function load(){const section=FINID_SECTIONS[sectionIndex],pageCount=section.pages.length;previous.disabled=pageIndex===0;next.disabled=pageIndex>=pageCount-1;pageLabel.textContent=pageCount?`Page ${pageIndex+1} of ${pageCount} · ${section.count} images`:`No images`;if(!pageCount){cards.replaceChildren();loading.textContent="No images in this category.";loading.hidden=false;return}render(JSON.parse(document.getElementById(section.pages[pageIndex]).textContent))}
FINID_SECTIONS.forEach((section,index)=>{const option=node("option",`${section.title} (${section.count})`);option.value=index;sectionSelect.append(option)});sectionSelect.onchange=()=>{sectionIndex=Number(sectionSelect.value);pageIndex=0;load()};previous.onclick=()=>{if(pageIndex){pageIndex--;load()}};next.onclick=()=>{if(pageIndex+1<FINID_SECTIONS[sectionIndex].pages.length){pageIndex++;load()}};load();
</script>'''
    )


def write_reports(
    store: ResultStore,
    metadata: ReportMetadata,
    progress: Callable[[int, int, str], None] | None = None,
) -> int:
    """Write one report per encounter with optional image-level progress.

    Parameters:
        store: Disk-backed run results.
        metadata: Run metadata shown in every report.
        progress: Optional callback receiving completed cards, total cards, and
            a user-facing label.

    Returns:
        Number of encounter reports written.
    """
    written = 0
    report_total = store.processed_images()
    reported = 0

    def advance() -> None:
        """Publish throttled report progress."""
        nonlocal reported
        reported += 1
        if progress is not None and (reported == report_total or reported % 25 == 0):
            progress(
                reported,
                report_total,
                f"Writing reports… {reported:,}/{report_total:,}",
            )

    if progress is not None:
        progress(0, report_total, "Preparing encounter reports…")
    for encounter_row in store.encounters():
        encounter = Path(str(encounter_row["path"]))
        report_root = Path(str(encounter_row["output_path"] or encounter))
        report_root.mkdir(parents=True, exist_ok=True)
        counts = store.encounter_counts(encounter)
        identity_count = store.encounter_identity_count(encounter)
        report_path = report_root / report_filename(encounter)
        assets_name = _asset_directory_name(encounter)
        assets_path = report_root / assets_name
        virtualized = counts["Total"] > VIRTUALIZE_THRESHOLD
        descriptor: int | None = None
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{report_path.name}.",
                suffix=".tmp",
                dir=report_root,
            )
            temporary_path = Path(temporary_name)
            handle = os.fdopen(descriptor, "w", encoding="utf-8")
            descriptor = None
            with handle:
                _header(handle, encounter, metadata, counts, identity_count)
                if virtualized:
                    _write_virtual_body(
                        handle,
                        store,
                        encounter,
                        report_root,
                        counts,
                        advance,
                    )
                else:
                    handle.write(f'<h2>Identified images <span class="muted">({counts["IDed"]})</span></h2>')
                    if not counts["IDed"]:
                        handle.write('<div class="notice">No confident identifications.</div>')
                    for group in store.primary_identity_groups(encounter):
                        identity = str(group["identity"])
                        handle.write(f'<h3>{html.escape(identity)} <span class="muted">({int(group["image_count"])})</span></h3><section class="cards">')
                        for row in store.identified_images(encounter, identity):
                            _card(handle, store, row, report_root, advance)
                        handle.write("</section>")
                    for category in ("FinSaddle", "Eyes", "Rest"):
                        _section(
                            handle,
                            category,
                            store.encounter_images(encounter, category),
                            counts[category],
                            store,
                            report_root,
                            True,
                            advance,
                        )
                handle.write(f'<p class="muted">Elapsed {metadata.elapsed_seconds:.1f}s · {metadata.throughput:.2f} images/s. Original JPEGs were not moved or modified.</p></main></body></html>')
            os.replace(temporary_path, report_path)
            temporary_path = None
            if (assets_path / ASSET_MARKER).is_file():
                shutil.rmtree(assets_path)
            written += 1
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    if progress is not None:
        progress(report_total, report_total, f"Wrote {written:,} encounter reports.")
    return written
