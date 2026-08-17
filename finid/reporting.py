"""Stream one HTML report per encounter from SQLite run state."""

from __future__ import annotations

import html
import hashlib
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
THUMBNAIL_SIZE = (480, 480)
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
    message: str = ""


def _header(handle: object, encounter: Path, metadata: ReportMetadata, counts: dict[str, int]) -> None:
    """Write report header and encounter totals."""
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
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:20px 0}}.metric,.card,.notice{{background:#fff;border:1px solid var(--line);border-radius:12px}}.metric{{padding:14px}}.metric strong{{display:block;font-size:1.5rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px}}.compact{{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}}.card{{overflow:hidden}}.image-wrap{{display:block;position:relative;background:#e8eef6}}.card img{{display:block;width:100%;height:auto}}.body{{padding:12px}}.box{{position:absolute;border:3px solid var(--box-color);box-shadow:0 2px 7px #0008;pointer-events:none}}.detail{{border-top:1px solid var(--line);padding-top:7px;margin-top:7px}}.notice{{padding:14px}}code{{overflow-wrap:anywhere}}a{{color:var(--blue)}}
.pager{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:18px 0}}.pager select,.pager button{{font:inherit;padding:8px 10px}}.pager button:disabled{{opacity:.45}}
</style></head><body><main><div style="color:var(--blue);font-weight:700">Fin Identification Report</div>
<h1>{html.escape(encounter.name)}</h1><p><span class="status {status_class}">{status}</span> <span class="muted">Generated {html.escape(metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S %Z'))}</span></p>
<p class="muted">Detector: {html.escape(metadata.detector_name)} · Identifier: {html.escape(metadata.identifier_name)} · Identification ≥ {metadata.threshold:.3f} · Fin/FinSaddle ≥ {metadata.fin_confidence:.3f} · Eyes ≥ {metadata.eye_confidence:.3f}</p>""")
    if metadata.message:
        handle.write(f'<div class="notice">{html.escape(metadata.message)}</div>')
    handle.write('<section class="summary">')
    for label in ("Total", "IDed", "FinSaddle", "Eyes", "Rest", "Failures"):
        handle.write(f'<div class="metric"><strong>{counts[label]}</strong>{label}</div>')
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


def _card(
    handle: object,
    store: ResultStore,
    row: object,
    report_root: Path,
    advance: Callable[[], None] | None = None,
) -> None:
    """Write one image card with assignment, scores, and overlays."""
    identities = list(store.identities(int(row["id"])))
    detections = list(store.detections(int(row["id"])))
    href = _image_href(row, report_root)
    width = height = 0
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
        handle.write('<div class="detail"><strong>Identities:</strong> ' + " · ".join(
            f'{html.escape(str(item["identity"]))} {float(item["score"]):.3f} {html.escape(str(item["score_type"]))}' for item in identities
        ) + "</div>")
    if detections:
        handle.write('<div class="detail"><strong>Detections:</strong> ' + " · ".join(
            f'{html.escape(str(item["class_name"]))} {float(item["confidence"]):.3f}' for item in detections
        ) + "</div>")
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


def _thumbnail(
    source: Path,
    destination: Path,
) -> tuple[int, int] | None:
    """Generate one bounded-size JPEG thumbnail and return source dimensions."""
    try:
        with Image.open(source) as image:
            width, height = image.size
            image.draft("RGB", THUMBNAIL_SIZE)
            thumbnail = image.convert("RGB")
            try:
                thumbnail.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                thumbnail.save(destination, "JPEG", quality=78, optimize=True)
            finally:
                thumbnail.close()
        return width, height
    except OSError:
        return None


def _virtual_card_data(
    store: ResultStore,
    row: object,
    report_root: Path,
    assets_name: str,
    thumbnail_root: Path,
) -> dict[str, object]:
    """Build serializable card data and its sequentially generated thumbnail."""
    identities = list(store.identities(int(row["id"])))
    detections = list(store.detections(int(row["id"])))
    href = _image_href(row, report_root)
    digest = hashlib.sha1(str(row["relative_path"]).encode("utf-8")).hexdigest()[:16]
    thumbnail_name = f"{int(row['id'])}-{digest}.jpg"
    dimensions = _thumbnail(Path(str(row["path"])), thumbnail_root / thumbnail_name)
    source = (
        quote(f"{assets_name}/thumbs/{thumbnail_name}", safe="/")
        if dimensions is not None
        else href
    )
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
        "src": source,
        "filename": str(row["copied_filename"] or row["filename"]),
        "relative_path": str(row["relative_path"]),
        "category": str(row["cluster_category"]),
        "side": str(row["cluster_side"] or "—"),
        "identities": [
            [str(item["identity"]), float(item["score"]), str(item["score_type"])]
            for item in identities
        ],
        "detections": [
            [str(item["class_name"]), float(item["confidence"])]
            for item in detections
        ],
        "overlays": overlays,
        "failure": str(row["failure_message"] or ""),
    }


def _write_chunk(
    path: Path,
    cards: list[dict[str, object]],
) -> None:
    """Write one JavaScript data chunk consumed by the local report."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("window.finidReceive(")
        json.dump(cards, handle, ensure_ascii=True, separators=(",", ":"))
        handle.write(");\n")


def _write_virtual_section(
    store: ResultStore,
    rows: Iterable[object],
    section_index: int,
    assets_name: str,
    assets_root: Path,
    report_root: Path,
    advance: Callable[[], None],
) -> list[str]:
    """Stream one gallery section into bounded JavaScript chunks."""
    chunk_names: list[str] = []
    cards: list[dict[str, object]] = []
    thumbnail_root = assets_root / "thumbs"
    for row in rows:
        cards.append(
            _virtual_card_data(
                store,
                row,
                report_root,
                assets_name,
                thumbnail_root,
            )
        )
        advance()
        if len(cards) == REPORT_PAGE_SIZE:
            name = f"section-{section_index:04d}-page-{len(chunk_names):05d}.js"
            _write_chunk(assets_root / name, cards)
            chunk_names.append(quote(f"{assets_name}/{name}", safe="/"))
            cards = []
    if cards:
        name = f"section-{section_index:04d}-page-{len(chunk_names):05d}.js"
        _write_chunk(assets_root / name, cards)
        chunk_names.append(quote(f"{assets_name}/{name}", safe="/"))
    return chunk_names


def _replace_assets(staged: Path, destination: Path) -> None:
    """Atomically replace only an app-managed report asset directory."""
    if destination.exists() and not (destination / ASSET_MARKER).is_file():
        raise PermissionError(f"Refusing to replace unmanaged report assets: {destination}")
    backup_root: Path | None = None
    backup: Path | None = None
    try:
        if destination.exists():
            backup_root = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent)
            )
            backup = backup_root / "previous"
            os.replace(destination, backup)
        os.replace(staged, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        if backup_root is not None:
            shutil.rmtree(backup_root, ignore_errors=True)
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root, ignore_errors=True)


def _write_virtual_body(
    handle: object,
    store: ResultStore,
    encounter: Path,
    report_root: Path,
    assets_root: Path,
    assets_name: str,
    counts: dict[str, int],
    advance: Callable[[], None],
) -> None:
    """Write a paged gallery shell and disk-backed JavaScript card chunks."""
    sections: list[dict[str, object]] = []
    section_index = 0
    for group in store.primary_identity_groups(encounter):
        identity = str(group["identity"])
        chunks = _write_virtual_section(
            store,
            store.identified_images(encounter, identity),
            section_index,
            assets_name,
            assets_root,
            report_root,
            advance,
        )
        sections.append(
            {
                "title": f"Identified — {identity}",
                "count": int(group["image_count"]),
                "chunks": chunks,
            }
        )
        section_index += 1
    for category in ("FinSaddle", "Eyes", "Rest"):
        chunks = _write_virtual_section(
            store,
            store.encounter_images(encounter, category),
            section_index,
            assets_name,
            assets_root,
            report_root,
            advance,
        )
        sections.append(
            {"title": category, "count": counts[category], "chunks": chunks}
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
const sectionSelect=document.getElementById("finid-section"),cards=document.getElementById("finid-cards"),pageLabel=document.getElementById("finid-page"),loading=document.getElementById("finid-loading"),previous=document.getElementById("finid-prev"),next=document.getElementById("finid-next");let sectionIndex=0,pageIndex=0,requestToken=0;
function node(tag,text,cls){const value=document.createElement(tag);if(text!==undefined)value.textContent=text;if(cls)value.className=cls;return value}
function render(items){cards.replaceChildren();for(const item of items){const card=node("article",undefined,"card"),link=node("a",undefined,"image-wrap"),image=node("img");link.href=item.href;image.src=item.src;image.alt=item.filename;image.loading="lazy";link.append(image);for(const box of item.overlays){const overlay=node("span",undefined,"box");overlay.style.cssText=`left:${box.left}%;top:${box.top}%;width:${box.width}%;height:${box.height}%;--box-color:${box.color}`;link.append(overlay)}card.append(link);const body=node("div",undefined,"body");body.append(node("strong",item.filename),node("div",`Original: ${item.relative_path}`,"muted"),node("div",`Destination: ${item.category} · Side: ${item.side}`));if(item.identities.length)body.append(node("div",`Identities: ${item.identities.map(value=>`${value[0]} ${value[1].toFixed(3)} ${value[2]}`).join(" · ")}`,"detail"));if(item.detections.length)body.append(node("div",`Detections: ${item.detections.map(value=>`${value[0]} ${value[1].toFixed(3)}`).join(" · ")}`,"detail"));if(item.failure)body.append(node("div",`Problem: ${item.failure}`,"detail"));card.append(body);cards.append(card)}loading.hidden=true}
window.finidReceive=items=>{if(Number(document.currentScript?.dataset.token)!==requestToken)return;render(items)};function load(){const section=FINID_SECTIONS[sectionIndex],pages=section.chunks.length;previous.disabled=pageIndex===0;next.disabled=pageIndex>=pages-1;pageLabel.textContent=pages?`Page ${pageIndex+1} of ${pages} · ${section.count} images`:`No images`;cards.replaceChildren();if(!pages){loading.textContent="No images in this category.";loading.hidden=false;return}loading.textContent="Loading gallery page…";loading.hidden=false;const token=++requestToken,script=document.createElement("script");script.src=section.chunks[pageIndex];script.dataset.token=token;script.onload=()=>script.remove();script.onerror=()=>{loading.textContent="Could not load this gallery page."};document.body.append(script)}
FINID_SECTIONS.forEach((section,index)=>{const option=node("option",`${section.title} (${section.count})`);option.value=index;sectionSelect.append(option)});sectionSelect.onchange=()=>{sectionIndex=Number(sectionSelect.value);pageIndex=0;load()};previous.onclick=()=>{if(pageIndex){pageIndex--;load()}};next.onclick=()=>{if(pageIndex+1<FINID_SECTIONS[sectionIndex].chunks.length){pageIndex++;load()}};load();
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
        """Publish throttled report and thumbnail progress."""
        nonlocal reported
        reported += 1
        if progress is not None and (reported == report_total or reported % 25 == 0):
            progress(
                reported,
                report_total,
                f"Writing reports and thumbnails… {reported:,}/{report_total:,}",
            )

    if progress is not None:
        progress(0, report_total, "Preparing encounter reports…")
    for encounter_row in store.encounters():
        encounter = Path(str(encounter_row["path"]))
        report_root = Path(str(encounter_row["output_path"] or encounter))
        report_root.mkdir(parents=True, exist_ok=True)
        counts = {
            category: store.encounter_category_count(encounter, category)
            for category in ("IDed", "FinSaddle", "Eyes", "Rest")
        }
        counts["Total"] = sum(counts.values())
        counts["Failures"] = store.encounter_failure_count(encounter)
        report_path = report_root / report_filename(encounter)
        assets_name = _asset_directory_name(encounter)
        assets_path = report_root / assets_name
        virtualized = counts["Total"] > VIRTUALIZE_THRESHOLD
        staged_assets: Path | None = None
        if virtualized:
            staged_assets = Path(
                tempfile.mkdtemp(prefix=f".{assets_name}.", dir=report_root)
            )
            (staged_assets / "thumbs").mkdir()
            (staged_assets / ASSET_MARKER).write_text(
                "FinIdentification report assets\n",
                encoding="utf-8",
            )
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{report_path.name}.", suffix=".tmp", dir=report_root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                _header(handle, encounter, metadata, counts)
                if virtualized and staged_assets is not None:
                    _write_virtual_body(
                        handle,
                        store,
                        encounter,
                        report_root,
                        staged_assets,
                        assets_name,
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
            if staged_assets is not None:
                _replace_assets(staged_assets, assets_path)
                staged_assets = None
            os.replace(temporary_name, report_path)
            if not virtualized and (assets_path / ASSET_MARKER).is_file():
                shutil.rmtree(assets_path)
            written += 1
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            if staged_assets is not None:
                shutil.rmtree(staged_assets, ignore_errors=True)
            raise
    if progress is not None:
        progress(report_total, report_total, f"Wrote {written:,} encounter reports.")
    return written
