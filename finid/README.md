# FinID Processing Pipeline

This package implements encounter discovery, fin detection and identification,
optional image clustering, and encounter-level HTML reports. It is designed for
large source trees, including datasets with hundreds of thousands of JPEGs,
without retaining all image paths or results in RAM.

## Package layout

- `pipeline.py` coordinates discovery, inference, classification, copying, and
  final publication.
- `storage.py` provides temporary SQLite-backed run state.
- `models.py` discovers identification models and performs bounded identifier
  inference.
- `reporting.py` writes encounter reports, large-gallery data chunks, and
  thumbnails.

The desktop interface in `../finid_app.py` builds a `PipelineConfig`, starts the
pipeline on a worker thread, and displays log and progress events.

## Complete run lifecycle

### 1. Configuration and safety checks

The pipeline validates:

- the source directory;
- identification, Fin/FinSaddle, and eye confidence values;
- detector and identifier batch sizes;
- the optional clustering output directory;
- that input and output directories do not overlap; and
- that an existing non-empty output contains `.finid-managed`.

A new clustering output may be absent or empty. An existing non-empty output is
replaced only when it carries the management marker. Source JPEGs are never
moved, renamed, overwritten, or modified.

### 2. Single-pass source inventory

The source filesystem is traversed once with streaming `scandir` iterators.
Traversal memory is proportional to directory depth, not the number of images.
Symlinked files and directories are ignored.

During this scan, provisional directory relationships and JPEG paths are
written to temporary SQLite tables. The application does not build an in-memory
list containing all source images.

After the scan, encounter ownership is resolved from the SQLite inventory:

1. Directories beginning with `YYYY-MM-DD` establish dated encounter roots.
2. If a dated directory contains only `ENCOUNTER*` child directories, those
   children become separate encounters.
3. When two or more sibling directories begin with `GROUP`, each group becomes
   a separate encounter, including groups nested below a trip directory.
4. A lone `GROUP1` remains inside its existing encounter.
5. Nested camera, trip, side, and other directories remain recursively owned by
   the closest encounter root.
6. JPEGs without a dated encounter ancestor are recorded as skipped and never
   sent to inference.

For example:

```text
source/
  2026-08-17 Survey/
    Trip1/
      GROUP1/
        Camera A/a.jpg
      GROUP2/
        Camera B/b.jpg
```

This produces two encounters with mirrored paths:

```text
2026-08-17 Survey/Trip1/GROUP1
2026-08-17 Survey/Trip1/GROUP2
```

The provisional scan tables are discarded after permanent encounter ownership
has been written to SQLite.

### 3. Runtime and model loading

The detector is configured with a bounded batch, image size, confidence floor,
maximum detection count, and bounded source window. MPS and FP16 are used when
available and selected.

Detector class metadata comes from the model. Classification recognizes these
raw class-name families, case-insensitively:

- `fin_left` and `fin_right`;
- `finSaddle_left/right`, `fin_saddle_left/right`, and
  `saddle_left/right`; and
- `eye_left` and `eye_right`.

Other detector classes are retained in report data but do not qualify for an
identification or fallback category.

The identification model is loaded only when at least one selected detector
class is recognized as FinSaddle. Selecting only plain fins, eyes, or unrelated
classes therefore avoids allocating identifier model memory.

### 4. Detection

The detector receives source paths from an ordered SQLite cursor. Its input
window and prediction batch remain bounded regardless of dataset size.

For every returned source image, the pipeline records each retained detection:

- numeric class ID;
- raw class name;
- normalized side;
- detector confidence;
- bounding coordinates; and
- whether it is an identification candidate.

The number of retained detections per image is bounded by `max_detections`.

### 5. FinSaddle crop preparation and identification

Only selected, above-threshold FinSaddle detections become identification
crops. Plain `fin_left/right` detections are deliberately not identified.

Crops are queued across source images until the configured identifier batch is
full. This allows images with one FinSaddle crop each to form an efficient GPU
batch while maintaining a strict memory bound. At end-of-stream or safe stop,
the final short batch is processed.

Each crop is closed immediately after prediction. The source image used for
cropping is also closed immediately after its crops are created. Identifier
scores meeting the identification threshold are stored in SQLite.

If several crops return the same identity for one source image, only its
highest score is retained. All distinct accepted identities remain associated
with the single source image.

### 6. Single image classification

Every successfully processed image receives exactly one assignment in this
priority order:

1. **IDed** when one or more selected FinSaddle crops have an accepted identity.
2. **FinSaddle** when a FinSaddle detection passes the Fin/FinSaddle threshold
   but no identity was accepted.
3. **Eyes** when an eye detection passes the eye threshold.
4. **Rest** otherwise.

Plain fin detections never qualify for identification or the FinSaddle fallback
category. An image containing only plain fins therefore goes to `Rest`.

When qualifying detections for the winning category occur on both sides,
`LEFT` wins. Processing or decoding failures are recorded as `Rest`, together
with their error messages.

At this stage, assignments exist only in temporary SQLite. No source images
have been copied and no visible encounter output folders are expected yet.

### 7. Why clustered folders appear near the end

Cluster output is deliberately delayed until detector and identifier inference
has ended. This provides three important properties:

- an existing managed output is not damaged by an incomplete new run;
- image copying does not compete with inference for source-disk bandwidth; and
- a copying, report, thumbnail, or disk-space failure can be rolled back.

When finalization begins, the application creates a hidden sibling staging
directory similar to:

```text
.selected-output.finid-staging-AbCdEf/
```

Encounter and category folders are built inside this staging tree. Consequently,
the selected output may remain empty or unchanged while the activity display
says that images are being identified.

### 8. Cluster copying

For every completed SQLite image row, the pipeline creates the required
encounter structure:

```text
<output>/<relative encounter path>/
  LEFT/
    IDed/
    FinSaddle/
    Eyes/
  RIGHT/
    IDed/
    FinSaddle/
    Eyes/
  Rest/
```

Each source path is copied exactly once. Nested source directories are flattened
inside the selected category, while the source hierarchy is mirrored only down
to the encounter root.

Accepted identities prefix an identified filename in descending score order:

```text
NKW-001_NKW-017_original.jpg
```

If flattened paths produce the same destination filename, the later collision
receives a deterministic eight-character hash derived from its encounter-relative
source path. The copied filename is stored back in SQLite for reporting.

### 9. Encounter reports

Exactly one HTML report is written directly inside each encounter output root.
When clustering is disabled, it is written inside the source encounter root.

Each report contains:

- encounter totals and run metadata;
- identified images grouped by highest-scoring identity;
- all accepted identities for multi-identity images;
- FinSaddle, Eyes, and Rest galleries;
- copied filename and original encounter-relative path;
- destination category and side;
- identity and detector scores;
- detection overlays; and
- processing errors.

Reports are never created in date/year grouping directories, category folders,
camera folders, or other nested source directories.

Small encounters are written as ordinary static galleries. Encounters with more
than 500 completed images use a virtualized gallery:

- card data is divided into managed JavaScript chunks of at most 100 images;
- the browser loads only the current page into the DOM;
- 480-pixel JPEG thumbnails are generated sequentially, one image at a time;
- each thumbnail links to the corresponding full image; and
- only one `.html` report is created.

Virtual report assets are stored in a managed `FinID_<encounter>_assets`
directory. Its `.finid-report-assets` marker allows safe replacement. These
thumbnail JPEGs are explicitly excluded from future source inventories.

### 10. Atomic output publication

After every copy, thumbnail, data chunk, and report has succeeded:

1. An existing managed output is moved to a temporary backup location.
2. The completed staging directory is renamed to the selected output path.
3. The previous backup is removed.

Directory renames on the same filesystem are atomic. If publication fails
before replacement, the previous managed output is restored. Failed staging
trees are removed.

This is why completed encounter folders become visible together near the end
instead of appearing incrementally during inference.

## Temporary SQLite state

SQLite stores all dataset-sized state:

- encounter roots and mirrored paths;
- source path and encounter ownership;
- raw detections, side, confidence, and boxes;
- accepted identities and primary identity;
- classification and copied filename;
- failures; and
- skipped undated JPEGs.

Compound indexes support ordered inference, category galleries, and
primary-identity grouping. The highest-scoring identity is materialized on the
source-image row so reports do not repeatedly calculate it with correlated
queries.

Writes are committed in bounded groups rather than synchronizing SQLite for
every image. The database and its WAL files are deleted when the run has fully
finished.

## Whole-run progress

The progress indicator covers the complete pipeline rather than reaching 100%
when inference ends. Filesystem discovery is indeterminate because the total
number of paths is unknowable until traversal finishes. It then switches to a
monotonic determinate scale covering:

1. inventory resolution and runtime loading;
2. detection, FinSaddle crop preparation, and identification;
3. copying completed images into staged encounter clusters, when enabled;
4. report cards, paged gallery data, and thumbnails; and
5. atomic publication of the completed output.

Copy and report progress is updated every 25 images and at each phase boundary.
This keeps the interface responsive without creating hundreds of thousands of
queued GUI events. The detailed label shows the active phase and image count.
Any displayed ETA applies only to the named active phase, not to the complete
run. Inference timing starts after model loading so discovery and setup do not
distort its estimate.
The bar reaches 100% only after final output has been published, or after safe
partial/error finalization has ended; processed-image totals remain visible in
the run summary.

## Memory behavior

RAM use does not grow with the number of JPEGs. It is primarily bounded by:

- detector and identifier model weights;
- the detector batch and source window;
- the identifier crop batch;
- one source image being cropped;
- at most `max_detections` crops from one detector result;
- SQLite's bounded page cache; and
- directory depth and encounter metadata.

The GUI coalesces progress messages rather than queuing one event per image.
Only representative per-image failures are sent to the activity log; all
failures remain available from SQLite-backed reports.

Large reports are streamed to disk. Only one report page worth of card data and
one thumbnail source image are held while virtualized report assets are built.

## Stopping and failures

Stopping is cooperative. The detector finishes or interrupts its current
bounded operation, and any already queued FinSaddle crops are identified before
their images are finalized. Images that never reached a completed state remain
excluded from partial results.

A stopped run may still publish a partial staged output and partial encounter
reports. The reports clearly identify the run as partial.

If detector inference omits a source path without a deliberate stop, that path
is marked as a failed `Rest` image. Model-level or output-finalization errors are
reported in the run summary. Existing managed cluster output remains protected
by staging and rollback.

## Performance tuning

The most important controls are:

- **Detector batch:** usually the primary throughput control.
- **Identifier batch:** useful when many images contain FinSaddle crops.
- **Detector image size:** reducing it can substantially improve speed but may
  reduce recall for small or distant fins.
- **FP16:** normally beneficial on Apple Silicon MPS.
- **Source storage:** a local SSD reduces JPEG-read and decode stalls.

Increase one batch setting at a time and benchmark the same representative
subset. Stop increasing when throughput no longer improves, swap becomes
active, memory pressure rises, or an out-of-memory recovery is logged.

The displayed inference rate excludes the later copy, thumbnail, report, and
atomic-publication stages.

## Output markers and cleanup

Two markers distinguish app-owned output from user data:

- `.finid-managed` marks a clustering output root that may be rebuilt.
- `.finid-report-assets` marks a virtual report asset directory that may be
  replaced or removed.

The application never recursively replaces an unmarked non-empty clustering
output or an unmarked report asset directory.
