# Fin Identification

Fin Identification is a macOS desktop tool for finding confident
individual-orca matches in large folder trees of JPEG images.

Open `LaunchFinIdentification.command` to start. The launcher creates a Python
3.12 environment when necessary, installs the pinned shared
`FinDetection-MPS-Core` runtime, selects local inference hardware, and opens the graphical
application. When the interface is closed normally, its launcher Terminal tab
closes automatically. Setup failures remain visible so their messages can be
read and copied.

## Workflow

1. Choose a tree containing date-prefixed encounter folders such as
   `2026-08-17 Survey`, or choose one dated encounter directly.
2. Choose the detection model, then select one or more model-derived object
   classes. Selected FinSaddle classes are eligible for identification; plain
   fin classes are recorded but currently go to Rest. All classes are selected
   whenever the model changes.
3. Choose the fin-identification model and set the minimum score required for
   a good identification.
4. Choose one of three output modes: HTML reports beside the originals,
   managed cluster folders inside each encounter, or managed cluster folders
   in a separate output location. A previous cluster output can be rebuilt
   only when it contains the app's `.finid-managed` marker.
5. Set the identification, Fin/FinSaddle, and eye confidence thresholds, then
   select **Start identification**.

The app creates exactly one `FinID_<encounter>.html` report per encounter.
A dated folder containing only `ENCOUNTER*` child folders is split into those
explicit encounters; nested trip, side, camera, and lone group folders remain
part of their owning encounter. When an encounter contains sibling `GROUP*`
directories such as `GROUP1` and `GROUP2`, each group becomes its own encounter
and receives its own output tree and report. JPEGs without a dated encounter ancestor are
skipped before inference and counted in the run summary.

Reports begin with identified images grouped by their highest-scoring identity,
then show FinSaddle, Eyes, and Rest. Cards include source and copied paths,
assignment, side, identity and detector scores, and detection overlays. Report-only
mode writes reports in source encounter roots and leaves images in place.
In-place clustering creates a managed `FinID_<encounter>_clusters` directory
inside each encounter. Separate-output clustering mirrors the source hierarchy
through each encounter root. Both clustering modes copy each original once into
`LEFT`, `RIGHT`, or `Rest` and include the report with the cluster folders.
Original JPEGs are never moved or modified.

An IDed image containing an eye detection above the configured eye threshold
adds `EYE` between its identity prefix and original filename, for example
`NKW-1307_EYE_original.jpg`. This also marks eye-only images connected through
a successful camera burst; saddle-only IDed images keep the existing name.

When an identification attempt is below the acceptance threshold, FinSaddle
report cards show the three most likely identity candidates for each processed
saddle crop. Candidate labels use the same color as their corresponding
detection boxes.

Eye-only photos are connected to saddle results from the same camera burst. A
burst consists of photos in one immediate source folder whose consecutive EXIF
capture times are no more than two seconds apart. Eyes inherit all accepted
orca identities from identified saddle photos in that burst, or fall back to
the saddle folder when no saddle was identified. Missing EXIF capture times do
not use filename or filesystem-time fallbacks. When RIGHT identification is
disabled, RIGHT eye photos remain in `RIGHT/Eyes` and are not burst-linked.

Encounters with more than 500 images use a paged gallery. The report loads at
most 100 cards at once from page data embedded in the single HTML file. Images
are loaded lazily, and each card links to the full image. No companion report
asset directory is created.

## Models

Model weights are private and intentionally excluded from GitHub. After
downloading the application, copy the separately supplied weights into these
folders (the launcher creates the folders if they are absent):

- Fin-recognition weights belong in `model_recognition/` and use
  `model_<name>.pt`.
- Supported ResNet and ArcFace identification checkpoints belong in
  `model_identification/`. ArcFace galleries use `<model>.gallery.pt`.

Models are inspected at launch. Incompatible files are explained in the
activity log rather than appearing in the selector.

## Performance

On Apple Silicon, both models use Apple MPS. On an Intel-based Mac they
automatically use CPU inference with PyTorch 2.2.2, the final release that
publishes Python 3.12 Intel macOS wheels. Defaults are selected from available
memory, beginning with detector/identifier batches of 2/8 on a 16 GB Mac.
Higher-memory Macs receive larger batches. Both stages automatically halve
their active batch after an MPS out-of-memory error.

Paths, detections, assignments, report grouping, and results are streamed
through temporary SQLite storage, so steady-state memory does not grow with the
number of images. The filesystem is scanned once; provisional paths and
directory relationships are resolved into encounters on disk. FinSaddle crops
are queued across source images into bounded identifier batches, and primary
identities are indexed directly for report generation. Cluster trees are
assembled in staging and swapped into place only after copying and report
generation succeed.
The class list comes directly from each detection model through the shared
`FinDetection-MPS-Core` contract. Only selected, above-confidence FinSaddle
detections are cropped in memory and passed to the identification
model. No numeric class IDs are hard-coded in finID; encounter classification
uses the model's class names and raw detector scores. Identification accepts
FinSaddle crops only. Plain fin detections currently go to Rest and never to
identification or the FinSaddle fallback category.

## Tests

```bash
venv/bin/python -m unittest discover -s tests
```
