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

1. Choose the root folder containing `.jpg` or `.jpeg` images.
2. Choose the detection model, then select one or more model-derived object
   classes whose crops should be passed to identification. All are selected
   whenever the model changes.
3. Choose the fin-identification model and set the minimum score required for
   a good identification.
4. Select **Start identification**.

Every scanned directory receives a report named
`FinID_<folder name>.html`. Reports contain accepted identifications and link
to parent and child reports. Accepted fin boxes are drawn as colored browser
overlays, with each identity name below the image shown in the matching color.
Images are displayed from their original locations; they are never copied or
changed. `.DS_Store` and existing
`FinID_*.html` files are ignored, while other non-JPEG files are listed as
skipped.

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

Paths and results are streamed, and run state is held in temporary SQLite
storage, so steady-state memory does not grow with the number of images.
The class list comes directly from each detection model through the shared
`FinDetection-MPS-Core` contract. Only selected, above-confidence detections
are cropped in memory and passed to the identification model. No class IDs are
hard-coded in finID. Core owns class metadata normalization, selection,
confidence partitioning, and overlap suppression so every consuming app uses
the same behavior.

## Tests

```bash
venv/bin/python -m unittest discover -s tests
```
