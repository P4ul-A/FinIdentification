# Private fin-identification models

This folder is included in GitHub downloads, but all `*.pt` weights are
excluded by the repository `.gitignore`.

Copy separately supplied ResNet or ArcFace checkpoints here. An ArcFace
checkpoint named `<model>.pt` should use `<model>.gallery.pt`; the legacy
`arc_face.pt` plus `gallery.pt` pair is also supported.

The desktop app validates compatible checkpoint/gallery pairs at startup.
