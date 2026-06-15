# WACV 2027 Fingerprint Synthesis Report Draft

This directory contains a WACV 2027-style report draft for the current
fingerprint synthesis project.

Main files:

- `main.tex`: paper draft.
- `main.bib`: references.
- `wacv.sty`, `ieeenat_fullname.bst`: copied from
  `/home/nguyenthanhlam/wacv-2027-author-kit-template`.
- `figures/`: QC sheets copied from the current generation outputs.

Build command, once a TeX distribution is available:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Current TODOs in the draft:

- Fill the final SD302A FPGAN-Control comparison table after deciding whether
  to use paper numbers or a reproduced same-recognizer protocol.
- Add final FID/sFID/IS and minutiae statistics once the final generator
  checkpoint and sample set are fixed.
- Add measured A4000 inference and Stage-2 UNet LoRA fine-tuning benchmarks.
