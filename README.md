# ICLR 2026 Fingerprint Synthesis Report Draft

This directory contains an ICLR 2026-style report draft for the current
fingerprint synthesis project.

Main files:

- `main.tex`: paper draft.
- `main.bib`: references.
- `iclr2026_conference.sty`, `iclr2026_conference.bst`,
  `math_commands.tex`: copied from the ICLR 2026 template archive.
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
