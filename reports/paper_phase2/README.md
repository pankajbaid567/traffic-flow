# Phase 2 IEEE Conference Style Paper

`phase2_ieee_paper.pdf` is an authentic two-column US Letter manuscript generated deterministically with embedded Times New Roman fonts, vector math expressions, formatted tables, and 350 DPI figures.

`phase2_ieee_paper.tex` contains the matching editable LaTeX source conforming to the official IEEEtran conference format.

### Authors:
Pankaj Baid, Ashutosh Singh, Shreya Suman, Narendra Singh, Ranvendra Pratap Singh

### Core Artifacts:
- `phase2_ieee_paper.pdf`: Compiled camera-ready PDF.
- `phase2_ieee_paper.tex`: Editable IEEEtran LaTeX source.
- `paper_content.json`: Machine-readable structured narrative blocks.
- `qa_report.json`: QA claims audit.
- `figures/`: Publication-grade figures.
- `phase2_ieee_source.zip`: Overleaf bundle (`.tex`, `authors.json`, this README, included figures); rebuild command in `reports/phase2/reproduction.md`.
- `phase2_slides.pptx`: Presentation, built by `scripts/build_phase2_slides.py`.

Every Phase 2 number is read from `data/metadata/phase2/` (written by `scripts/phase2_eval_all.py` and `scripts/phase2_run_signal_sim.py`); nothing is typed in by hand.
