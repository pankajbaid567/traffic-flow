# Phase 1 IEEE style paper

`phase1_ieee_paper.pdf` is a two-column US Letter paper with embedded Times New Roman fonts, numbered references and generated tables/figures. `phase1_ieee_paper.tex` contains the matching editable content using the IEEEtran conference class. It can be compiled in a TeX environment with IEEEtran installed; the supplied PDF is generated deterministically by ReportLab. This is an IEEE-style capstone manuscript, not an accepted/published IEEE article. No conference name, DOI or copyright attribution is invented.

Authors: Pankaj Baid, Ashutosh Singh, Shreya Suman, Narendra Singh, Ranvendra Pratap Singh. Affiliation has not been supplied and is left blank. Update `authors.json` and rerun `python scripts/build_phase1_paper.py` to revise the author block.

Tables and figures derive from verified Phase 1 forecast rows. Complete numerical evidence is in `data/metadata/`. `paper_content.json` is an editable structured copy of the narrative, tables and figure captions.
