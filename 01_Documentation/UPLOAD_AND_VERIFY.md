# Upload and verification guide

This directory is the 2026-11-07 local release prepared from the retained experiment materials. Upload its CONTENTS to the repository root, preserving the directories and UTF-8 filenames. README.md should appear at the repository root.

## Verify after downloading

From the repository root, using Python 3.10 or later:

```text
python 09_Reproduction_Tools/verify_files.py
python 09_Reproduction_Tools/reproduce.py
```

Both commands work offline. The first verifies the published file hashes; the second recomputes the submitted numerical results from the supplied final labels and predictions. Numerical results are written to reproduced/ and do not overwrite frozen experiments. No API credential is required.

## Data entry points

- data/frozen_split_index.csv lists all 100 development and 300 test record IDs.
- data/reference_labels/ provides machine-readable exports of the final expert labels. These are exports of the original annotation TXT, not new annotations. The original TXT remains authoritative.
- docs/RESULTS_TO_FILES.md maps individual manuscript results to their actual files.
- docs/DATA_DICTIONARY.md defines data fields and evaluation keys.
- docs/ONLINE_RERUN.md describes separate new model runs. Run preparation does not invoke inference.

The complete final 3065-passage corpus and 310-record manual review workbook are retained. The package uses the fixed final corpus for numerical reproduction; it does not reconstruct every historical manual editing step.

## Publish

Keep LICENSE, LICENSE-DATA.md, THIRD_PARTY_NOTICES.md, CITATION.cff and provenance/ in the upload. Use Git/GitHub Desktop for the directory rather than treating a ZIP as the only repository content. Do not add personal credentials or new run outputs. Archive the published version separately to obtain a real DOI, then record that DOI in repository metadata and the manuscript. This package has not itself been uploaded or assigned a DOI.
