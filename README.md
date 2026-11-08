# ECKE: Evidence-Constrained Knowledge Extraction

Research inputs, code, frozen predictions, and evaluation records accompanying **ECKE: Evidence-constrained knowledge extraction with LLMs for ancient Chinese medical texts**.

## Reading order

| Location | Manuscript component |
|---|---|
| [Results-to-files map](01_Documentation/RESULTS_TO_FILES.md) | Start here to locate the evidence for each result |
| [Corpus preparation](02_Experiments/01_Corpus_Preparation) | Cleaning, topic screening, and the final 3065 passages |
| [ECKE](02_Experiments/02_ECKE) | Entity recognition, relation extraction, overall evaluation, and Oracle |
| [External comparisons](02_Experiments/03_External_Comparisons) | Eight comparison methods; Table 3 |
| [Ablations](02_Experiments/04_Ablation_Studies) | Four ablation settings; Table 4 |
| [Verification results](06_Verification_Results) | Read table3.csv, table4.csv, and metrics_summary.json first |
| [Figure data](04_Figure_Data) | Numerical and network source data; display images are not bundled |

Within each experiment, follow inputs → configuration → code → predictions → evaluation. Read summary reports before inspecting individual JSONL records. JSONL files are machine-readable research outputs, not narrative documentation.

`03_Baseline_Packages` contains model-specific runtime packages and original records. `05_Frozen_Data_Indices` identifies the fixed development/test split and screening candidates. `07_Provenance` records file origins and hashes. `01_Documentation` explains data formats, result mappings, and new inference runs.

## Offline verification

From the repository root, with Python 3.10 or later:

```text
python 09_Reproduction_Tools/verify_files.py
python 09_Reproduction_Tools/reproduce.py
```

These commands need neither credentials nor network access. Outputs go to `reproduced/` and do not replace frozen predictions. They reproduce evaluation of retained outputs, not a new API inference run. New model runs are described in [ONLINE_RERUN.md](01_Documentation/ONLINE_RERUN.md); future API or GPU outputs are not guaranteed to be identical.

The 488 original and cleaned books retain their Chinese filenames. Original passages, lexicon entries, annotation labels, predictions, and experimental prompts remain in Chinese because they are the actual research inputs and outputs. Reader-facing filenames and documentation are in English. [DATA_DICTIONARY.md](01_Documentation/DATA_DICTIONARY.md) defines evaluation fields and label meanings.

## Licensing and citation

- [MIT license for own code](08_Licensing_and_Version/LICENSE)
- [CC BY 4.0 license for own lexicon and annotations](08_Licensing_and_Version/LICENSE-DATA.md)
- [Third-party notices](08_Licensing_and_Version/THIRD_PARTY_NOTICES.md)
- [Citation information](08_Licensing_and_Version/CITATION.cff)
- [Package and model metadata](08_Licensing_and_Version/PACKAGE_METADATA.json)

Only this README is at the repository root. GitHub may not automatically recognize license and citation files placed in subdirectories. The `.gitignore` inside the licensing directory is not a repository-wide exclusion rule; do not add credentials, temporary caches, or new-run outputs to a release. A DOI must identify an actual archived version and is not fabricated here.
