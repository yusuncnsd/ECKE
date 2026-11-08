# Original lexicon workbook and JSON verification

Verification date: 2026-10-02.

The source workbook is Xiaoke_Concept_Lexicon_Deduplicated_20260814.xlsx; its structured conversion is xiaoke_lexicon_normalized_v1.json. Both are retained in the same directory.

The workbook has one sheet (sheet1), 680 rows, four columns, and no header row. JSON contains 680 concepts whose source-row references cover rows 1–680 completely. Workbook SHA-256: c7b55f85cd78ca01936cbddcccddcb6548007e8d266c68ba7042d8ca9ccc6ea8. This equals audit.source_sha256 in the JSON, identifying the original conversion input.

Row-by-row checks of concept names, alias sets, and category mapping found no differences. Aliases were split on the original enumeration delimiter; sorting in JSON does not alter alias content. Row 97 already includes the concept's own name among its aliases; the conversion did not introduce it.

Category normalization maps formula to the study's Formula category, etiology/pathogenesis to Pathogenesis, and therapeutic principles/methods to Therapeutic method. Chinese medicinal, Disease, Symptom, and Disease location retain their types. No duplicate rows were deleted and no repairs were applied.

JSON SHA-256: a9025150158e5360e0651f812182d788548ed4a06ad9afc083fc89a9572b64b4.

The workbook was relocated without changing its content; pre/post-move checksums agree. The JSON, experimental settings, and manuscript were not changed by that archival operation, and no experiment was run. Retain both files: the workbook is the source lexicon and JSON is the structured program input. A historical relocation alone did not repair old runtime paths; verify paths before a new screening run. Public filenames have subsequently been translated without changing workbook cells or JSON concept data.
