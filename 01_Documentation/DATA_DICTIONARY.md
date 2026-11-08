# Data and evaluation dictionary

## Inputs and splits

`Final_Xiaoke_Passage_Corpus.jsonl`: one passage per line. `record_id` is the stable key (for example `XK_001231`); `text` is the processed passage; `source`, `source_file` and `chapter` record provenance. `source_start/source_end` refer to a source segment of the source book, not necessarily a same-length slice of the reconstructed `text`. Do not use these source bounds as entity offsets.

Development/test JSONL files add frozen role/order and sampling features. Their IDs must be disjoint. Historical split audit files may describe an earlier Schema; the only authoritative Schema for extraction and evaluation is `xiaoke_schema_v5.json` (version `xiaoke_schema_v5_20260816`). Historical metadata are retained, not silently rewritten into a claim of a different sampling chronology.

Development expert TXT contains original/preannotation/corrected sections. Use the original corrected-entity and corrected-relation sections as development reference labels. Test Gold uses the original blinded entity and relation sections. Machine preannotation sections are predictions, not Gold. The final test Gold alone does not establish the details of the independent-expert adjudication workflow.

## Structured ECKE predictions

Entity fields include `surface_form`, `start`, `end`, `normalized_concept`, `entity_type`, `evidence_text`, `entity_id`, `recognition_source`, `alignment_status` and a relevance explanation. Entity offsets use zero-based, half-open Python character indices into the current passage text. UTF-8 bytes are not character offsets. An `ocr_interrupted` field requires separate interpretation and must not be presented as exact substring matching.

Relation fields include head/tail entity IDs and surface/type, `relation_type`, evidence text/start/end and decision reason. Entity IDs locate endpoints in the same record. Scientific Schema: Chinese medicinal–Treats–Disease; Formula–Treats–Disease; Therapeutic method–Indication–Disease; Pathogenesis–Causes–Disease; Pathogenesis–Involves–Disease location; Symptom–Suggests–Pathogenesis; Disease–Manifests as–Symptom.

## Exact reported evaluation

Within each passage, collapse duplicates and match entity `(surface_form, entity_type)` pairs and directed relation `(head_surface, relation_type, tail_surface)` triples. Do not normalize, trim or translate JSONL prediction values again. TXT parsers strip formatting around `|` fields as in the original evaluator. List order is irrelevant; triple direction is not. Across-passage occurrences remain separate. These metrics do not measure exact span, normalization, or evidence attribution accuracy.

Micro-P/R/F1 aggregate TP/FP/FN across the 300 passages. CI uses the original Python `random.Random(20260822)` passage bootstrap, 5000 resamples, with sorted percentile indices `int(.025*(N-1))` and `int(.975*(N-1))`. The random generator is reset separately for the two tasks, as in the original evaluator.

Baseline JSONL generally stores entity pairs and relation triples directly. CARE evaluation uses `folded_entities`, not the raw span list. Its full raw output is retained. ECKE, ablation and LLMRE structured dictionaries are converted without changing their surface strings. Original terminal failures remain empty predictions and are not removed from the test denominator.

Full-corpus entity counts use `(surface,type)` and relation counts in the manuscript use `(head_surface,label,tail_surface)`. Figure 3 frequency selection also includes endpoint types; its typed distinct relation count therefore differs from the manuscript's 3220 surface-only count. This is a different key, not a corrected result.
