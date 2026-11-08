# LLMRE source method and task adaptation

Source: Lei Zhao, Ling Kang, Quan Guo. *Zero-shot document-level biomedical relation extraction via scenario-based prompt design in two-stage with LLM*. Computational Biology and Chemistry, 2026, 123:108978. DOI: https://doi.org/10.1016/j.compbiolchem.2026.108978. Author code: https://github.com/tyrone1979/LLMRE. Mapping follows Sections 2.1–2.2 and src/task_ner.py, src/relation.py, src/task_re.py, and meta/prompt_templates.json.

| Original mechanism | Implementation here | Adaptation |
|---|---|---|
| Zero-shot two-stage NER/normalization before RE | Recognize textual entities, assess synonyms, form shared concept clusters, identify hierarchy, then generate and judge Schema-compatible relations | Fixed 300 ancient-text passages; no parameter training or test-Gold prompt tuning |
| Scenario prompts include context, task, positive/negative cases, output format | Inputs/Prompts.json fixes entity, synonym, hierarchy, and relation prompts | Seven entity types, seven relation patterns, and the study's topic scope |
| Shared indices for synonyms; representative candidates followed by synonym expansion | Deterministic clusters select a representative; outputs list the aliases actually expressed in evidence | Expand only forms directly supported by the same continuous evidence, not other co-occurring disease names |
| Hierarchy identification and upper-concept filtering | Assess hierarchy between distinct clusters and pass parent clusters into relation judgment | Retain a directly expressed upper-disease relation; do not inherit it solely from a lower-concept relation |
| Independently judge Schema-generated candidates | Full passage and positive/negative scenarios are sent for each compatible candidate | Biomedical chemical/gene/disease relations are replaced by the fixed topic Schema; Qwen controls backbone differences from ECKE |
| Original local Llama 3.1 70B and biomedical data | Online qwen-plus-2025-07-28, temperature 0, fixed seed, 300 test passages | Task adaptation of algorithmic structure, not reproduction of original published scores |

Evidence constraints implement the common annotation scope; ECKE's lexicon anchoring, explicit–implicit fusion, and other core modules are not injected. Prediction reads neither the domain lexicon, ECKE predictions, nor Gold. Entity surfaces must occur in the source. Relations require continuous textual evidence containing the actual endpoints. Candidates, clusters, hierarchy judgments, and outputs are retained for audit.

Synonym/hierarchy candidates are queried individually, preserving the author's candidate-level granularity. API format, JSON handling and resume caching are environment adaptations. Report these differences rather than claiming identical original settings.
