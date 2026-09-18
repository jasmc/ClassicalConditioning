# Analysis glossary

| Term | Meaning in this repository |
| --- | --- |
| **Candidate route** | The supported active analysis workflow. Its results are exploratory until scientific gates are approved. |
| **Recipe** | A versioned, named set of compatible stage behaviour and artifact identities. |
| **Recording ID** | Acquisition identity in the form `YYYYMMDD_NN`: date plus fish number, without condition. |
| **Fish key** | Stable biological identity: experiment, day, and fish number. |
| **CS** | Conditioned stimulus; one possible time-alignment event. |
| **US** | Unconditioned stimulus; another possible time-alignment event. |
| **Trial map** | Frozen expected trial/alignment/phase/block mapping for an experiment. |
| **Condition** | Experimental group encoded in raw recording naming and mapped to an experiment specification. |
| **Cohort** | Explicitly reviewed set of fish, frozen in a manifest; not merely a filename filter. |
| **Metric** | A frame-level numerical description of tail activity. The active route carries three candidates. |
| **Bout** | A detected episode of movement from the one shared detector, independent of which metric describes it. |
| **Temporal profile** | Trial-aligned, time-binned per-recording summary used for figures and trial outcomes. |
| **Trial outcome** | Baseline/response summary for one trial and outcome definition. |
| **Coverage** | Evidence about valid/expected samples or observed time; low coverage is not the same as low activity. |
| **Completion marker** | A metadata JSON artifact that authenticates a completed stage’s outputs and upstream lineage. |
| **QC** | Quality-check evidence: reports, summaries, diagnostics, and images for human review. |

These definitions describe current package terminology; historical scripts may
use similar words differently. Consult the active module and recipe identity
before comparing historical and candidate outputs.
