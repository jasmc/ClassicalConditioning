# Analysis Architecture and Alternative Routes

## Final integrated analysis pipeline

This is the intended end state for the entire local analysis. Behavior is the
required primary modality. Imaging is an optional, additive branch available
only for recordings whose resolved acquisition capabilities declare it.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F6F6FA', 'primaryTextColor': '#2E2E38', 'primaryBorderColor': '#C4C4CD', 'lineColor': '#747480', 'secondaryColor': '#FFE600', 'tertiaryColor': '#4696FF', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph TD
    GOV["Scientific scope, immutable source inventory,<br/>approved recipes and acquisition metadata"]:::dark
    CAP["Resolved recording capabilities<br/>behavior required; imaging optional"]:::highlight
    BRAW["Behavior sources<br/>camera + tracking + protocol"]
    IRAW["Optional imaging sources<br/>galvo + functional TIFF + anatomy"]
    BINT["Lossless behavior intake<br/>and acquisition QC"]:::accent
    TRIAL["Canonical identities, clocks,<br/>stimulus events and trial map"]
    BPIPE["Canonical behavior analysis<br/>legacy-equivalent or approved corrected"]
    BOUT["Behavior frame, bout,<br/>trial and fish outcomes"]
    ICAP{"Imaging declared<br/>for recording?"}
    INA["Record imaging as<br/>not applicable"]
    IINT["Imaging intake<br/>source and array validation"]
    SYNC["Behavior-imaging synchronization<br/>mapping + residuals + uncertainty"]
    REG["Motion correction and registration<br/>within trial + across trial/plane"]
    IQC["Imaging frame, trial<br/>and plane QC"]
    PIX["Pixel response maps<br/>named response recipes"]
    ROI{"Approved imaging route"}
    SR["Suite2p cell route"]
    CR["Correlation-grown ROI route<br/>experimental until validated"]
    IOUT["Imaging trial, plane<br/>and ROI outcomes"]
    JOIN["Canonical multimodal integration<br/>behavior-primary left join"]:::highlight
    BC["Behavior-primary cohort<br/>independent of imaging availability"]
    MC["Imaging-valid and<br/>multimodal cohorts"]
    BS["Primary behavior statistics<br/>and sensitivity"]
    IS["Imaging statistics<br/>fish-aware hierarchy"]
    XS["Cross-modal associations<br/>fish/trial/plane/ROI aware"]
    LEARN["Optional learner analysis<br/>separately validation-gated"]
    PANEL["Frozen table, model<br/>and panel-data artifacts"]
    FIG["Publication SVG/PDF, review PNG,<br/>local HTML/notebooks"]:::success
    REL["Immutable behavior release<br/>with optional multimodal extension"]:::dark

    GOV --> CAP
    CAP --> BRAW
    CAP --> IRAW
    BRAW --> BINT --> TRIAL
    TRIAL --> BPIPE --> BOUT
    CAP --> ICAP
    ICAP -->|No| INA
    ICAP -->|Yes| IINT
    IRAW --> IINT
    BINT --> SYNC
    TRIAL --> SYNC
    IINT --> SYNC --> REG --> IQC
    IQC --> PIX --> IOUT
    IQC --> ROI
    ROI --> SR --> IOUT
    ROI --> CR --> IOUT
    BOUT --> JOIN
    IOUT -. optional imaging outcomes .-> JOIN
    INA -. availability state .-> JOIN
    BOUT --> BC
    JOIN --> MC
    BC --> BS
    BC --> LEARN
    IOUT --> IS
    MC -. cohort eligibility .-> IS
    JOIN --> XS
    MC -. cohort eligibility .-> XS
    BS --> PANEL
    IS --> PANEL
    XS --> PANEL
    LEARN --> PANEL
    PANEL --> FIG --> REL

    classDef highlight fill:#FFE600,stroke:#2E2E38,stroke-width:2px,color:#2E2E38
    classDef accent fill:#4696FF,stroke:#4696FF,stroke-width:1px,color:#FFFFFF
    classDef success fill:#2DB757,stroke:#2DB757,stroke-width:1px,color:#FFFFFF
    classDef dark fill:#2E2E38,stroke:#1A1A24,stroke-width:1px,color:#F6F6FA
```

The canonical behavior path does not depend on the imaging capability decision.
The synchronization stage may consume behavior clocks, events, and trial
identity, but no imaging stage may publish or modify behavior artifacts.

The integration stage is a behavior-primary left join. Behavior-only trials stay
in the final table; unavailable or invalid imaging is represented by explicit
availability/QC status and missing imaging values, never biological zero.

Implementation of the optional branch is deferred to
[the behavior and imaging integration plan](./BEHAVIOR_IMAGING_INTEGRATION_PLAN.md).
The inherited imaging implementation is assessed separately in
[the imaging pipeline critique](../docs/analysis/IMAGING_PIPELINE_CRITIQUE.md).

## Modality dependency rule

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F6F6FA', 'primaryTextColor': '#2E2E38', 'primaryBorderColor': '#C4C4CD', 'lineColor': '#747480', 'secondaryColor': '#FFE600', 'tertiaryColor': '#4696FF', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph LR
    RB["Raw behavior artifacts"] --> BA["Canonical behavior analysis"] --> BO["Behavior outcomes"]
    RB --> SY["Clock/event synchronization"]
    RI["Raw imaging artifacts"] --> SY --> IA["Imaging analysis"] --> IO["Imaging outcomes"]
    BO --> MI["Multimodal integration"]:::highlight
    IO --> MI
    IA -. prohibited dependency .-> X["No behavior recalculation"]:::danger

    classDef highlight fill:#FFE600,stroke:#2E2E38,stroke-width:2px,color:#2E2E38
    classDef danger fill:#FF4136,stroke:#FF3C00,stroke-width:1px,color:#FFFFFF
```

Required invariant:

```text
logical_hash(behavior outputs | imaging disabled)
    == logical_hash(behavior outputs | imaging enabled)
```

## Detailed behavior branch

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F6F6FA', 'primaryTextColor': '#2E2E38', 'primaryBorderColor': '#C4C4CD', 'lineColor': '#747480', 'secondaryColor': '#FFE600', 'tertiaryColor': '#4696FF', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph TD
    RAW["Immutable acquisition<br/>camera + tracking + protocol"]:::dark
    IN["Lossless local ingestion<br/>Parquet + JSON"]:::highlight
    CHECK{"Acquisition integrity<br/>acceptable?"}
    REVIEW["Review reader, acquisition,<br/>or protocol issue"]:::warning
    PREP{"Preprocessing route"}
    LEGACY["Legacy-equivalent<br/>timing, filter, point-15"]:::muted
    CORRECTED["Approved corrected<br/>timing, gaps, filtering"]:::accent
    REP{"Tail representation"}
    ANGLE["Local/cumulative angles"]
    XYM["Measured body-centred XY"]
    XYR["Reconstructed XY<br/>only if validated"]
    METRIC{"Activity metrics"}
    M1["Legacy distal speed"]
    M2["Manuscript segment-speed sum"]
    M3["All-point angular RMS"]
    M4["Whole-tail XY RMS"]
    M5["Whole-tail XY mean"]
    M6["Curvature-change RMS"]
    DETECT{"Movement route"}
    D1["Legacy threshold"]
    D2["Calibrated threshold/hysteresis"]
    D3["Probabilistic state<br/>exploratory"]
    OUT["Common trial mapping<br/>and frozen outcome functions"]
    O1["Total activity"]
    O2["Movement probability"]
    O3["Conditional intensity"]
    O4["Bout rate and duration"]
    COHORT{"Population route"}
    C1["Primary technical-valid cohort"]:::highlight
    C2["Sensitivity cohorts"]
    C3["Legacy cohort comparison"]
    ANALYSIS{"Analysis route"}
    A1["Primary population inference"]
    A2["Learner classification<br/>optional / validation-gated"]
    A3["PCA, rhythmicity, waves<br/>exploratory"]
    FIG{"Figure mode"}
    F1["Publication<br/>SVG + PDF"]:::success
    F2["Static review<br/>PNG"]
    F3["Interactive local<br/>self-contained HTML/notebook"]

    RAW --> IN --> CHECK
    CHECK -->|No / uncertain| REVIEW
    REVIEW --> IN
    CHECK -->|Yes| PREP
    PREP --> LEGACY
    PREP --> CORRECTED
    LEGACY --> REP
    CORRECTED --> REP
    REP --> ANGLE
    REP --> XYM
    REP --> XYR
    ANGLE --> METRIC
    XYM --> METRIC
    XYR --> METRIC
    METRIC --> M1
    METRIC --> M2
    METRIC --> M3
    METRIC --> M4
    METRIC --> M5
    METRIC --> M6
    M1 --> DETECT
    M2 --> DETECT
    M3 --> DETECT
    M4 --> DETECT
    M5 --> DETECT
    M6 --> DETECT
    DETECT --> D1
    DETECT --> D2
    DETECT --> D3
    D1 --> OUT
    D2 --> OUT
    D3 --> OUT
    OUT --> O1
    OUT --> O2
    OUT --> O3
    OUT --> O4
    O1 --> COHORT
    O2 --> COHORT
    O3 --> COHORT
    O4 --> COHORT
    COHORT --> C1
    COHORT --> C2
    COHORT --> C3
    C1 --> ANALYSIS
    C2 --> ANALYSIS
    C3 --> ANALYSIS
    ANALYSIS --> A1
    ANALYSIS --> A2
    ANALYSIS --> A3
    A1 --> FIG
    A2 --> FIG
    A3 --> FIG
    FIG --> F1
    FIG --> F2
    FIG --> F3

    classDef highlight fill:#FFE600,stroke:#2E2E38,stroke-width:2px,color:#2E2E38
    classDef accent fill:#4696FF,stroke:#4696FF,stroke-width:1px,color:#FFFFFF
    classDef success fill:#2DB757,stroke:#2DB757,stroke-width:1px,color:#FFFFFF
    classDef warning fill:#FF6D00,stroke:#FF6D00,stroke-width:1px,color:#FFFFFF
    classDef muted fill:#F6F6FA,stroke:#C4C4CD,stroke-width:1px,color:#2E2E38
    classDef dark fill:#2E2E38,stroke:#1A1A24,stroke-width:1px,color:#F6F6FA
```

All nodes inside this scheme execute locally. There is no Databricks route.

## Final storage architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F6F6FA', 'primaryTextColor': '#2E2E38', 'primaryBorderColor': '#C4C4CD', 'lineColor': '#747480', 'secondaryColor': '#FFE600', 'tertiaryColor': '#4696FF', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph LR
    BRAW["Immutable behavior sources<br/>TXT and acquisition metadata"]:::dark
    IRAW["Immutable imaging sources<br/>TIFF, galvo and anatomy"]:::dark
    PARQUET["Lossless Parquet tables<br/>scientific tables and mappings"]:::highlight
    DENSE["Benchmarked dense-array format<br/>imaging and frame x point arrays"]
    JSON["JSON metadata<br/>schemas + provenance"]
    MAN["Artifact manifest<br/>logical hashes + lineage"]
    TABLES["Outcomes, cohorts,<br/>statistics and panel data"]
    PUB["Publication<br/>SVG + PDF"]:::success
    PNG["Static<br/>PNG"]
    HTML["Interactive local<br/>HTML/notebook"]

    BRAW --> PARQUET
    IRAW --> DENSE
    IRAW --> PARQUET
    PARQUET -. benchmark if dense .-> DENSE
    PARQUET --> JSON
    DENSE --> JSON
    PARQUET --> MAN
    DENSE --> MAN
    JSON --> MAN
    PARQUET --> TABLES
    DENSE --> TABLES
    TABLES --> PUB
    TABLES --> PNG
    TABLES --> HTML

    classDef highlight fill:#FFE600,stroke:#2E2E38,stroke-width:2px,color:#2E2E38
    classDef success fill:#2DB757,stroke:#2DB757,stroke-width:1px,color:#FFFFFF
    classDef dark fill:#2E2E38,stroke:#1A1A24,stroke-width:1px,color:#F6F6FA
```

## Route selection rules

| Decision | Available routes | Rule |
| --- | --- | --- |
| Preprocessing | Legacy or corrected | Legacy proves equivalence; corrected requires scientific gate P |
| Legacy downstream transform | Standard `main` route or historical LogMedian route | Characterize separately; neither description overrides executable behavior |
| Tail representation | Angles, measured XY, reconstructed XY | Use measured XY when valid; reconstruct only with proven angle/length semantics |
| Activity metric | Six explicit metrics | Compare as distinct quantities; select using frozen validation |
| Movement state | Legacy, calibrated, probabilistic | Detector identity is independent of metric identity |
| Outcome | Total, probability, conditional, bouts | Complementary outcomes, not interchangeable versions |
| Cohort | Primary, sensitivity, legacy | One explicit manifest per population |
| Analysis | Population, learner, mechanistic | Population is primary; learner/mechanistic routes have separate gates |
| Imaging capability | Not applicable, expected, valid, invalid | Resolve per recording; source presence cannot silently select a route |
| Synchronization | Matched, unmatched, synthesized under approved recovery | Preserve residual and uncertainty; never infer events from fish names |
| Registration | Within-trial, cross-trial/plane, anatomy positioning | Save separately and name the final product consumed downstream |
| Imaging response | Pixel maps, Suite2p cells, correlation-grown ROIs | Distinct scientific methods with separate recipe IDs and gates |
| Multimodal cohort | Imaging-acquired, imaging-valid, multimodal | Never replace or redefine the behavior-primary cohort |
| Multimodal join | Behavior-primary left join | Missing imaging remains unavailable, not zero; reject one-to-many expansion |
| Figure | Publication, static, interactive | Same panel data; presentation mode cannot change science |
