# Final Architecture and Current Implementation Map

This map preserves the complete intended architecture while showing what is
implemented now, what exists only as a single-recording candidate, what should
be implemented next, and what remains planned or deliberately deferred.

All execution is local. Raw scientific data is immutable. Behavior is the
required primary modality; imaging is optional and additive.

## Status legend

| Style | Meaning |
| --- | --- |
| Green | Implemented, tested, and locally runnable |
| Orange | Implemented on bounded pilot/prototype inputs; not paper-authoritative |
| Yellow | Immediate implementation priority |
| Off white | Preserved final-plan work not implemented yet |
| Purple | Explicitly deferred optional work |
| Dark | Immutable boundary or final release boundary |

## Complete behavior architecture with current status

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F6F6FA', 'primaryTextColor': '#2E2E38', 'primaryBorderColor': '#C4C4CD', 'lineColor': '#747480', 'secondaryColor': '#FFE600', 'tertiaryColor': '#4696FF', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph TD
    subgraph FOUNDATION["Foundation and control plane"]
        GOV["Paper scope, owners and<br/>approved scientific decisions"]:::planned
        ENV["Pinned uv environment<br/>typed errors + environment report"]:::implemented
        ART["Artifact integrity<br/>SHA-256 + atomic publication"]:::implemented
        ART2["Full schema registry, logical hashes,<br/>resolver and conversion registry"]:::planned
        CFG["Immutable allDelay domain/config<br/>legacy recipe + stage hashes"]:::pilot
        CFG2["Complete experiment/config audit<br/>all retained experiments"]:::planned
    end

    subgraph SOURCES["Immutable acquisition boundary"]
        RAW["Raw behavior sources<br/>camera TXT + tracking TXT + protocol TXT"]:::boundary
        INV["Recursive recording inventory<br/>completeness + collision checks + hashes"]:::implemented
    end

    subgraph INTAKE["Canonical behavior intake"]
        BATCH["Resumable multi-recording runner<br/>dry-run + skip verified stages + failure report"]:::priority
        LOSSLESS["Lossless Zstandard Parquet intake<br/>camera + tracking + protocol"]:::implemented
        ACQ["Acquisition integrity QC<br/>frames + clocks + source hashes"]:::implemented
        IDS["Canonical fish, recording, event<br/>trial, alignment, phase and block identities"]:::pilot
        COHORTRAW["Complete local multi-fish<br/>source inventory"]:::planned
    end

    subgraph LEGACY["Frozen legacy-preservation route"]
        L1["legacy-paper-v1<br/>stage-1 preprocessing"]:::implemented
        L3["historical-logmedian-v1<br/>bounded-memory stage-3 transform"]:::implemented
        L45["Legacy stages 4-5<br/>aggregation, windows and statistics"]:::pilot
        LVAR["Four historical learner variants<br/>characterized and frozen"]:::pilot
        LREPORT["Legacy behavior and issue reports<br/>difference evidence"]:::implemented
    end

    subgraph CORRECTED["Corrected and candidate-development route"]
        CPREP["Approved corrected preprocessing<br/>gaps + measured time + filtering + missingness"]:::planned
        TAIL["Shared tail representation<br/>angles + measured body-centered XY"]:::pilot
        M1["Manuscript segment<br/>angular-speed sum"]:::pilot
        M2["All-segment<br/>angular RMS"]:::pilot
        M3["Whole-tail XY<br/>RMS speed"]:::pilot
        M4["Whole-tail XY<br/>mean speed"]:::pilot
        M5["Curvature-change<br/>RMS"]:::pilot
        MOVE["Generic movement-state layer<br/>smoothing + calibration + hysteresis + bouts"]:::pilot
        SENS["Smoothing sensitivity<br/>and US positive controls"]:::pilot
        TRACE["Balanced trace review<br/>PNG + HTML + annotation CSV"]:::pilot
        MANUAL["Multi-fish manual/video validation<br/>and detector robustness grid"]:::planned
    end

    subgraph OUTCOMES["Common outcomes and population analysis"]
        OUTFN["Generic measured-time outcomes<br/>all five metrics"]:::pilot
        O1["Total activity"]:::pilot
        O2["Movement probability<br/>and fraction moving"]:::pilot
        O3["Conditional intensity"]:::pilot
        O4["Bout count, rate<br/>and duration"]:::pilot
        COHORT["Immutable cohort manifests<br/>primary + sensitivity + legacy"]:::planned
        SCALE["Corrected scaling and<br/>normalized outcomes"]:::planned
        MODEL["Fish-aware mixed-effects models<br/>contrasts + diagnostics + validation"]:::planned
        LEGSTAT["Frozen legacy statistics<br/>comparison route"]:::planned
        COMPARE["Five-metric comparison report<br/>coverage + controls + effects + sensitivity"]:::planned
    end

    subgraph DELIVER["Frozen deliverables"]
        PANEL["Frozen tables, model results<br/>and panel-data artifacts"]:::planned
        PNG["Static review figures<br/>PNG"]:::pilot
        PUB["Publication figures<br/>semantic SVG + PDF + provenance"]:::pilot
        FINALFIG["Final manuscript figures<br/>from approved cohort/results"]:::planned
        RELEASE["Immutable reproduction release<br/>code + environment + recipes + hashes"]:::boundary
    end

    RAW --> INV --> BATCH
    ENV --> BATCH
    ART --> LOSSLESS
    ART -. future extension .-> ART2
    CFG --> IDS
    CFG -. future extension .-> CFG2
    GOV --> BATCH
    COHORTRAW --> BATCH
    BATCH --> LOSSLESS --> ACQ --> IDS

    IDS --> L1 --> L3 --> L45
    L45 --> LVAR
    L1 --> LREPORT
    L3 --> LREPORT
    L45 --> LREPORT

    IDS --> CPREP --> TAIL
    TAIL --> M1
    TAIL --> M2
    TAIL --> M3
    TAIL --> M4
    TAIL --> M5
    M1 --> MOVE
    M2 --> MOVE
    M3 --> MOVE
    M4 --> MOVE
    M5 --> MOVE
    MOVE --> SENS
    MOVE --> TRACE
    SENS --> MANUAL
    TRACE --> MANUAL

    M1 --> OUTFN
    M2 --> OUTFN
    M3 --> OUTFN
    M4 --> OUTFN
    M5 --> OUTFN
    MOVE --> OUTFN
    OUTFN --> O1
    OUTFN --> O2
    OUTFN --> O3
    OUTFN --> O4
    O1 --> COHORT
    O2 --> COHORT
    O3 --> COHORT
    O4 --> COHORT
    COHORT --> SCALE --> MODEL
    L45 --> LEGSTAT
    MODEL --> COMPARE
    LEGSTAT --> COMPARE
    LREPORT --> COMPARE

    COMPARE --> PANEL
    PANEL --> PNG
    PANEL --> PUB
    PNG --> FINALFIG
    PUB --> FINALFIG
    FINALFIG --> RELEASE
    ART2 -. final provenance .-> RELEASE
    CFG2 -. final recipes .-> RELEASE

    classDef implemented fill:#2DB757,stroke:#2DB757,stroke-width:1px,color:#FFFFFF
    classDef pilot fill:#FF6D00,stroke:#FF6D00,stroke-width:1px,color:#FFFFFF
    classDef priority fill:#FFE600,stroke:#2E2E38,stroke-width:2px,color:#2E2E38
    classDef planned fill:#F6F6FA,stroke:#C4C4CD,stroke-width:1px,color:#2E2E38
    classDef deferred fill:#3D108A,stroke:#3D108A,stroke-width:1px,color:#FFFFFF
    classDef boundary fill:#2E2E38,stroke:#1A1A24,stroke-width:1px,color:#F6F6FA
```

## Optional imaging and learner extensions

These branches remain part of the complete architecture. They are not required
to complete the current five-metric behavior comparison.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F6F6FA', 'primaryTextColor': '#2E2E38', 'primaryBorderColor': '#C4C4CD', 'lineColor': '#747480', 'secondaryColor': '#FFE600', 'tertiaryColor': '#4696FF', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph LR
    subgraph BEHAVIOR["Frozen behavior outputs"]
        BO["Behavior trial/fish outcomes"]:::planned
        BC["Behavior-primary cohort"]:::planned
        BR["Behavior release"]:::boundary
    end

    subgraph LEARNER["Deferred learner-methodology track"]
        LQ["Question whether discrete<br/>single-fish classification is justified"]:::deferred
        LA["Compare continuous scores, trajectories,<br/>mixtures, PCA and hierarchical alternatives"]:::deferred
        LV["Power, simulation and<br/>held-out validation"]:::deferred
        LO["Optional learner outputs<br/>only if scientifically retained"]:::deferred
    end

    subgraph IMAGING["Deferred optional imaging track I00-I12"]
        IR["Immutable TIFF + galvo<br/>+ anatomy sources"]:::boundary
        II["Imaging intake<br/>source and array validation"]:::deferred
        SY["Clock/event synchronization<br/>residuals + uncertainty"]:::deferred
        RG["Within/across-trial registration<br/>and motion correction"]:::deferred
        IQ["Frame, trial and plane QC"]:::deferred
        PX["Pixel response maps"]:::deferred
        ROI{"Approved ROI route"}:::deferred
        S2["Suite2p cell route"]:::deferred
        CR["Correlation-grown ROI route<br/>experimental"]:::deferred
        IO["Imaging trial, plane<br/>and ROI outcomes"]:::deferred
        JOIN["Behavior-primary left join<br/>missing imaging is unavailable, not zero"]:::deferred
        MC["Imaging-valid and<br/>multimodal cohorts"]:::deferred
        MS["Imaging and cross-modal<br/>fish-aware statistics"]:::deferred
        MR["Optional multimodal<br/>release extension"]:::boundary
    end

    BO --> LQ --> LA --> LV --> LO
    BC --> LQ
    IR --> II --> SY --> RG --> IQ --> PX
    IQ --> ROI
    ROI --> S2 --> IO
    ROI --> CR --> IO
    BO --> JOIN
    IO --> JOIN
    BC --> JOIN
    JOIN --> MC --> MS --> MR
    BR --> MR

    classDef implemented fill:#2DB757,stroke:#2DB757,stroke-width:1px,color:#FFFFFF
    classDef pilot fill:#FF6D00,stroke:#FF6D00,stroke-width:1px,color:#FFFFFF
    classDef priority fill:#FFE600,stroke:#2E2E38,stroke-width:2px,color:#2E2E38
    classDef planned fill:#F6F6FA,stroke:#C4C4CD,stroke-width:1px,color:#2E2E38
    classDef deferred fill:#3D108A,stroke:#3D108A,stroke-width:1px,color:#FFFFFF
    classDef boundary fill:#2E2E38,stroke:#1A1A24,stroke-width:1px,color:#F6F6FA
```

Required behavior/imaging invariant:

```text
behavior outputs with imaging disabled
    == behavior outputs with imaging enabled
```

Imaging may consume canonical behavior clocks, event identities, and outcomes.
It may never recalculate, overwrite, exclude, or redefine behavior artifacts or
the behavior-primary cohort.

## Current runnable reality

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F6F6FA', 'primaryTextColor': '#2E2E38', 'primaryBorderColor': '#C4C4CD', 'lineColor': '#747480', 'secondaryColor': '#FFE600', 'tertiaryColor': '#4696FF', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph LR
    NOW["One local complete recording<br/>20221115_04"]:::boundary
    I["Authenticated inventory"]:::implemented
    P["Lossless Parquet + QC"]:::implemented
    L["Legacy stage 1 + LogMedian"]:::implemented
    F["Five candidate frame metrics"]:::pilot
    M["Movement + sensitivity<br/>+ trace review"]:::pilot
    O["Five-metric temporal outcomes"]:::pilot
    G["PNG + semantic SVG/PDF<br/>rendering infrastructure"]:::pilot
    N["Next: resumable<br/>multi-recording runner"]:::priority
    NEED["Needs additional local raw recordings<br/>for cohort-scale execution"]:::planned
    C["Then: cohort + scaling<br/>+ statistics + comparison"]:::planned

    NOW --> I --> P
    P --> L
    P --> F --> M --> O --> G
    G --> N --> NEED --> C

    classDef implemented fill:#2DB757,stroke:#2DB757,stroke-width:1px,color:#FFFFFF
    classDef pilot fill:#FF6D00,stroke:#FF6D00,stroke-width:1px,color:#FFFFFF
    classDef priority fill:#FFE600,stroke:#2E2E38,stroke-width:2px,color:#2E2E38
    classDef planned fill:#F6F6FA,stroke:#C4C4CD,stroke-width:1px,color:#2E2E38
    classDef boundary fill:#2E2E38,stroke:#1A1A24,stroke-width:1px,color:#F6F6FA
```

## Final storage architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F6F6FA', 'primaryTextColor': '#2E2E38', 'primaryBorderColor': '#C4C4CD', 'lineColor': '#747480', 'secondaryColor': '#FFE600', 'tertiaryColor': '#4696FF', 'fontFamily': 'Arial, Noto Sans, sans-serif'}}}%%
graph TD
    RAW["Paper data/Raw single fish data<br/>immutable TXT and optional TIFF/galvo"]:::boundary
    PROC["Processed data/recording-id<br/>behavior + optional imaging + multimodal"]
    QC["Quality checks/recording-id<br/>behavior + optional imaging + multimodal"]
    META["Metadata<br/>inventories + recipes + manifests + hashes"]
    TAB["Tables<br/>outcomes + cohorts + comparisons"]
    MOD["Models<br/>fits + contrasts + diagnostics"]
    FIG["Figures<br/>PNG + publication SVG/PDF"]
    REL["Frozen release manifest<br/>code + environment + inputs + outputs"]:::boundary

    RAW --> PROC
    RAW --> QC
    RAW --> META
    PROC --> TAB
    QC --> TAB
    META --> TAB
    TAB --> MOD
    TAB --> FIG
    MOD --> FIG
    META --> REL
    TAB --> REL
    MOD --> REL
    FIG --> REL

    classDef boundary fill:#2E2E38,stroke:#1A1A24,stroke-width:1px,color:#F6F6FA
```

## Immediate priority lane

The complete architecture above remains authoritative. Current execution
priority is narrower:

1. Implement the resumable multi-recording runner.
2. Make the remaining local raw recording triplets available and inventory them.
3. Run identical legacy and five-metric candidate stages for every recording.
4. Freeze and compare primary, sensitivity, and legacy cohorts.
5. Run identical fish-aware models for all five metrics.
6. Produce the comparison report and final PNG/SVG/PDF figures.
7. Revisit learner methodology only after the behavior results are frozen.
8. Implement the optional imaging branch later without changing behavior hashes.

