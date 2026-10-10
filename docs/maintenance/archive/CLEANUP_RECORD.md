# Cleanup evidence — 2026-10-10

The requested repository cleanup and JOAQUIM transfer are complete. The published archive contains **2,050 files, 1,252,400,031 bytes**. Transfer checked source and destination SHA-256, embedded freeze/ZIP entries and bundle links. A separate verification passed before prune; prune independently verified the archive again and checked all local sources before removal. [Completion record](completed-transfer.json) records every removed path. No transfers remain blocked.

## Final folder map

| Folder | Retained purpose |
|---|---|
| `Plans/` | Fourteen numbered work packages, README, sole implementation status index |
| `docs/analysis/decisions/` | Scientific decisions and bout-only policy |
| `docs/analysis/audits/` | Exclusion/selection audit |
| `docs/analysis/figures/reviews/` | Paper/panel concerns; all 27 proposed main panels, supplementary placeholders grouped by parent |
| `docs/analysis/figures/freezes/` | Human-readable scope/authority and historical freeze narratives |
| `docs/maintenance/` | Governance and current/historical transfer guidance |
| `configs/paper-figures/selections/` | Seven original byte-identical selection pointers |
| `records/frozen-analyses/` | Seventy exact scientific definition/manifest/settings/code entries |
| `reviews/`, `outputs/` | Navigation READMEs; payloads external |
| `src/`, `tests/`, `scripts/`, `legacy/` | Supported code, tests, adapters and consumed reproduction/comparison routes |
| `docs/maintenance/archive/` | Inventories, relocations, review catalogue, recovery and verification evidence |

The first release remains main Figures 1–4 with required evidence. Supplementary composition, tail mechanisms and imaging remain separate. Substantive handoff definitions, approvals, limitations and unfinished work have durable destinations; remaining work is assigned to the owning plans. `Handovers, etc` is removed.

## Scientific preservation and review consolidation

Seven selection files moved byte-for-byte. All 70 exact frozen record/code entries retain their original hashes, including historical accepted freezes and the selected E/G evidence extracted after read-only SSD authentication. No scientific definition, numerical result, presentation style or freeze was changed. Active links/readers use relocation records independently of immutable manifests.

[Review catalogue](REVIEW_CATALOGUE.md) identifies the current consolidated review per scientific lineage. Unique alternatives and evidence remain archived with their bundle paths. Exact duplicate identities are indexed; copies were retained on the SSD where bundle links require them. Four pre-existing broken relative links in historical F/G/H HTML are recorded with verified navigation targets in [historical link recovery](historical-link-recovery.json); original HTML bytes remain unchanged.

[Retained code inventory](retained-code.json) records supported tools and frozen snapshots. Named superseded one-off scripts were archived only where no supported source/test consumer or frozen dependency was found. Legacy helpers remain where supported reproduction/tests consume them. Working environment and lockfile are preserved; generated caches in emptied review/output bundles were removed.

## Space and recoverability

Working-tree payload reduction: **1,252,400,031 bytes**. Additional previously verified deletions: **74,386 bytes** (integrated 62,715-byte Figure 4 patch, disposable 6,148-byte Finder metadata and a reappeared 5,523-byte exact duplicate of the preserved request context). The patch is recoverable as Git blob `1348dae7141f369b18d810b3fc186c013db98b0d`. Git history was not rewritten, so committed historical objects still occupy `.git`; these figures describe working-tree reduction, not total disk reclamation.

[Starting inventory](starting-inventory.json) and [transfer checkpoint](transfer-inventory.json) remain immutable historical evidence. Their pending states are superseded by [completed transfer](completed-transfer.json). Post-transfer control files and current retained files are audited separately.

## Verification

Full post-transfer suite and final audit results are recorded in [verification.json](verification.json). Tests cover absent drive, changed sources, corruption, collisions, interrupted transfer, traversal, embedded hash mismatches and relocation. Scientific recomputation, raw-data movement, restyling, refreezing and Git-history rewriting did not occur.

Two initially failing pre-existing tests were repaired: legacy oracle tests now restore scoped import paths, and trace preparation checks the existing author-selected metric instead of a superseded expectation. Scientific implementation was not changed by those repairs.
