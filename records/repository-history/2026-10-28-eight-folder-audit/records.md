# Independent audit: records

Audit date: 28 October 2026. Baseline: `702193b9591b62bf12820bc2b01ca5e8928f9528`.
Focused review of category navigation, preservation enforcement, recovery roles
and recent maintenance discoverability. No historical record was rewritten.

## Ownership

Keep exact freezes, dated scientific evidence, version-bound figure reviews,
completed repository history and archive control/recovery records. Exact source
snapshots are evidence and belong here when required by a protected bundle;
maintained reusable runtime code belongs in `src/`. Permanent media, numerical
payloads and verbose audit inventories remain external.

## Findings and dispositions

- **R1 — Improve current history navigation.** [Repository-history navigation](../README.md)
  omits direct entries for the completed 27 October consolidation and 28 October
  offline-cache retirement, although both have local READMEs and current storage
  guidance links them. Add links to
  `2026-10-27-storage-consolidation/README.md` and
  `2026-10-28-offline-cache-retirement/README.md`. The dated description of
  23 October CI introduction should explicitly say it is historical; current
  no-CI policy remains in the rules owner. Do not rewrite that dated evidence.
- **R2 — Preserve record-role exceptions.** Frozen records contain original code
  and settings; repository history contains migration-original documents,
  including an old `AGENTS.md`. These are evidence, not active instructions or
  duplicate supported implementation. Keep exact bytes and recorded aliases.
  The structure checker separately enforces frozen-index identities and named
  mutable-index originals; file extension cannot select mutability.
- **R3 — Retain recovery indexes and categories.**
  [Archive navigation](../../repository-archive/README.md) distinguishes mutable
  relocation/expected-version indexes from immutable published transactions.
  The large relocation index supplies exact historical identities and current
  source recovery. Splitting or deduplicating it for tidiness is unsupported by
  this review. No payload or record deletion is established as safe.
- **R4 — Separate maintenance evidence from artwork discovery.**
  `scripts/reviews/versions.py:63–80` walks all archive files, exempting only two
  specifically named maintenance directory families. Placing this audit's logs
  in the otherwise appropriate archive/repository-history area produced twelve
  new-source errors in the external check. Those were this task's files, not
  pre-existing corrupt artifacts. They were moved to the explicit JOAQUIM
  `ClassicalConditioning Maintenance/repository-history/2026-10-28-eight-folder-audit/`
  evidence home. Propose a documented, narrowly registered maintenance-control
  exclusion in the existing storage/browser contract, with tests ensuring no
  artwork is hidden; do not reuse another task's exempt directory prefix or
  rebuild selected figure navigation just to store maintenance logs.

## Proposed instruction wording — not active

Owner: `records/README.md`; detailed storage/preservation rules stay central.

> File records by evidentiary role: accepted frozen versions, scientific support,
> figure history/review, archive control or completed repository maintenance.
> Identify scope and date, distinguish a maintained navigation index from the
> evidence it indexes, and link current guidance rather than copying its status.
> Exact source/document snapshots are retained evidence, not instructions to
> execute. Correct immutable evidence through a linked successor/addendum;
> change a recovery index only under its supported format and preservation rules.
> Keep verbose inventories and permanent payloads at their external destinations.

## Verification and limits

Baseline local checking authenticated 366 protected requirements and ten states.
External checking was also requested; its result is recorded separately in the
audit validation receipt. The review did not read all 567 tracked record files
or authenticate every historical narrative's claims. Navigation additions need
direct target checks and the normal local structure check; no record conversion
or new recovery schema is proposed.
