# Folder-audit implementation source recovery

- [Exact pre-change source snapshot inventory](source-snapshot-inventory.json):
  523 files copied and verified through the supported archive tool; originals
  retained unless an independent scoped disposition says otherwise.
- [Command successor routing](command-successors.json): seven compatibility
  facades and one exact retained source-bound original.
- [Plan 13 transfer inventory](plan-13/transfer-inventory.json) and
  [completion](plan-13/completed-transfer.json): one completed plan retired after
  local preservation, alias publication and external verification.
- [Implementation/preservation evidence](../../repository-history/2026-10-28-folder-audit-implementation/README.md).

The source snapshot inventory deliberately has a distinct name: it preserves
versions of live mutable source and is not an implicit latest-source selector
or authorization to prune all snapshot members. Use its explicit path with
`repository_archive.py verify --manifest ...` for strict copy authentication.
The published external checkpoint remains immutable. No historical source hash,
selection or scientific definition is replaced by a successor's identity.

- [Plan 15 transfer inventory](plan-15/transfer-inventory.json) and
  [completion](plan-15/completed-transfer.json): completed follow-up account
  preserved in full. After both scoped retirements, new compact navigation
  pointers keep immutable audit links readable; they do not replace archived
  original hashes or authorize another prune.
