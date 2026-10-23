# Study source migration, 23 October 2026

Implemented [Plan 12](../../../Plans/12_STUDY_ARTIFACT_STORAGE.md). Eighteen full
study/discussion sources now live on JOAQUIM; short repository pointers retain
navigation and necessary method boundaries. The [source registry](../../../docs/paper/study-sources.json)
records current editable sources. [Dispositions](dispositions.json) list source
roles, consumers and immutable historical exceptions.

The native [transfer inventory](transfer-inventory.json) and
[prune receipt](completed-transfer.json) preserve exact originals at the recorded
SSD history destination. Local pointer pages were written only after strict
copy/hash verification and pruning. Hash-specific recovery aliases refer to
those originals; they do not select a newer scientific input.

Four supported readers resolve external sources. The review catalogue now has
57 entries and HTML reading snapshots for external Markdown; the long reading
inventory stays on the SSD. Six topic indexes group Figure 2 G/H/I, Figure 3,
Figure 4 and shared material. Figure 1's agenda is retained in that shared
catalogue; no paper artwork was changed.

[Verification](verification.json): 62 relevant tests passed; 18 canonical and
original source pairs verified; all 567 previously inventoried review/source
files retain their bytes. Thirty-two generated HTML pages and 763 links were
checked statically. All 365 local preserved requirements and seven scoped states
pass. Ten archive transactions verify. Four pre-existing Figure 1/2 external
working-copy mismatches remain, so the overall external structure check reports
failure for those four files. No hashes were changed to conceal them.

Existing HTML routes, notebook keys, embedded sources and user notes remain
untouched. No fits or figure renders were run. Browser-rendered usability and
print behavior were not checked: prior local-file browser access was rejected,
and no workaround was attempted. GitHub Actions account billing remains an
independent previously recorded blocker.
