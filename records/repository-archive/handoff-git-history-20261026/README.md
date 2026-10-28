# Mac handoff Git-history preservation — 26 October 2026

The source/history audit found 1,860 Git objects, including 16 commits, in the
9 October Mac handoff that are absent from the maintained repository. They
include old stashes, untracked-file snapshots and historical development states.
The [native-object backup](</Volumes/JOAQUIM/ClassicalConditioning-Archive/handoff-history/2026-10-26-mac-git-recovery/>)
preserves all 5,172 readable objects from that repository independently.

The [transfer inventory](transfer-inventory.json) authenticates eight payload
files; [completion](completed-transfer.json) records preservation only. Native
Git pack verification, a separate bare-repository reconstruction, full object
ID/type/size comparison and `git fsck --full` passed. The source packets and
required G exports remain unchanged, and no code was merged into active work.
The pack stores Git object identities; its serialization differs from the old
`.git` directory, which remains intact in the handoff.

The archive's `evidence/` companions are authenticated by the
[source-review record](../../repository-history/2026-10-26-storage-disposition-review/handoff-source-review.json).
They include the full file/object comparisons, PC integrity check, unshared
commit inventory and an exact ZIP of 82 Git control/miscellaneous files. This
preserves the existing Mac index, refs, configuration, logs and temporary-object
bytes separately from the native object pack. Three generated packaging files
also have exact archived copies and recovery-index entries.

For recovery, use a new bare repository. Put the archived pack and index into
its `objects/pack/`, restore refs from `original-refs.txt`, and use
`history-provenance.json` for the additional recovery refs identifying the 16
otherwise unshared commits. Verify the object inventory and fsck result before
using any recovered source. Scientific and presentation authority remain with
the maintained repository's scoped records.

The published transfer manifest records the temporary candidate source directory
used to build this preservation copy. Those task-created candidates were cleaned
after verification; the original handoff was never pruned. Supported archive
verification reads the retained destination payload.
