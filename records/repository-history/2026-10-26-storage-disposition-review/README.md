# JOAQUIM disposition review — 26 October 2026

Author requested using the drive's `START-HERE.html` to understand possible
deletion, consolidation and archival. This is an assessment, not an approved
disposal inventory or a new scientific/presentation selection.

The full editable [assessment manuscript](</Volumes/JOAQUIM/ClassicalConditioning Outputs/studies/repository/storage-dispositions-2026-10-26/study.md>)
and [HTML reader](</Volumes/JOAQUIM/ClassicalConditioning Outputs/studies/repository/storage-dispositions-2026-10-26/study.html>)
live together on JOAQUIM. [Candidate evidence](</Volumes/JOAQUIM/ClassicalConditioning Outputs/studies/repository/storage-dispositions-2026-10-26/candidates.json>)
records exact scopes, sizes, recommendations and conditions; the accompanying
inventory, duplicate groups, targeted rechecks and provenance retain the evidence.

- Fresh metadata inventory: 24,150 files outside four OS-managed directories;
  no scan errors. No comprehensive raw/processed hash audit was attempted.
- External review-library check: 63 entries across six groups passed.
- Fresh targeted checks: six selected/input hashes, eight quarantine files and
  three raw-copy/provenance comparisons. Five loose root SVGs have exact named
  Figure 1 source counterparts. The root PNGs remain distinct variants.
- Size-limited duplicate screening found 1,639 groups. Arithmetic duplicate
  bytes are not an estimate of safe deletion: manifests, retained source
  snapshots, frozen recovery and notebook origins can require separate copies.
- Static reader/provenance checks passed: 46 hash records, no runtime resources
  outside the HTML, no missing local target paths. Browser visual/print review
  was unavailable because browser URL policy rejected `file://` navigation.
- All 23 protected existing index, registry, selection and state files remained
  byte-identical. No existing payload was moved, deleted, merged or redrawn;
  notebook storage and scientific analyses were not changed.

## Published identities

| Artifact | SHA-256 |
|---|---|
| `study.md` | `87a43b306bc4d94f306e2e6d57d0be7aadbabcb62d484cbc2dcf96748024986e` |
| `study.html` | `f2b5c7e278db2af9ae78c76a005d6dd2e2e68d1abf036277490ea28126154496` |
| `candidates.json` | `a8032bcf9274a739764802f87126209fb2192faa08e9f9fd387053db3eb8fcb1` |
| `guide-provenance.json` | `0b76ed40b7197d2e0dd696529df2d2777371cd9a8ebc0571dfb61b364326adc9` |

Keep detailed discussion and revisions in the external topic workspace. Before
any eventual action, resolve its exact dependency closure, applicable author
authorization, retained identities and supported storage/disposition procedure.
This review does not reopen completed general repository/script cleanup.

## Subsequent authorized duplicate removal

After the assessment, the author explicitly requested another equality check
and removal of an extra copy. The [completed disposition](raw-duplicate-disposition.json)
records fresh direct byte comparisons and SHA-256 checks for both reviewed raw
file pairs. Only their two `(2)` copies were removed (372,479,605 logical bytes);
both retained unsuffixed originals were rehashed and unchanged after removal.
The external assessment remains the preserved pre-disposition review.

## Font-cache dependency check

The author next requested the cache dependency check. The [assessment](font-cache-assessment.json)
records all 69 exact cache paths and hashes, with 57 recommended removal
candidates (33.338 MiB) and 12 retained files (7.128 MiB) bound by historical
Figure 2 build manifests. Six manifest copies resolve to three version roots;
their declared cache bytes still match. The scan inspected 11,420 text/source
files with no read errors; compressed ZIP contents and Git objects were outside
scope. No local open handles or rendering processes were found for these caches.

An isolated temporary Fontconfig run successfully generated a new cache from
copies of the actual fonts. This verifies regeneration on this Mac, not exact
historical Windows rendering. All 69 existing caches and all 16 checked font
files remained unchanged. No cache files were deleted during this check.

The author subsequently authorized the proposed removal. The
[completed cache disposition](font-cache-disposition.json) records fresh
pre-removal hashes and local open-handle checks, removal of the 57 listed files
(34,957,264 logical bytes), and successful post-removal hash verification of 56
protected files, including all 12 manifest-bound caches and all 16 checked font
files. Configuration files, manifests, current authority and cache directories
were retained. The preceding assessment remains a pre-disposition record.

## Interrupted-intake quarantine recheck

The [quarantine assessment](intake-quarantine-assessment.json) records fresh
hash authentication of all eight quarantined files, all three complete published
intake files and all three original raw inputs for `20221115_04`. All 14 match
their recorded identities. The quarantined tracking file is an exact byte prefix
of the complete published tracking file and lacks the required Parquet footer.
Both camera files contain the same 8,361,388 rows and matching column values;
their source-path metadata differs. Five downstream markers report complete.
No local handles remained after closing the inspection's Parquet readers.

The eight staging files total 1,723,733,906 bytes (1.605 GiB). Disposal is
recommended with complete data, raw originals and transfer records retained;
this recheck executed no deletion and did not change quarantine-only retention.

The author subsequently authorized disposal. The
[completed disposition](../../repository-archive/intake-quarantine-20261024/disposition.json)
records removal of only those eight staging files (1,723,733,906 logical bytes),
fresh pre-removal authentication of the six retained raw/complete intake files,
and unchanged retained file identities afterward. Original transfer records
remain byte-identical. Exact prior recovery-index/policy bytes are preserved in
`quarantine-disposition-originals/`; the SSD disposition folder preserves prior
drive navigation. Eight retired recovery aliases were removed from the live
index and the completed disposition was bound into the existing audit policy.
The drive map now reports the removal rather than claiming staging recovery.
The supported external structure audit passed with no issues or unavailable
evidence: 365 protected files, seven current states and 13 transfer transactions
checked. The targeted quarantine audit recognized all eight exact authorized
absences. Strict transfer verification and pruning were not changed.

## Redundant root SVG removal

The author authorized removal of the five checked root SVG copies without
another approval checkpoint. The [completed disposition](root-svg-disposition.json)
records fresh direct byte comparisons, SHA-256 checks and removal of only
`Asset 5.svg`, `Asset 6.svg`, `Asset 7.svg`, `Asset 10.svg` and `Asset 11.svg`
from the SSD root (155,991 logical bytes). The five descriptively named source
SVGs and all five root PNG previews were retained and rechecked unchanged.
Current presentation state remained unchanged.

The recovery index and relocation map resolve both the old `/Volumes/JOAQUIM/`
and `J:/` references to the exact retained sources, tested with and without
explicit hashes. The distinct historical desktop `Asset 7.svg` for Figure 2
was not redirected. Prior index bytes are preserved in
`root-svg-disposition-originals/`.

## Matching historical Figure 1 PNG/SVG names

The author requested coherent figure/panel names and clarified that the five
artwork pairs are different historical versions of Figure 1's opening panels.
The [naming record](historical-figure1-png-renaming.json) records exclusive copy,
byte verification and removal of the five generic root PNG names. Each PNG now
uses the same descriptive base name as its unchanged SVG and lives beside it
in the existing SSD `figure1-assembly/schemes/` folder. All five historical
variants remain distinct; artwork bytes and current selected state are unchanged.
Ten old root/J: PNG references resolve to their exact retained bytes, checked
both with and without explicit hashes. Prior index bytes are preserved in
`historical-png-renaming-originals/`.

## Figure 2 G guides and alternatives: retain for manual review

The author explicitly confirmed that both explanatory-guide revisions and
scientific/model versions remain pending manual review. The
[bundle inventory](figure2-g-guide-bundle-review.json) records five complete
guide bundles (four earlier revisions totaling 274.045 MiB, plus the current
guide), unique text/code/metadata, repeated artwork and four distinct notebook
keys. All 728 declared artifacts and 618 copied source files matched their
recorded hashes. The five bundles and identified alternative review pages were
rehashed after inspection and remained unchanged.

The pre-completion guide's original 13 chapter sections are byte-identical in
the current guide, which adds chapter 14 and retains its notebook key. Earlier
previews preserve explanation and/or reader-code changes; preview 02 and
preview 03 share manuscript text but have distinct builder/notebook snapshots.
All 20 bound figure assets from each earlier guide match the current guide.
None of these comparisons establishes scientific equivalence among G methods.

The existing [G navigation](../../../docs/paper/figures/figure-02/panel-g/README.md)
now distinguishes those guide revisions from the four exploratory presentation
pages, two independent analysis folders and historical Delay comparison.
Original G payloads, reading paths, notebook keys, scientific selections and
presentation state remain unchanged. Personal notebook storage and older
browser origins were not inspected. No G content was merged, moved, deleted or
physically archived, and no storage capacity was reclaimed. Retain both scopes
until the author's manual review provides their later dispositions.

## Generated review-library snapshots

The author authorized checking the five older generated library snapshots and
removing only unused reproducible copies. The
[completed disposition](review-library-snapshot-disposition.json) records
retirement of `e60a23f244cd959b`: 68 files, 1,002,651 logical bytes. All reading
content exactly matched retained current snapshot `e3bf3afb2fad2b0b`. Three old
navigation/control files (`catalogue.json`, `README.md`, `manifest.json`) are
preserved losslessly as compressed metadata in the reconstruction recipe.
Reconstruction of all 68 original files into a temporary directory passed exact
SHA-256/size checks before removal; every reconstruction input was verified again
after removal. The recipe's current-snapshot source files remain dependencies.

The other four old snapshots remain. `41f93a0882fe7f38` is referenced by a
historical publication record; `ec59f6af7a70c7bd` is still linked from canonical
G/H/I and Figure 4 study navigation and a migration verification record. Older
snapshots also preserve earlier G/H/I source text, and `72e807204380c830`
retains a differing historical Figure 4 source snapshot. No automatic retirement
was inferred for them. Current publication pointers, retained snapshot files,
original studies and every pending G guide/scientific variant remain unchanged.

## Dated handoff source/history comparison

The [source-review record](handoff-source-review.json) reconciles all 848 PC
working files and all 48 differences flagged in the prior audit to exact blobs
in reachable maintained Git history, including the five modified and four
untracked files captured in that packet. All 6,786 ZIP members matched their
per-file hashes. Its 8,506 readable Git objects are present in maintained Git,
and an isolated integrity check passed.

The Mac working tree has 1,351 files: 600 exact historical Git blobs, 737
CRLF-to-LF-only matches, eight exact registered external recoveries, three exact
current generated files and three otherwise unmatched generated packaging files.
Those three generated files now have exact verified archive copies. Git-object
inspection additionally found 1,860 objects, including 16 commits, absent from
maintained Git. An independently verified
[native history archive](../../repository-archive/handoff-git-history-20261026/README.md)
preserves all 5,172 Mac Git objects plus original refs and generated metadata.
Its companion control ZIP preserves 82 operational/miscellaneous Git files.

Every original Mac working/Git file and the PC transport controls was rehashed
after preservation and remained unchanged. The Mac index already differed from
the 9 October transfer checksum before this audit; its exact present bytes were
preserved, and no cause was inferred. Two absent Figure 3 files match the prior
15 October author-authorized disposition. All three required G freeze/SVG/
selection exports matched their recorded hashes.

Both handoff packets remain intact; no packet file was retired and no code was
merged. The Mac snapshot carries unique history. The PC code snapshot is a
possible later compact-replacement candidate, but its checksum-bound portable
ZIP remains until a replacement is prepared and verified. Source recoverability
does not itself demonstrate that every historical scientific idea was adopted.

Targeted archive verification passed for all eight payload files and all three
new recovery aliases. The broader external check verified 14 archive transactions
and found no unavailable external evidence; its sole reported issue was the
newly appeared `scripts/maintenance/storage_migration.py` lacking a catalogue
entry. That concurrent work was left untouched. The exact final validation
result is recorded in the source-review JSON.
