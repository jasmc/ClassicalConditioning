# Scripts-only reassessment — 2026-10-11

52 scripts were reassessed. One unused exact duplicate was removed after checking its retained canonical bytes; 48 supported scripts and their dependency closure remain. The three historical scripts were transferred to JOAQUIM, independently verified, and removed locally on 2026-10-11. No visual reviews were changed.

## Completed safely

`scripts/prepare_figure2_G_logmedian_freeze.py` exactly matched the frozen Figure 2 G `code/renderer_source.py`. No active package/test/script import or execution consumer was found. The immutable source still exists byte-for-byte; the repository relocation map directs the old script path to it. No unique bytes were deleted and no freeze was changed. The block-log-median duplicate remains because supported D/E scripts import it.

## Verified archived historical scripts

- `prepare_mac_ssd_handoff.py`: fixed October 9 handoff producer.
- `audit_mac_ssd_handoff.py`: fixed audit of that dated handoff.
- `archive_delay_review_versions.py`: fixed dated Delay all-versions consolidation recipe; its completed artifact already exists in the archive catalogue.

All three external copies retain their original SHA-256; local copies were removed only after verification. See [completed transfer](completed-transfer.json). Their [checkpoint](transfer-inventory.json) records hashes, consumers and dispositions. The completed transfer used `scripts/repository_archive.py` with `--manifest docs/maintenance/archive/scripts-reassessment-20261011/transfer-inventory.json`, then transfer, verify and prune in separate steps. Do not regenerate historical reviews or remove these unique sources before verification.

Additional scientific renderers/audits are retained because they serve current extraction, analysis or supported dependencies. Similar purpose or unreferenced filenames alone do not justify deletion or merging differing formulas. Scientific/code consolidation would need its own behavioral review; this reassessment does not change calculations.
