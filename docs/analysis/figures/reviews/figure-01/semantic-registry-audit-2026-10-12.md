# Figure 1 semantic registry audit — 12 October 2026

The registry is useful but its coverage must be stated by scope. The frozen
editable assembly has 276 scientific records for E/F/G/H; legacy A–D retain
structural handles and source provenance. The preceding presentation draft had
no scientific records. The explicit freeze validator uses roles, mappings,
protection and review evidence, but ordinary draft generation did not use a
scientific registry to enforce shared-panel relationships.

The current D/E draft adds 563 explicitly classified records, including source
frame/time/value hashes, roles, events, numeric bin values, tick values, row
labels, axis rectangles and colourbar-to-mappable links. They are embedded in
the editable SVG and consolidated HTML provenance. The shared renderer assigns
roles from scientific inputs, applies role styles from figure-elements.json,
records the requested legacy typography and retained V12 palette, and checks
D/E identity/layout relationships. Exported clip rectangles, row labels and
bin colours are checked against the map. Missing imported registry IDs or
mappable links now fail instead of silently disappearing.

## Scientific result and layout evidence

Both previously selected panels already identified Delay fish 20221115_07 and
trials 9, 17, 63, 66 and 93. Their processing was inconsistent: E used cadence
reconstruction and different raw vigor values. The maximum selected-frame raw
value difference was 3.5898914 rad/ms; timing differed by up to 0.0202903 s.

At the author's request, draft E uses D's stored Vigor and measured CS-relative
time. The previous E bout mask is joined by trial and FrameID (one-to-one),
with 6 unmatched boundary frames excluded and reported. Raw and coloured data
use the same eligible positive finite numeric inputs. The established log,
0.5 s median-bin, [−15,0) baseline and percentile scaling recipe is retained;
colours are recomputed. All five rows meet the baseline-bin requirement. The
raw visual cap excludes no numerical values (197 eligible values exceed 1
rad/ms); 15 bins exceed colour limits without numerical clipping. D's frozen
angle values and per-trial centring remain intact. The legend discloses these
changes; draft E is not presented as the exact frozen V12 row.

D/E share 450×60-unit data-axis rectangles, row centres/order/labels, events,
x limits and a 750×550-unit body contract excluding E's colourbar. Their
labels place block name above trial number; trial 9 is Pre-Train, matching D's
selected source and the protocol's pre-training phase. Both vertical quantity
labels and common below-axis CS-relative time labels were visually reviewed.
An independent read-only review found no material clipping/overlap and no
mismatch between exported geometry and registry. Final-scale measurements
report zero tick/spine findings. Figure 2 is carried forward byte-for-byte
from its authenticated review; historical frozen artifacts are unchanged.

## How the agent uses the map

1. Read AGENTS.md, scoped current state and authenticated selections; do not
   select an artifact merely because its name looks recent.
2. Locate an element by its stable SVG ID. Read its explicit role, source/data
   mapping, units, coordinate system, processing, relationships and evidence.
   Structural IDs only locate XML; they do not establish scientific meaning.
3. Edit the supported renderer using that meaning. Apply role styles at final
   scale and preserve protected scientific inputs. Related panels share an
   explicit identity/layout contract; raw/colour representations share inputs.
4. Check both the registry relationships and the actual serialized export.
   Review legibility, stacking, clipping and scientific meaning; update the
   affected legend only when its definitions/explanations change.
5. On an explicit freeze request, prepare a complete candidate and use the
   freeze gate. Then inventory prior versions for the author's keep/archive/
   delete choices. Preserve them pending those decisions.

This is persistent file-based context, not automatic model learning. Future
chats must read the map and instructions. Programmatic checks provide concrete
failure conditions; an agent's memory or a pointer alone does not prove use.

## Remaining limits

This is a reviewed D/E draft map, not a complete freeze-record contract or a
whole-figure registry. Other draft panels still lack imported scientific
records. Freeze-specific scientific/structure/visual evidence and protection
records need completion; existing nonuniform F–H display transforms need
resolution before a new freeze. The normal renderer does not invoke the
freeze gate. SVG measurements cannot independently prove every analytical
choice; authenticated numeric inputs and actual scientific review remain
necessary. G's arrows still mark the 3 s Trace example and have not been
reassigned to D/E's Delay trials.

Exact current artifact paths/hashes remain solely in current-state.json.
