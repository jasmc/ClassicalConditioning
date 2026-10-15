The color range scale in E is much, has a huge white space from the data in E, whereas in H that doesn't happen. I want the color scales in E and H to be aligned to the right, but I also want to fill in this space a bit better. But this is something that we'll leave for later because this will require making the data region in D and E larger. So E's data can approach the other color scale range. There are lots of constraints because I must also not separate too much D from E and all of that. Like, in terms of the frames, I'm not sure that those frames that now are kind of ensured automatically worked well at F, G, H. I know there are some conflicting things. Yeah, we should just point that out. Like F and G and G to H, they are not fulfilling this exactly because there's more space. But then if we shorten it here, then the figure frame is also not matched. So I understand all of that.


/Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/reviews is full of shit and images. clean up from stale items, redundant items, and move the figures to the SSD Joaquim. a lot of what is in reviews might need to be merged together as well.

then, plan a full audit of every folder to do a general clean up of the repo. png and svg and pdf, etc should not be stored in the repo and need be moved to Joaquim SSD. keep only in the repo the frozen analyses and plotting ways that have been frozen. all the other versions, respective figures and possibly code should be moved to Joaquim SSD.



the main figures assemble can be kept in the repo. choose a good format for AI agents to work on them on the repo

plan also to create some rules or instructions to defined where new files should be kept (folders).


-----


at the moment, i think there is no automatic discarding of fish running automatically. however, there are plans for that and there might be code already. there used to be different steps of discarding (after preprocessing and then before making some of the plots or running some of the statistical analysis). lets discuss this and attempt to freeze a final solution.


-----

in figure 1, panels the raw vigor shown together with the scaled vigor for specific example trials of a given fish need to match the indicated trials in one of the single fish heatmaps with arrowheads. do you understand? lets find the best trials to show that.

i might also change the control and 3strace example fish.



Figure 2 needs the full datasets.
the stats need to then be frozen.
all LLM need to be audited and confirmed.


-----








------------------------------------------



20230307_12_trace_black-2_mitfaminusminus,elavl3gff,10uasgcamp6fef05



----------------------------------------------------




/Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/configs/example-run.json add a comment if possible explaining how to use this file and what the different options are. if not possibble to add comment, add that info in /Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/README.md


update README "Six-metric corrected route.": there are only 3 metrics right now. confirm candidate-corrected-runner-v1 as well.

remove -v1, etc, suffices from anywhere in the app; i dont want any reference to versions of this kind in filenames. stop stamping different versions. i only want the last one, which should overwrite the previous ones


make a plan for: no figure should be optional (e.g., "[optional] cohort metric-comparison figures") 

attend to "Optionally intake raw triplets. With run_intake: true". intake must not be optionals and must happen for fish that are newly identified in the analysis pipeline; fish with already lossless parquet are preprocessed whereas those logged as having issues are skipped (does this log already exist??).

figure-candidate-profiles and run_figures must happen simultaneously and always (later i will pick one of the metrics and the code will be adapted and pruned).

add to README where i can  check the experiment-specific parameters.

compare figures in refactored code vs in legacy and ask to justify the differences.


review and make README more structured, organized, and make it look  like a README without mentions to old code and old implementations.

what is /Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/tests? are all those files useful or can we delete them?



use repo SciFigEditor to take the rules and formatting for the figures.



number the figures that will be part of the paper, starting with Figure X. and indicating the panel letter.







add comments explaining all code blocks of the repo so that i can review everything manually more easily



understand everything in Plans, src, docs

Figure 3 / Figure 4 scope — author direction, clarified 2026-10-14:
Figure 3 metric direction, 2026-10-15: keep only the legacy metric
(`legacy_distal_angular_speed`, tail bend angular speed); delete Figure 3
drafts for other metrics and restrict its renderer accordingly.
Figure 3 develops, quantifies and validates the individual-learning
representation / learner definition.
The Control–Learner, Control–Nonlearner and Learner–Nonlearner behavioral
comparisons belong in Figure 4 only. Figure 4 compares these groups within
each experiment (Delay, 3sTrace, 10sTrace), as well as across experiments.
The learner definition and validation remain to be resolved under Plan 04 /
Gate L. Keep insufficient-information fish explicit, and distinguish
same-data descriptive comparisons from evaluation on observations that did
not define learner status.

Author 2026-10-15 — individual fish: plan Figure 3 illustrations using the
legacy classifiers, discarding rules and old per-fish heatmap/summary-grid
outputs. Deliberately choose a few illustrative fish for the main paper and
show all fish in supplementary data. Connect heatmaps, exact trial outcomes,
classification evidence and eligibility/coverage; do not silently omit
unclassified or excluded fish. Detailed proposed display and source review:
docs/paper/figures/figure-03/individual-fish-display.md.






---

Author 2026-10-15 — invisible frames: implement shared rules/tooling for future
paper, supplementary and analysis figures, without rearranging current drafts.
One millimetre-based inset is shared at the figure outside edge; adjacent panels
supply two insets. Preserve natural shapes and scientific geometry. Calibration
compared 1.0/1.5/2.0 mm at 89/183 mm; author selected 1.0 mm (tighter) on
2026-10-15. The shared policy is active for future work; current drafts stay unchanged.
Implementation and remaining route coverage are recorded in
records/figure-reviews/layout-frame-implementation.json; procedures live in the
figure workflow's invisible-frame section. No freeze or cleanup is requested.
