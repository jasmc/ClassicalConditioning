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





