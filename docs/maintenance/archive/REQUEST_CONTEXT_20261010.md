

/Plan mode

see what is at /Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/Plans. anything that is not a real plan, move or plan to delete if stale. decisions, handoffs, freezing of panels or figures or any other decisions are not to be here. e.g., /Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/Plans/EXCLUSION_AND_SELECTION_INVENTORY.csv should not be here.
except for README or Index or sth like that, all real true plans must be renamed and given a number, as most of the current plans. order them as it makes sense. 
these files /Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/Plans/PANEL_REVIEW_COMMENTS.md and
/Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/Plans/PAPER_REVIEW_CONSIDERATIONS.md might need to be updated and contain placeholders or so for all other figures. i think /Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/Plans/GOVERNANCE.md is also misplaced.
plan this very carefully. if needed, and in case it makes sense to keep them, move them to other folders (create new ones if needed)

the files defining the frozen figures, etc need to be in an appropriate folder as well.

the plans that are to be kept (even if they need to be first updated) have to be coordinated with the remaining plans and their structure, so there is a clean story at Plans folder. remember that the first release is going to be the first 4 figures.

see what can be removed from /Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/Handovers, etc. perhaps, some things there can be integrated in decisions or freezing files, to be kept in other folders.




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





