# Figure 2 G: analysis guide and review notebook

Maintained explanatory source for the versioned HTML guide. Numerical tables and
artwork placeholders are resolved from authenticated saved outputs by
`scripts/build_figure2_g_analysis_guide.py`. Editing this source does not select
an analysis, change a cohort, or freeze a figure.

## 01. How to use this guide

Your task is to choose the final scientific analysis and the figure that communicates it. This guide separates the choices so you can inspect the evidence, understand what each method estimates, and explain your decision without asking AI to interpret a plot for you.

Read chapters 2–4 first. They establish the experiment, the basic statistical ideas and the exact contents of selected G. Then compare the outcomes, independent analyses, timing results, exclusions, learners and alternative presentations. Chapter 12 is a decision worksheet. The glossary and source inventory are reference material.

<div class="callout"><span class="label">The central distinction</span><p>Evidence that the conditions change differently does not automatically identify the acquisition trial. A population effect does not determine how many individual fish learned. A later interval containing zero does not establish extinction.</p></div>

### What “current” means here

**Scientific selection:** the scoped October 9 freeze, “Native Historical LogMedian difference, phase-aware LMM.” Its source outcomes and statistics remain the selected scientific G. The freeze itself records heavy tails, an unavailable Test3 local fit, exploratory status and no learning-onset claim.

**Presentation selection:** the saved Figure 2 working state at this guide’s build. This assembly has received subsequent layout and typography changes. Its panel g is shown separately from the original frozen artwork. Those presentation changes do not constitute a new scientific freeze.

**Exploratory alternatives:** independent analyses and separate figures preserved for comparison. A version can be implemented and numerically checked without being statistically validated or selected for the paper. “Fixed” in the discussion means specific problems were repaired; it does not mean every inferential issue was resolved.

This guide is a dated reconstruction from saved evidence, source code and the recorded requests. It is not a verbatim transcript. The later October 15 individual-fish comparison is included as relevant context and labeled accordingly. No new biological analysis is run to fill gaps in the historical record.

| Status | What it establishes | What it does not establish |
| --- | --- | --- |
| Proposed | A question, method or threshold was discussed | That it was executed |
| Implemented | Code and outputs exist | That inference is calibrated |
| Numerically checked | Specified calculations, identities or fit gates passed | That assumptions describe the experiment |
| Statistically validated | Error rates/coverage pass stated criteria in relevant situations | Universal correctness outside those situations |
| Selected | The author chose a scoped version | Approval of every alternative or missing scope |
| Frozen | A particular scoped artifact and its dependencies were recorded | A whole-figure freeze or a new scientific claim |

### Notes that remain under your control

Every teaching chapter has a review notebook. Write your understanding, questions and provisional choices there. The browser reports whether notes are saved locally or only held in the current tab. **Export notes** gives you a JSON backup; **Import notes** previews the version/evidence match and requires a deliberate replacement action. Notes never modify model settings, fish membership, figures or freeze records.

The notebook belongs to this guide version and browser origin. A different localhost port, browser profile or device may not see the same saved notes. Keep exported backups. The guide and its assets live on JOAQUIM, as requested; the drive must remain available. External papers require an internet connection, but the saved explanations and local outputs do not.

{{sources:selection,figure-state,exploratory-state,review02}}

## 02. The questions and the experiment

### Two questions, two kinds of evidence

**Condition difference in change:** after allowing for the Pre reference, does the Delay response change differently from control during training or testing? This is a comparison of changes, not merely a test that Delay fish change over time. Control fish can also drift, habituate or change activity.

**Timing of acquisition:** when does a defined conditioned-direction effect reach a meaningful magnitude and persist? This requires an operational definition of “acquired,” a time scale and uncertainty for the timing. A first small p-value is not that definition.

The population analysis concerns the condition trajectories. An individual-fish acquisition time or a count of learners is an additional question with different data requirements and validation needs.

### Schedule used by G

<div class="stage-map"><div><b>Pre: CS5–14</b>10 scheduled trials<br>Reference period before training</div><div><b>Training: CS15–64</b>50 scheduled trials<br>Five 10-trial training blocks</div><div><b>Testing: CS65–94</b>30 scheduled trials<br>Three 10-trial testing blocks</div></div>

G uses acquisition CS ordinals **5–94**, inclusive: 90 scheduled trials per fish. The first four acquisition ordinals are not part of this panel. Avoid confusing these acquisition ordinals with other displays that may renumber their first displayed trial.

| Block in G | CS ordinals | Role |
| --- | --- | --- |
| Pre | 5–14 | Reference |
| Train1–5 | 15–24, 25–34, 35–44, 45–54, 55–64 | Training progression |
| Test1–3 | 65–74, 75–84, 85–94 | Testing progression |

There are **28 control and 29 Delay fish**. The 57 × 90 schedule contains **5,130 fish-trial rows**. The paired LogMedian outcome is available in **4,811**; **319** remain unavailable. Fish remain cohort members even when they contribute no outcome to a particular trial or phase summary.

For each CS, the baseline window is **[−15, 0) seconds** and the response window is **[0, 9) seconds**, relative to CS onset. The closing parenthesis means the endpoint is excluded. Historical discard code uses different inclusive endpoint conventions; chapter 9 keeps those separate.

CS is the conditioned stimulus; US is the reinforcer. A response window is defined relative to the CS, not by simply renaming the US response. Actual event overlap was examined as a sensitivity. Protocol labels alone do not guarantee every recorded event has the intended timing.

### Why fish are the independent units

Frames within a trial share the same fish and behavior; trials within a fish share its physiology and history. They are repeated observations, not thousands of independent animals. The model therefore groups observations by fish, and population bootstraps resample entire fish trajectories.

A mixed model can use different numbers of available trials per fish. It does not require throwing away a fish because one trial is missing. It still needs a defensible outcome and assumptions about availability and dependence. “LMM handles missing data” does not mean it repairs behavior-dependent selection or creates intensity measurements for windows without bouts.

**Review question:** if you describe G as “movement suppression,” do you mean the intensity of movement when a valid bout is detected, the probability of moving, or total activity across the whole window? These are different scientific outcomes.

{{sources:selection,analysis-input-import,analysis-input-verification,plan03,plan06}}

## 03. Statistical basics with worked examples

### Outcome, estimand, estimate and uncertainty

An **outcome** is the measured quantity for a fish and trial. An **estimand** is the population quantity the analysis is intended to describe. An **estimate** is the number calculated from this sample. Its **uncertainty interval** describes the precision of that estimation procedure under its assumptions.

For example, “response-minus-baseline LogMedian on this fish’s trial” is an outcome. “The baseline-adjusted control–Delay difference in change averaged over CS55–64” is an estimand. The saved adjusted estimate **0.0947** is an estimate; **[0.0117, 0.1778]** is its exploratory joint interval. Neither number is itself a learner count or an acquisition date.

### A contrast of changes

Use the guide’s population sign convention:

<div class="formula">Δ(t) = [control(t) − Delay(t)]<br>　　　 − average of [control(s) − Delay(s)] over Pre CS5–14.</div>

Positive Δ means the Delay response is lower, relative to control and the Pre difference: the conditioned suppression direction. It removes a stable between-condition Pre offset. It does not by itself establish the cause of a change; biological interpretation still depends on the design and measurement.

**Toy example, not experimental data:** suppose both conditions have a Pre value −0.02. Later control is −0.03 and Delay is −0.13. The later control–Delay gap is 0.10 and the Pre gap is 0, so Δ = 0.10. If both groups instead fell equally to −0.13, their within-group changes would be large but Δ would be zero.

G’s categorical D coefficients use the opposite, **Delay−control**, sign. A negative D and positive Δ can express the same suppression direction. Always read the sign definition before comparing tables.

### What logarithms change

Natural logarithms turn a ratio into a difference: ln(R/B) = ln(R) − ln(B). A contrast of 0.10 log units corresponds to exp(−0.10) ≈ 0.905 in the conditioned-direction ratio comparison, or about 9.5% lower on that ratio scale.

This is an interpretation of the particular log-scale estimand. It does not mean every fish moved 9.5% less, nor that whole-window total activity fell by that amount. G measures conditional bout intensity and has several aggregation stages.

**Toy frame example:** response values [1, 1, 10], baseline [1, 1, 1]. The response arithmetic mean is 4 and baseline mean is 1, so ln(mean response / mean baseline) = ln(4) ≈ 1.386. The response median of logs is 0 and the baseline median of logs is 0, so their difference is 0. Logging an arithmetic mean does not remove the burst’s prior influence on that mean.

Even for medians, the sample convention matters. With two values [1, 9], median(ln values) = ln(3), whereas ln(median values) = ln(5). Averaging the two middle observations need not commute with a nonlinear transform.

### Mixed models, in familiar terms

A **linear mixed-effects model**, abbreviated LME or LMM, combines a population relationship with fish-specific variation. “Linear” refers to how coefficients enter the model; a cubic spline curve can still be fitted in an LMM.

- **Fixed effects:** the population condition, phase and time relationships, plus the baseline covariate where included.
- **Random intercept:** a fish may tend to have consistently higher or lower outcomes.
- **Random slope:** a fish may depart from the population time trend in a systematic way.
- **Residual:** what remains unexplained for a trial after the model’s components.
- **Spline:** a restricted set of smooth basis functions describing the time curve. It is more flexible than one straight line, but can still miss abrupt or brief changes.

Fish random effects account for part of the dependence among repeated trials. They do not automatically make residuals independent, remove heavy tails, correct protocol corruption or make every fish equally influential. A model fitted to all available rows also differs from simply giving every fish one equal-weight phase summary.

**LMM is not LLM.** The saved inference is a numerical mixed-model fit. AI assisted code development and discussion; it is not a term in the likelihood and does not supply the trial p-values. Source files, model formulas, coefficients and numerical artifacts are available for review without AI.

<details><summary>Baseline subtraction versus baseline adjustment</summary>

Let Lᴿ be the response-window LogMedian and Lᴮ the baseline-window LogMedian. A direct-change model uses z = Lᴿ − Lᴮ as its outcome. Algebraically, that fixes the baseline coefficient at 1 if rewritten as a response model. A baseline-adjusted response model instead estimates the coefficient βᴮ in Lᴿ = βᴮ Lᴮ + other terms + error.

These approaches can disagree when baseline varies, is measured with error, or relates differently to the response than a unit coefficient assumes. Adjustment is not intrinsically the more correct answer; the baseline’s scientific role matters. In the saved additive adjustment, population contrasts use a common baseline value, and its main-effect contribution cancels between conditions. Baseline still affects the fitted coefficients and covariance through estimation.

The displayed G lines use z. Its inferential model uses Lᴿ adjusted for Lᴮ. That distinction must remain explicit.

</details>

### Confidence bands, IQR and p-values

A **pointwise 95% interval** targets one specified quantity at a time. Inspecting 90 such intervals does not turn them into a joint 95% band for the whole trajectory. A **simultaneous band** is constructed to cover an entire declared family together, if its procedure is calibrated. It is usually wider.

An **IQR** is the middle 50% of observed fish values. It describes variation among fish, not estimation uncertainty. G’s shaded bands are confidence intervals, not IQRs.

A frequentist 95% confidence procedure is intended to cover the fixed target in 95% of repeated experiments under its assumptions. It is not a 95% probability that this particular fish learned. A p-value describes incompatibility with a specified null model; it is not the probability that the null hypothesis is true or the chance a fish is a learner.

<details><summary>Why multiplicity matters, and why Holm and BH differ</summary>

With 90 independent null tests at 0.05, the chance of at least one false rejection would be 1 − 0.95⁹⁰ ≈ 99%. Trial tests here are correlated, so that toy number is not their actual error rate. It illustrates why a test family must be declared.

Holm adjustment aims to control the chance of any false rejection within its family, provided the individual p-values are valid. Benjamini–Hochberg (BH) targets the expected false-discovery proportion under its applicable assumptions. BH across trials is not the same as a simultaneous confidence band, and BH correction cannot repair misspecified model p-values.

The new max-statistic bootstrap instead takes, for each accepted fish-resampled model fit, the largest absolute standardized deviation over the declared 90 trials plus three phase averages. Its 95th percentile is the common critical value used with the original standard errors. The declared target family matters as much as the bootstrap count.

</details>

### Numerical checks and scientific validation

**Converged** means the optimizer reported finding a solution. **Singular** means a variance/covariance component effectively collapsed under the chosen gate. A positive covariance, full-rank design and an appropriate Hessian check help assess numerical behavior. They do not establish that the model captures the true trajectory.

**Calibration** asks whether a complete procedure actually has the claimed coverage or error rate when the truth is known in relevant simulations. The saved simulations show why this distinction matters: a smooth model can optimize successfully yet miss an abrupt truth with an apparently precise interval.

<details class="quiz"><summary>Check your understanding</summary>

**Question:** the mean estimate is unchanged but a wider joint interval now includes zero. Has the observed effect disappeared?

**Answer:** no. The point estimate is unchanged; the inferential statement changed because the uncertainty procedure or family changed. You must state both the magnitude and the remaining uncertainty.

**Question:** the first significant trial is 19. Does that imply acquisition at 19?

**Answer:** no. It is the first rejection for a particular null, model and correction. A meaningful timing claim needs its own magnitude, persistence and uncertainty definition.

</details>

{{sources:selected-candidate-specification,analysis-difference-fit-diagnostics,analysis-adjusted_response-fit-diagnostics,plan06}}

## 04. Exactly what selected Figure 2 G shows

{{figure:frozen-g}}

### Follow the calculation from frames to the line

The selected metric is legacy distal angular speed. Eligible samples are **valid, adjacent moving/bout frames** with finite positive vigor. Adjacency requires FrameStep = 1; the selected corrected bout policy excludes non-bout samples. Positive filtering is needed before taking logarithms. “Valid” here is the recorded detector/metric eligibility, not a blanket claim that every aspect of raw tracking was perfect.

For fish i, trial t:

<div class="formula">Lᴿᵢₜ = median of ln(vigor) in eligible response-bout frames, [0,9) s<br>Lᴮᵢₜ = median of ln(vigor) in eligible baseline-bout frames, [−15,0) s<br>zᵢₜ = Lᴿᵢₜ − Lᴮᵢₜ<br>Displayed line for condition c at t = median of available zᵢₜ across fish in c.</div>

Blue is control; magenta is Delay. A value below zero means lower typical logged bout intensity in the response window than the baseline window. It does not say how much of the window the fish spent moving. The line is unsmoothed: no new rolling median, downsampling or second log(x+1) is added in this selected reconstruction.

An empty eligible window remains missing. It is not assigned zero before or after logging. The number of contributing fish can vary across trials even though cohort membership remains 28/29.

### What the shading means

The selected descriptive bootstrap uses **5,000 resamples, seed 10**, drawing fish with replacement separately within each condition. Each sampled fish carries its whole trial trajectory, including missing trials. At each trial the median across the sampled available fish is recalculated. The middle 95% of those bootstrap medians supplies the pointwise percentile interval.

These resamples estimate uncertainty in the plotted condition medians. They do **not** refit the statistical LMM, and they are not the source of the D/M/R or black trial stars. Overlap or nonoverlap of these two descriptive bands is not the formal condition-change test.

### The longitudinal model behind the trial stars

The model outcome is **response LogMedian Lᴿ**, with **baseline LogMedian Lᴮ as an estimated covariate**. It is not a direct test of the displayed median z lines.

The fixed mean function has condition interactions with separate phase intercepts, a linear Pre trend, a cubic training spline with four basis features, and a cubic testing spline with four basis features. The phase bases use the known schedule; continuity across phase boundaries is not imposed. Fish have a random intercept and a globally scaled-trial random slope. Estimation is maximum likelihood.

The tested trial quantity is the fitted control−Delay population difference at a common baseline, minus its mean fitted Pre5–14 difference. Population predictions use the fixed effects; they are not predictions for a particular fish and are not medians across fitted individual curves.

The saved trial contrasts, covariance and model input can be inspected in the source inventory. The exact selected model input was recovered from the embedded archive by matching its freeze hash; a differently hashed relocated model-input file was not substituted.

### Read the four statistical rows separately

| Mark | Question | Model / family | Selected result |
| --- | --- | --- | --- |
| D | Is the condition difference in this 10-trial block different from Pre? | Baseline-adjusted categorical block interaction; fish intercept/slope; Holm across 8 non-Pre interactions | Train1–5 and Test1 pass |
| M | Is there a local condition level difference at the block’s centered trial? | Separate local block model; fish intercept; BH across 9 blocks | Train2–5 and Test1 pass |
| R, raw and BH | Do the local condition slopes differ? | Local block slope interaction; fish intercept; 9 scheduled tests in the adjusted family | Train1 and Test1 pass in both rows |
| Black trial star | Is fitted control−Delay change from the average Pre gap nonzero at this trial? | Phase-aware longitudinal model; BH across 90 scheduled trial contrasts | CS19–71 pass |

These are related but distinct hypotheses fitted to the same experiment. They are not four independent confirmations. A local M comparison does not subtract Pre; D does. A local slope can differ without implying a persistent level change.

For block marks, * / ** / *** correspond to p < 0.05 / 0.01 / 0.001 using the declared row’s p-value. For the black trial strip, one star denotes a BH-adjusted rejection at 0.05; the count of stars along the strip is not a p-value tier.

The Test3 local model is singular. Its M/R results are **unavailable**, not evidence of zero effect. The BH adjustments retain all nine scheduled tests, treating unavailable tests as p = 1 during correction instead of shrinking the family. No random-effects fallback was silently inserted for the freeze.

<details><summary>Exact selected D results</summary>

{{table:current-d}}

The saved joint test of all eight block interactions is χ² ≈ 121.977, df = 8, p = 1.2945 × 10⁻²². This model-conditional omnibus result asks whether the interaction coefficients are jointly zero. It does not identify acquisition, estimate a meaningful effect magnitude, or replace the planned phase questions.

</details>

<details><summary>Available saved M/R annotation p-values and the missing-table boundary</summary>

{{table:current-mr}}

The complete original selected MR-local-block-tests.csv has not been recovered in the checked sources. The selected sidecar preserves the displayed entries above. Do not interpret this partial list as the full table of non-rejections, and do not borrow the arithmetic-mean outcome’s M/R file.

</details>

### Diagnostics and the honest status of G

Thirteen of fourteen attempted LogMedian fits passed the recorded numerical gates; Test3 local did not. The phase residual excess kurtosis is about **8.027**, compared with 0 for a Gaussian reference, indicating substantial heavy tails. The recorded 57 phase leave-one-fish-out fits passed their numerical checks. Numerical success and stability to single omissions are useful, but do not calibrate Gaussian/Wald p-values.

The selected freeze already disclaims an acquisition-onset claim. The later independent simulation findings reinforce caution about the shared smooth trajectory family. They do not directly estimate the false-positive rate of every selected D/M/R family, and do not prove every p-value in G false.

**Assessment:** no selected outcome-extraction error or fitted trial-contrast discrepancy was found in the matched comparison. G is a reproducible exploratory display with useful directional evidence and limitations. It is not a completed, validated confirmatory analysis or a precise acquisition estimator.

### The current assembled presentation

{{figure:current-g}}

{{legend:current}}

The current assembly uses smaller, reorganized statistical rows and later typography/layout choices. Some redundant D/M/R prefixes can be omitted from glyphs because the hypothesis rows are labeled separately. Presentation mappings preserve the statistical meaning. The original selected panel above remains the reference for the October 9 scientific content. Neither the whole assembly nor this teaching guide is newly frozen.

{{sources:selection,frozen-definition,frozen-sidecar,frozen-inventory,selected-model-input,selected-phase-contrasts,selected-D-interaction-tests,selected-joint-interaction-test,comparison,current-registry}}

## 05. Outcome and model alternatives before the independent review

The October 9 all-versions review preserves alternatives leading to the selected LogMedian outcome. They are included here because “log,” “median” and “historical” refer to different processing choices, not interchangeable labels.

### What each outcome measures

| Version | Per-fish calculation | What changes / scientific meaning | Saved inference and status |
| --- | --- | --- | --- |
| All-frame arithmetic means | mean(response) / mean(baseline), including non-bout frames | Mixes activity intensity with time spent moving; superseded under the current bout-only policy | Historical references; old J outputs were inaccessible in that audit; no substitute claimed |
| Bout-only arithmetic means | mean(response bout vigor) / mean(baseline bout vigor) | Average bout intensity; large bursts contribute strongly; ratio reference 1 | Own saved LMM analysis, exploratory |
| Log arithmetic-mean ratio | ln(mean response / mean baseline) | Log after arithmetic averaging; reference 0; does not undo burst influence | Own baseline-adjusted log-response global/phase comparisons |
| Literal within-window medians | median(response bout vigor) / median(baseline bout vigor) | Retains finite zeros; changes within-window location summary | Descriptive only; no own LMM stars |
| Historical LogMedian, selected | median(ln positive response vigor) − median(ln positive baseline vigor) | Typical logged positive bout intensity; positivity filter and interpolation matter | Fresh selected phase-aware LMM; exploratory limitations recorded |
| Exponentiated LogMedian | exp(per-fish LogMedian difference) | Ratio-scale representation of that per-fish log difference | Descriptive alternative; not identical to literal median ratios |
| Mean of logged frames | mean(ln positive response vigor) − mean(ln positive baseline vigor) | Geometric-average comparison; same positive-frame inputs as LogMedian | Descriptive only; no fitted inference transferred |

For the corrected matched alternatives, the cohort, metric, CS ordinals and baseline/response windows stay fixed. Changes to the window summary or positivity filtering change the outcome and can change eligibility. The original all-frame comparison has a different sample policy and must not be described as simply another view of the current bout-only estimand.

The selected LogMedian name does **not** restore the entire February/March pipeline. Historical source plotting also involved rolling-median/downsampling steps and a later log(x+1) operation on prepared summaries. The current selected reconstruction logs positive eligible frames once and does not reproduce those extra historical transforms.

In the matched October 9 review, 18,610 baseline and 10,859 response bout-frame values were nonpositive and excluded by LogMedian’s positive filter; the finite-zero-retaining alternatives treat those differently. The largest recorded discrepancy between a literal median ratio and exp(LogMedian difference) was approximately **0.17238**. These versions are not related by a universally exact axis relabeling.

{{archive:9}}

{{archive:14}}

{{archive:10}}

### Why a global spline was challenged

The older arithmetic-mean global model used a single df5 cubic spline over CS5–94, with interior knots at CS35 and CS65. It did not have a separate training-start phase at CS15. Its BH90 trial strip rejected at **CS5–71**, including all Pre trials.

That is mathematically possible because each trial’s fitted condition gap is compared with the *average* fitted Pre gap. The ten centered Pre contrasts average zero but need not individually be zero. A shared smooth trend can differ from that average on both sides. Such stars cannot mean the fish acquired conditioning before training.

The phase-aware arithmetic-mean model uses a linear Pre component and separate df4 cubic Train/Test bases with phase intercepts. Its BH strip becomes **CS19–70**. The selected LogMedian phase model gives **CS19–71**. This improves alignment with the experimental phases but does not establish biological onset or validate the mean function.

| Outcome / trial model | D Holm8 rejections | M BH9 rejections | R raw / BH9 rejections | Trial BH90 range |
| --- | --- | --- | --- | --- |
| Arithmetic means, global | 6 | 4 | 2 / 1 | 5–71 |
| Arithmetic means, phase-aware | 6 | 4 | 2 / 1 | 19–70 |
| LogMedian, phase-aware selected | 6 | 5 | 2 / 2 | 19–71 |

These are saved exploratory comparisons, not independent replications. They use their own outcomes/models. The global curve’s pre-training stars are not part of selected current G.

<details><summary>Model comparison numbers and an earlier numerical diagnostic repair</summary>

For arithmetic means, the saved global versus phase-aware comparison had AIC −6095.30 versus −6099.46 and BIC −5985.17 versus −5911.58. AIC weakly favored the more complex phase model; BIC favored the global model. Neither establishes scientific correctness or valid timing intervals. Their observation-count convention is also not an independent fish-level validation criterion.

Mean training changes were 0.07788 versus 0.07876 log units; mean test changes were 0.02873 versus 0.03042. Residual excess kurtosis was 7.423 versus 7.363 and median genuinely adjacent residual lag1 0.136 versus 0.125. The phase split did not remove the residual problem.

The earlier global-LMM audit identified an error in how the optimizer result was supplied to a Hessian diagnostic. The structured ML parameter form was required; the array form could produce an inappropriate check. Corrected diagnostics and independently recalculated contrasts were saved without erasing the original bundles. Reproducing the dense star range with another optimizer did not validate its scientific interpretation.

For the later LogMedian comparison, 13/14 fits passed and excess kurtosis was 8.027, higher than arithmetic means’ phase value 7.363. LogMedian can be appropriate for typical intensity; these diagnostics do not show it is statistically superior.

</details>

{{archive:1}}

{{archive:5}}

{{archive:2}}

{{archive:6}}

{{archive:7}}

{{archive:8}}

{{archive:15}}

### Confidence interval versus IQR

The saved CI/IQR comparison holds the observed median lines fixed while changing what the shading represents. The IQR is the central half of fish observations; the pointwise bootstrap CI estimates uncertainty in the condition median. Replacing one with the other changes the reading of the plot, not the underlying fish values. Selected G uses the CI.

{{archive:11}}

The complete historical review, including embedded audit text and its recoverable ZIP, is available here: {{link:old-review}}. Its original dates and historical recommendations remain historical; this guide does not silently turn an abandoned recommendation into the current choice.

{{sources:old-review,frozen-definition,selected-candidate-specification,comparison-prose}}

## 06. The independent analysis and its numerical results

The independent work was designed to preserve G and address its two claims separately. Its selected report is in **20261014-analysis-02**, stage **report-7bb6bd520060**. The first run and earlier stages remain preserved; they were not overwritten with repaired results.

The reconstruction matched all **4,811 selected G outcomes**, with maximum absolute difference **4.44 × 10⁻¹⁶**, and authenticated **401 bound input files**. The October 9 recovery described 402 source records in its own inventory; these are different inventory scopes, not evidence of one lost fish or trial.

### Two population targets

**Direct change:** model z = response LogMedian − baseline LogMedian. This is closer in outcome definition to G’s displayed per-fish difference, but the model still estimates a conditional population mean relationship, not the displayed median across fish.

**Adjusted response:** model response LogMedian with baseline LogMedian as a covariate. This uses the same longitudinal mean family and outcome treatment as selected G’s phase model.

Both use condition interactions, separate phase bases and fish random intercept/slope. Neither was selected as the final primary target. Their contrasts use the same control−Delay change from average Pre, positive for suppression.

For the adjusted target, old versus new fitted trial contrasts differ by at most **1.5123 × 10⁻⁸** log units and standard errors by at most **5.5495 × 10⁻⁸**. Therefore the matched new adjusted curve is essentially the same fit. Its different annotations largely reflect the uncertainty/test-family change, rather than a newly discovered trajectory.

### Planned phase results

Late training is CS55–64, early testing CS65–74, late testing CS85–94. These are 10-trial averages of the fitted contrast or the specified comparator. They are not identical hypotheses to the categorical D coefficients or local M tests.

{{table:phases}}

All three methods use the selected 28/29 cohort. Available phase contributors can differ; the unsmoothed late-training summary has 27 contributing controls and 29 Delay fish. Its missing contributor has not been discarded from the cohort.

**What this comparison says:** both LMM targets have positive late-training joint intervals. Early testing is positive under direct subtraction but its adjusted-response interval includes zero. Late testing includes zero for both. The unsmoothed late-training point estimate has a similar direction but a joint interval spanning zero. Thus “late-training direction is consistent” is stronger than “late-training significance is independent of analysis choice.”

### How the new uncertainty is calculated

Main population runs requested **999 whole-fish model refits**, within condition. **989** direct-change and **999** adjusted-response replicates passed. Each target’s max-statistic family contains **90 trial contrasts plus three planned phase averages**. The bootstrap-t critical values are about **3.978** and **4.424**, respectively, rather than a pointwise 1.96 Gaussian multiplier.

The numerical acceptance rule requires at least 95% successful replicates. Failed attempts and retries are retained. Repeating an optimizer with the same model is different from quietly changing random-effects structure after a failure.

These are nominal simultaneous intervals until calibration is adequate. A fish bootstrap preserves the observed within-fish patterns; it does not fix systematic mean-function bias or establish identification under the real availability mechanism.

The independent workflow did **not** refit and calibrate the identical original D/M/R families. Its three planned phase contrasts and trial family are the implemented population comparison. A positive planned late-training interval is not a recalculated p-value for every original G statistical row.

**Why “5,000 versus 999” is misleading:** G’s 5,000 resamples calculate descriptive medians without refitting the LMM. The new 999 replicates refit a model and construct joint uncertainty. Estimator, refitting, standardization and family differ; a smaller resample count is not the explanation for the wider band.

### The simpler unsmoothed comparator

For each fish, subtract its mean available Pre z from each available trial z, then compare the condition means of those available fish changes. This is transparent and avoids imposing a smooth time curve. It is still conditional on available outcomes and uses a **mean**, not G’s median across fish.

Whole-fish resampling with 999 successful replicates gives the joint intervals in the table. A separate pointwise fish-phase analysis uses the following intervals:

{{table:fish-pointwise}}

The late-training pointwise interval excludes zero while the joint interval does not. That is not an arithmetic contradiction: the claimed family and procedure differ. Do not use the narrower interval as if it simultaneously supported every time point.

{{figure:unsmoothed}}

### Sensitivities: what was examined

{{table:sensitivities}}

The **categorical-trial sensitivity** estimates a separate condition effect for each scheduled trial rather than a smooth curve. Its saved pointwise intervals are a mean-function comparison; they are not a calibrated simultaneous timing analysis. They should not inherit stars or a timing claim from the smoother model.

The **actual-US-overlap sensitivity** removes affected outcome rows, not whole fish by response magnitude. The later run obtained accepted fits for both targets. This checks a particular timing contamination concern, not all possible protocol problems.

The **recording-day bootstrap** requested 199 resamples and obtained 199 accepted fits for each target. Late-training intervals remain positive; early testing remains dependent on baseline treatment. Day grouping is a sensitivity to dependence/balance, not proof that day is a randomized blocking factor or that every rig effect is estimable.

All 57 leave-one-fish-out fits pass for each target after numerical retries. This shows no omitted fish causes a numerical failure in that set and quantifies changes in estimates. It is not a license to remove the fish with the largest effect on significance.

<details><summary>Saved population figure and diagnostics</summary>

{{figure:population}}

{{figure:difference-diagnostics}}

{{figure:adjusted_response-diagnostics}}

</details>

### What implementation review repaired

| Issue found in the first independent review | Subsequent repair / retained boundary |
| --- | --- |
| Individual fitting and validation used different timing grids | One scheduled-grid rule used for fitting, simulation and leave-one-control-out validation; observed runs remain separate |
| Incomplete optimizer retries caused failed sensitivities/omissions | Same-structure retries added, with all attempts retained |
| Stale completion hash and unsafe resume/publication | Stage source/dependency snapshots, locks, hash verification and atomic stage publication |
| Failed full target crashed stabilization | Failure status propagates and suppresses its inference while other reporting can complete |
| Simulation report hardcoded settings | Descriptions derived from recorded settings |
| Unsmoothed summary incorrectly counted zero accepted experiments | Rebuilt acceptance summary from the intact 1,100 saved results; superseded report preserved |
| Slow Gaussian refits | Numerically checked Gaussian/Woodbury backend and integer fish multiplicities; an engineering equivalence check, not statistical validation |

The implementation review recorded **22 scoped unit tests**. The later star-presentation work added separate presentation tests. These verify specified behavior, not nominal scientific coverage. The first run’s defects should not be cited as if they still describe the selected second report, and the repairs should not conceal the remaining statistical problems.

<details><summary>Ideas discussed versus methods actually executed</summary>

The broader plans mention fish-level permutation checks, outcome-appropriate generalized models, GEE, hierarchical Bayesian alternatives and other trajectory/latent-state approaches. A proposal in a plan is not a completed analysis. The independent G comparison reported here executed the two Gaussian population targets, model-refit bootstrap, unsmoothed fish-change bootstrap, saved sensitivities and conditional individual spline/state-space explorations.

The older generic LME/permutation/bootstrap routes discussed in Plan 06 pooled conditions and asked about overall change. They do not answer the required condition difference in change and are not substitutes for the condition-aware results here. This guide does not claim a new condition-aware permutation result, validated movement-probability model, robust heavy-tailed population fit or Bayesian posterior analysis was completed when its saved outputs are absent.

For a final outcome such as movement occurrence or bout counts, the model family and exposure definition would need separate approval and validation. Their discussion does not license transferring the Gaussian conditional-intensity results.

</details>

{{sources:analysis-state,analysis-manifest,comparison,review01,review02,analysis-difference-result,analysis-adjusted_response-result,analysis-unsmoothed-fish-change-result,analysis-fish-phase-robustness,analysis-day-condition-balance,analysis-actual-US-window-overlaps}}

## 07. Acquisition timing and individual trajectories

### Define acquisition before asking for its trial

The exploratory timing rule asks for the earliest **scheduled training trial** at which a conditioned-direction effect exceeds a threshold for **K = 3 consecutive scheduled training trials**. It is restricted to CS15–64. A gap in observed trials cannot be compressed into “three consecutive trials,” and a run cannot bridge training and testing.

Candidate meaningful-effect thresholds were 0%, 5%, 10% and 20% on the conditioned-direction ratio scale:

<div class="formula">δ = −ln(1 − f)<br>0% → 0; 5% → 0.05129; 10% → 0.10536; 20% → 0.22314 log units.</div>

These are **unapproved exploratory thresholds**. Zero asks for any positive effect, including one too small to matter biologically. A 10% threshold asks a different scientific question. The choice must not be made because one yields a pleasant acquisition trial.

### Four timing quantities that must not be conflated

- **Point crossing:** the estimated curve exceeds δ for the required run.
- **Earliest compatible onset:** the upper uncertainty envelope first permits the required run.
- **First supported run / nominal latest bound:** the lower envelope first supports it for the required run.
- **Observed supported run:** corresponding evidence at actually available scheduled observations, reported separately from model-based latent timing.

For calibrated simultaneous envelopes, upper and lower curves can bracket a timing functional. Here those bounds are still nominal because calibration is incomplete. An absent latest bound means timing is not localized to a finite upper endpoint; it is not silently replaced by CS64 or CS94.

{{table:timing}}

At the 10% threshold, the two smooth point crossings are **CS48 and CS50**, with earliest-compatible trial **20** but **no finite latest bound**. The unsmoothed point crossing is **42**, also without a finite latest bound. These are not three competing precise acquisition dates.

At the zero threshold, nominal smooth bounds are **[15,22]**, with point crossings **17** and **16**. The unsmoothed nominal bound is much broader, **[15,51]**. None is a paper-approved biological acquisition interval. The selected G first BH star at **19** and the exploratory zero-threshold supported run at **22** also answer different operational questions.

**No meaningful acquisition time has been established by selecting the first star, point crossing, or most favorable threshold.** Group-level acquisition timing also does not imply that every fish acquired at that trial.

### Individual models tried, without assigning learners

The independent individual work fitted two conditional empirical-Bayes trajectory candidates: a spline and a state-space model. They use the direct-change outcome, fit only Pre/training **CS5–64**, and withhold early-test **CS65–69** for a predictive check. These are separate from the legacy classifiers, which use late testing to construct their labels.

The exploratory information rule requires at least **3 Pre and 10 training observations**. Those cutoffs are not approved paper eligibility rules. Saved timing outputs include **456 main candidate rows** (57 fish × 2 methods × 4 thresholds) and **224 leave-one-control-out rows**. These are candidate-method/threshold records, not hundreds of fish or learners.

Covariance parameters are estimated and then treated as fixed in individual uncertainty. Variance ×0.5/2 and AR1 ±0.1 sensitivities explore settings but do not integrate that estimation uncertainty. The saved pilot shows poor abrupt/transient trajectory coverage. **No new binary learner labels were assigned by this independent analysis.** Consequently there is no new approved “number of learners” to report from it.

<details><summary>Held-out predictive summaries, and what they do not validate</summary>

{{table:heldout}}

Prediction of early-test summaries can be informative. These summaries do not calibrate each fish’s acquisition-time interval, show a discrete learner class, or transform a shrinkage estimate into a probability of learning. Leave-one-control-out validation reduces a control’s self-reference; it does not replace independent biological validation.

</details>

{{sources:analysis-difference-result,analysis-adjusted_response-result,analysis-unsmoothed-fish-change-result,analysis-individual-model-diagnostics,analysis-individual-heldout-summary,analysis-individual-covariance-sensitivity-status,analysis-individual-missingness-review,review02}}

## 08. Calibration: what the procedures actually recovered

Simulation is useful because the true curve, phase effects and timing are known. A confidence procedure advertised as 95% should cover the relevant truth at roughly its intended rate across suitable repeated experiments. The scenarios, sample sizes, missingness mechanism and complete failure behavior matter.

### Smooth LMM population pilot

The expanded study has **110 independent simulated datasets**: 11 scenarios × 10 datasets. Each dataset is analyzed with both population targets, giving **220 model cases**, not 220 independent experiments. Each case requests **99 bootstrap refits**.

Scenarios include null, pre-existing difference, abrupt acquisition, gradual acquisition, transient response, unequal coverage, heavy tails/serial dependence, an influential fish, informative availability, tracking gaps and zero random-effect truth.

{{table:lmm-calibration}}

<div class="callout caution"><span class="label">Material limitation</span><p>For both smooth targets, the full-curve band covered the true curve in <strong>0/10 abrupt and 0/10 transient examples</strong>. More bootstrap replicates alone cannot make a restrictive smooth mean function represent a sharp or brief truth.</p></div>

The phase-average bands covered those particular examples in 10/10 cases. The abrupt examples’ onset bounds contained the true onset at CS30. Therefore the failed curve coverage does not establish that those particular timing bounds failed. It does establish a serious problem with interpreting the nominal trajectory band as generally reliable under those alternatives.

Zero-random-effect scenarios expose numerical and bootstrap failures; inference is withheld when the acceptance gate fails. A coverage fraction among accepted bands must be read alongside the number of withheld cases. Ten datasets per scenario are too few to approve a paper procedure or precisely estimate an error rate.

Because selected G uses the same adjusted smooth mean family, these misspecification concerns also affect interpretation of its trial curve. The pilot did **not** directly recalibrate all of G’s original BH/Holm D/M/R tests. It cannot establish their exact error rates or show each individual selected p-value is wrong.

### Unsmoothed comparison study

This separate study has **1,100 independent experiments**: 100 in each of 11 scenarios, with **499 bootstrap resamples** each.

{{table:unsmoothed-calibration}}

Full-curve coverage ranges from **92% to 98%**, with binomial uncertainty shown above. No false supported 10% run occurred in these 100 experiments per scenario. Zero out of 100 is not proof of zero risk; its saved 95% binomial upper bound is about **3.62%**.

The onset bounds contain truth in 100/100 per scenario, often because they are broad or unbounded. A very wide interval can cover well while localizing timing poorly. These simulations support further study of the comparator; they neither validate the LMM nor demonstrate precise acquisition recovery on this experiment.

### Individual pilot

The individual spline/state-space study used a small six-scenario pilot, with one simulated experiment per scenario. Multiple fish within an experiment are not independent replications of the entire procedure. Poor abrupt/transient coverage and omitted covariance-parameter uncertainty remain. No validated individual timing interval or canonical learner classification follows.

### What remains before final inference

Choose the intended estimand and realistic truth/availability scenarios, then compare a bounded set of trajectory and uncertainty methods using prespecified acceptance criteria. Include curve, phase and timing performance, fit failures, dependence and influential-fish behavior. A simpler phase-level conclusion may be defensible before a trial-level acquisition date is.

Do not choose the final method by which one yields the smallest p-value or earliest acquisition. Calibration and scientific fit to the question must be reviewed together.

{{sources:analysis-calibration-pilot,analysis-calibration-status,analysis-unsmoothed-calibration-summary,analysis-unsmoothed-calibration-status,analysis-individual-calibration-pilot,review02,plan03}}

## 09. Missing outcomes, discarding fish and technical QC

### Four different operations

| Situation | What it means | Consequence |
| --- | --- | --- |
| Trial outcome unavailable | At least one window has no eligible positive bout samples for the paired outcome | This trial cannot contribute conditional vigor; fish membership need not change |
| Fish excluded from cohort | A whole recording is removed under an explicit membership rule | Changes the population being analyzed |
| Fish unclassified by a learner rule | Required coverage/features are unavailable or its model route cannot classify it | No learner decision for that rule; not a non-learner conclusion |
| Eligible fish unflagged | Its inputs were sufficient, but the classifier rule was not met | Algorithm-specific result; not proof it cannot learn |

### Missing trials in G

Controls contribute **2,260/2,520** scheduled paired outcomes: **260 unavailable (10.3%)**. Delay contributes **2,551/2,610**: **59 unavailable (2.3%)**. All 319 unavailable pairs involve no detected valid bout in at least one window.

Conditional bout intensity is undefined when there is no eligible bout. Assigning zero would turn “no intensity measurement conditional on a bout” into “measured zero intensity,” changing the outcome. A missing outcome does not alone tell us whether the fish was biologically still or whether detection/measurement failed.

Availability is behavior-related and differs between conditions. Analyses of available conditional vigor cannot automatically be generalized to all-window movement, total activity or the full cohort’s movement probability. Fish resampling preserves the observed missing pattern but does not remove this selection issue.

The saved movement-occurrence/time-moving companion provides descriptive context and needs its own outcome-appropriate inferential analysis if it becomes a paper claim. It is not correct to attach G’s Gaussian-model stars to it.

<details><summary>Saved descriptive movement companion</summary>

{{figure:movement}}

</details>

The unapproved “≥95% detector-valid frames” stress rule retained **zero** paired outcomes. Detector-valid finite adjacent frames are not the same quantity as raw coordinate-valid frames or the fraction of time spent moving. This stress rule is not an adopted fish-quality policy.

### What February and March code actually says

The reviewed Git revisions are **2f63ef48361d6e80d1b8e1393894d7e5a3dedf55** (February 14) and **bf46bf7b02baaf6d8138d9772f255881f86c3c78** (March 24). Five relevant source files per revision were snapshotted and hashed; the preprocessing and LogMedian plotting files are byte-identical between these revisions.

Their `RUN_DISCARD` and `APPLY_FISH_DISCARD` defaults are false. This establishes the saved defaults, not what someone enabled interactively in every original run. No saved original discarded-fish roster was established in the inspected Git trees.

The statistical six-trial filter retains a fish meeting the requirement in **ANY block**, not every block, then retains its other eligible rows. Requiring every block would be a different rule and a different cohort.

<details><summary>Historical activity and viability rules, precisely</summary>

- Historical baseline **[−15,0]**, CR **[0,9]** and US **[0,5]** bout windows use inclusive endpoints. They are evaluated separately from G’s half-open outcome windows.
- Selected five-trial QC blocks are Late Pre CS10–14, Early Test CS65–69 and Late Test CS90–94. Baseline/CR rules require at least three bout-bearing trials in each selected block.
- The training-US rule requires a response at all specified US ordinals **18–63**.
- Historical last-US viability requires the first frame-level `Trial time (s)` where the last trial's `US end > 0` to be at least **0.4 s**, and a detected `Bout` somewhere in that last trial's inclusive **[0,5] s** US window. The 0.4 s check is on the annotated US-end time; it is not a minimum bout duration. Exact reconstruction needs the original segmented frame/Bout data and 700-FPS semantics, which are unavailable here.
- The combined projection instead requires raw last-US duration ≥400 ms plus detected movement in [0,5] s, an explicitly labeled proxy. Raw event duration is not the original frame-annotated end coordinate. Satisfying this proxy does not prove the exact historical viability rule was satisfied.
- The historical allDelay configuration has no Re-Train group; the retraining-US component is bypassed in these projections.

The rule catalogue and source snapshots are linked below. These projections use current authenticated corrected data. They reconstruct rule behavior, not the exact cohort originally analyzed in February or March.

</details>

### What happened when we tried those exclusions

The recipes were registered before fitting their condition effects. None replaces the primary selected cohort.

{{table:cohorts}}

The combined projection keeps **24 control and 27 Delay fish**, with **4,455 eligible outcomes**. The primary keeps 28/29 and 4,811. Filtering raises the late-training estimates from approximately 0.103/0.095 to 0.124/0.120 for direct/adjusted targets. All recipes have positive nominal late-training LMM intervals, and none provides a finite latest 10% acquisition bound.

An increase after behavioral filtering is not evidence that excluded fish were technically defective. Baseline/CR/US activity rules select behavior and change the population. In particular, filtering on a response related to conditioning can remove part of the biological heterogeneity one intended to study.

<details><summary>Reconstructed excluded identifiers by recipe</summary>

{{table:excluded}}

The full saved per-rule failures, overlaps and cumulative attrition remain available in the source inventory. These identifiers are reported for transparency, not proposed exclusions from the primary analysis.

</details>

### Independent technical evidence

Across the 57 inspected acquisition/preprocessing summaries, FrameID gaps, duplicates and reversals are zero and the coordinate-valid fraction is 1.0. Stream-boundary nonoverlaps of 0–5 frames are not automatic exclusions. These limited checks do not establish comprehensive tracking quality.

Delay fish **20221122_03** has genuine protocol-record anomalies: one invalid duration and an event outside acquisition. Its authenticated raw protocol shows timestamp precision loss in the first two reinforcer records. Intended timestamps remain unknown and were not reconstructed. The availability of selected CS5–94 outcome windows does not establish that this corruption is scientifically irrelevant.

A separate source-only sensitivity removes any fish with that protocol flag, selecting the rule independently of vigor and learner status. It removes this one Delay fish and keeps **28/28 fish, 4,721 outcomes**.

{{table:technical}}

Late-training intervals remain positive, and meaningful 10% timing remains unlocalized. This is evidence about robustness to that technical concern. It is not automatic approval to discard the entire fish. Decide whether the defect invalidates relevant exposure/timing, whether affected events alone can be handled, and whether a whole-recording rule is scientifically warranted.

**Current recommendation, not an adopted decision:** retain the selected cohort for review; keep outcome-dependent activity filters as explicit sensitivities; resolve the protocol-flag policy from technical evidence rather than its effect on a p-value. Missing bouts alone do not justify a blanket fish exclusion.

{{sources:analysis-legacy-source-catalog,analysis-legacy-assessment-status,analysis-legacy-cohort-phase-results,analysis-legacy-cohort-timing-results,analysis-legacy-sensitivity-membership,analysis-legacy-discard-rule-details,analysis-legacy-cumulative-flow,analysis-technical-recording-audit,analysis-raw-protocol-trace,analysis-technical-policy-before-fitting,analysis-technical-phase-results,analysis-technical-coverage-sensitivity}}

## 10. Legacy learner algorithms and their counts

“Learner” is an operational output of a specified rule. The four legacy scripts do not share one stable definition. The names Nominal, New, Improved and WIP identify implementations; “Improved” is a filename, not a validation conclusion.

### What all four do

The scripts turn trial outcomes into acquisition and recovery features using mixed-model fixed effects plus fish-specific epoch random slopes and their uncertainty. They compare multivariate departures with controls and require expected feature directions: reduced response for acquisition and increased response for recovery.

Their common coverage logic includes ordered five-trial sub-blocks, at least **three trials in every included five-trial block**, at least **six trials per active epoch**, complete selected features, and at least **ten complete fish** before classification. Control anchoring is used when at least three controls are available; fallback behavior is part of the legacy implementation.

These are classifier eligibility rules, not G’s whole-fish inclusion policy. Adding more middle-epoch blocks can make a rule stricter about coverage even while changing its estimated features.

### How the four rules differ

| Rule | Modeled trial quantity | Middle epoch | Multivariate threshold | Expected-direction gate |
| --- | --- | --- | --- | --- |
| Nominal | log(Mean CR + 1), covariate log(baseline + 1) | CS60–69: late training + early test | Shrinkage Mahalanobis distance; bootstrapped 75th control percentile, 1,000 resamples | Two votes: point direction and “conservative” shift |
| New | Same log(x+1) response/baseline treatment | CS40–69 | Mahalanobis; bootstrapped 75th percentile, 50 resamples by default | Point direction plus Gaussian directional score >0.60 |
| Improved | log(Mean CR), covariate log(baseline); positive inputs | CS40–69 | One-sided shrinkage-whitened joint score; empirical 90th control percentile | All required features in the expected direction, SE-aware |
| WIP | log(Normalized vigor), normally arithmetic CR/baseline ratio; no baseline covariate | CS40–69 | One-sided joint score; empirical 95th control percentile | Unanimous required directions |

The shared Pre epoch is CS5–14 and the late recovery comparison CS85–94. The middle epoch is used on opposite sides of acquisition and recovery contrasts, creating feature dependence. These rules therefore combine acquisition **and recovery**, including testing data. They are not training-only acquisition classifiers.

<details><summary>Technical details that change interpretation</summary>

The Nominal “conservative” decision uses a **0.5-SE shift**, while displayed feature intervals use a **1.96-SE** convention. A drawn interval excluding zero is not the literal Nominal vote rule. The New directional score is a Gaussian approximation used with a >0.60 threshold; it is not a validated probability that a fish learned.

Mahalanobis scores account for covariance among features with shrinkage. Improved/WIP use directional, whitened scores rather than making distance alone the decision. Empirical tail ranks or p-like diagnostics should not be substituted for the algorithm’s actual score cutoff and direction gates.

Improved/WIP reduce a control’s self-influence in its reference mean using leave-one-out calculations. The overall threshold is still learned from these controls; the observed control flags are not an independent estimate of out-of-sample false-positive risk.

Responses and uncertainty are based on conditional/shrunken fish effects. Nonfinite or nonpositive feature SEs can be floored to 0.01; models have legacy fallbacks. Those are documented behaviors, not accepted failure rules for a new canonical classifier. log(x+1) also depends on the units/scale of x, so a translated modern input can change the behavior of an old rule.

Historical individual plots can include a cumulative BLUP overlay on a separate vertical scale. That is a schematic of epoch features, not a fitted trial-by-trial behavioral curve. Heatmap, observed trial plot and classifier inputs can summarize the same recording differently. Similar colors do not make their outcomes identical.

</details>

### Count set A: September reapplication

{{table:september}}

This saved comparison was generated **September 24** using corrected outcomes translated as Mean CR = response total activity, baseline = baseline total activity and Normalized vigor = their ratio. It includes all 57 fish as classifier-eligible in those results. Its metadata explicitly sets **historical_equivalence = false**, **canonical_classifier = null**, and descriptive-classifier-characterization status.

Therefore **7/7/6/19 are not authenticated counts from the original February/March runs**. They are outputs of later reapplications of historical rules. The relevant early code and later saved output must not be collapsed into one historical result.

### Count set B: later bout-only reapplication

{{table:bout-learners}}

This October 15 comparison belongs to the later individual-fish workstream and is supplied as context. It uses the valid-bout-frame policy and exact legacy-classifier inputs appropriate to each rule. Its G-style main trial plots use G’s observed LogMedian difference, but the original classifier calculations remain separately defined.

The Delay eligible denominators are **27, 26, 26, 26**. All four have **23 eligible controls**. Counts 6/8/9/16 must not be displayed as if every method classified all 29 Delay fish. The unclassified fish remain cohort members; they are not added to the non-learner category.

### Why the counts changed

The two applications differ in input generation, activity/sample policy, translated outcome quantities and resulting coverage. The algorithms themselves differ in baseline treatment, epoch span, threshold and direction rule. Their complete-case requirements then select different eligible sets.

The recorded comparison establishes these differences, not a unique numerical decomposition assigning each changed label to one cause. Such a decomposition would need an explicitly designed matched-input experiment; it has not been silently invented here.

The wide variation is itself evidence that “number of learners” is method-dependent at present. The independent trajectory analysis in this chat assigned **no binary labels**. No canonical validated learner count was selected by any of these explorations.

### What can be concluded about individual learning

There is heterogeneity worth analyzing, but categorical classes must earn their interpretation. A continuous score, conditional trajectory or uncertainty-aware response representation may be preferable if a discrete boundary is unstable. That is a scientific choice still requiring validation, not permission to omit the learner workstream.

Using learner labels derived from the same outcomes to show that “learners suppress more” is circular as independent confirmation. Late-test recovery helped define these legacy labels, so it cannot then serve as held-out evidence for those same labels.

Detailed optional inputs and decisions: {{link:bout-classification}}. The existing full atlas remains at the source path recorded in its state; it is not embedded here because you requested little individual-fish material.

{{sources:legacy-matrix,legacy-review,september-learners,legacy-audit,bout-learners,bout-classification,atlas-state}}

## 11. Presentation alternatives, asterisks and neuroscience examples

### Why the exploratory G looks different

Original G places observed condition medians and several model-test rows in one panel. The exploratory layout separates **observations**, **the fitted condition-change contrast**, **planned phase estimates** and **numbers contributing**. That makes the inferential target and uncertainty visible, but produces a taller, structurally different panel.

The new adjusted fitted curve is almost identical to G’s phase fit. The redesign nevertheless changes the visible statistical story by replacing the original BH/Holm/local-test annotation families with a joint-band view of trial and planned phase contrasts. That is a substantive inferential-presentation change, not just adding stars to the same test.

Your concern that it feels unlike G is reasonable: continuity of familiar visual form is a real design consideration. The purpose of this guide is to make the tradeoff explicit, not to assume the redesign should be the final figure.

### Version without asterisks

{{figure:plain-g}}

The observed curves retain G’s selected medians and 5,000-resample pointwise bands. The adjusted contrast and phase estimates use the saved 999-refit joint uncertainty. The zero reference and interval position communicate the statistical result without a separate significance label. Contribution information exposes variable trial coverage.

This separate panel is approximately **183 × 122 mm**, compared with original G’s **183 × 98 mm** source canvas; the current assembly uses its own compact allocation. Screen enlargement in this guide is for reading, not a new publication-size decision.

### Dense asterisks, then one bracket

<details class="archive-figure"><summary>Superseded dense-star preview</summary>

{{figure:dense-g}}

</details>

The first star preview marked each qualifying trial individually and produced 40 closely spaced marks. It was retained as an unpublished preview. The later version groups the same contiguous qualifying trials with one bracket and a single binary asterisk.

{{figure:bracket-g}}

The bracket spans **CS22–61**. It means **each individual trial in that contiguous group** has a finite nominal joint 95% interval excluding zero and adequate declared coverage. It is not a pooled statistical test of the whole bracketed range. The separate late-training phase estimate also receives one star; early and late testing do not.

The new single * has one declared meaning: joint interval excludes zero. It does **not** encode * / ** / *** p-value tiers, use BH p-values, identify a learner, or show an acquisition interval. The range is 40 qualifying trials; it is not 40 independent discoveries. Model calibration remains unresolved.

The added stars and bracket do not change the saved fitted data geometry. Their decision file records the qualifying trials and exact rule. This annotation can be defensible as an exploratory visual aid only when the caption explains its meaning and limitations.

| Feature | Selected original G | Separate exploratory bracket version |
| --- | --- | --- |
| Observed summary | Median of fish LogMedian differences | Same selected observed median inputs |
| Observed shading | Pointwise fish-bootstrap CI | Same descriptive CI |
| Main inferential emphasis | D/M/R rows and BH90 trial strip | Explicit adjusted trial contrast, joint band and planned phases |
| Trial marks | BH90 rejects at 19–71 | Joint interval criterion at 22–61 |
| Bootstrap supplying model inference | Original conditional Gaussian/Wald tests; descriptive bootstrap is separate | 999 requested whole-fish model refits, all adjusted refits accepted |
| Timing claim | None | None validated; bracket is not onset |
| Authority | October 9 scoped scientific freeze | Separate unfrozen exploratory presentation |

### What neuroscience examples support

There is no single universal “LME plot” convention. The useful principle is to distinguish measured summaries from model-derived effects and show the magnitude and uncertainty of the contrast readers should interpret.

- **Yu and colleagues, Neuron (2022):** a primer on mixed models for clustered neuroscience observations. It supports explaining the nesting and analysis unit. It does not establish the correct trajectory family for zebrafish acquisition. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8763600/).
- **Heise, Mon and Bowman (2022):** an ERP tutorial addressing unequal trial counts and exclusions through LME analysis. It provides a relevant example of retaining incomplete repeated observations under specified assumptions; it does not resolve behavior-dependent no-bout intensity here. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8987653/).
- **Loewinger and colleagues, eLife:** a functional mixed-model framework for trial-level photometry dynamics. It motivates displaying time-varying model effects and their uncertainty. Its within-trial photometry problem differs from our across-training acquisition question. [Primary article and figures](https://elifesciences.org/articles/95802/figures). Direct page access was intermittently challenge-blocked during this guide’s verification; the publisher’s indexed article text was available. No claim of a newly inspected figure is made here.
- **eNeuro author guidance:** requests effect-size reporting with uncertainty and specification of the statistical analysis. It supports keeping estimates/intervals visible when stars are used. This is reporting guidance, not validation of our estimator. [Journal guidance](https://www.eneuro.org/content/preparing-manuscript).

**Presentation recommendation, not a selected final panel:** keep recognizable observed trajectories, make the tested condition-change estimand explicit, and retain effect estimates with intervals. If space requires a compact main G, consider a directly linked companion for detailed contrasts/diagnostics rather than compressing unreadable statistics. The final allocation across panels is your decision after the primary analysis is chosen.

{{sources:exploratory-state,asterisk-decisions,comparison-prose,dense-output-record,plain-g-legend,bracket-g-legend}}

## 12. Conclusions and your eventual decisions

### What the saved evidence supports

- The selected conditional bout-intensity outcomes and adjusted phase-model contrasts were reproduced. No numerical extraction error was identified in the matched check.
- Later training shows a conditioned-direction population change. Both smooth LMM targets have positive nominal joint intervals, and several cohort/covariance sensitivities retain that direction.
- Early-testing inference depends on baseline treatment. The adjusted joint interval includes zero while the direct-change interval excludes it.
- The unsmoothed late-training estimate points in a similar direction but its joint interval spans zero. Model-independent significance has not been established.
- Missing paired conditional-intensity outcomes are substantially more common in controls. This limits interpretation as an all-window or whole-cohort activity effect.
- No biologically meaningful precise acquisition trial, canonical number of learners, or extinction result has been established by these analyses.

### What “is current G wrong?” can and cannot mean

**Calculation:** the selected outcomes and matched fitted adjusted curve reproduce. That is evidence against a numerical discrepancy in those checked quantities.

**Interpretation:** treating its descriptive median bands as the intervals behind its stars, reading trial19 as acquisition, or treating disappearance of stars as extinction would be wrong. The saved panel’s own limitations already reject those interpretations.

**Final inferential adequacy:** heavy tails, mean-function misspecification, dependence and availability remain material. The current G and exploratory joint-band figure are not fully validated confirmatory analyses. A scientific freeze preserves a scoped artifact; it does not make these concerns disappear.

### Decisions to make in order

| Step | Decision you need to articulate | Evidence to review / completion criterion |
| --- | --- | --- |
| 1. Biological claim | Conditional bout intensity, total activity, movement occurrence, or distinct primary/secondary outcomes? | State the population and quantity in one sentence; avoid treating outcomes as interchangeable |
| 2. Outcome | Typical logged intensity or arithmetic-average intensity? | Justify the aggregation order, positivity/zero policy and role of bursts scientifically |
| 3. Cohort and trial availability | Which technical exclusions, and which trial contribution rules? | Reason-coded membership and eligibility; resolve protocol corruption; no selection by favorable significance |
| 4. Baseline treatment | Direct subtraction or covariate adjustment as primary? | State the estimand and baseline role; retain the other as a justified sensitivity |
| 5. Claim hierarchy | Which phase contrasts are primary, and is trial timing necessary? | Define reference, sign, windows and multiplicity family before final fitting |
| 6. Model and uncertainty | Which trajectory family and covariance/robustness approach? | Relevant calibration, failure gates, residuals, influence and availability stress; numerical success alone insufficient |
| 7. Acquisition | Meaningful δ, persistence K, scheduled-trial family and uncertainty? | A validated timing procedure, or explicitly report acquisition not localized |
| 8. Individual learning | Continuous representation, conditional trajectory or a categorical rule? | Eligibility, uncertainty and validation matched to the representation; avoid same-data confirmation |
| 9. Extinction, if claimed | What decline or sufficiently small remaining effect, relative to which reference? | A separately specified and validated analysis; not a nonsignificant late test |
| 10. Figure | Which observations, effects, intervals and annotations belong in G or a companion? | Values trace to the selected analysis; readable caption; distinguish model and observed quantities |
| 11. Freeze | Which exact scoped scientific and presentation versions? | Explicit author choices, authenticated dependencies, required numerical/visual checks and unresolved decisions recorded |

### A practical independent-review exercise

Choose one proposed sentence for the paper, such as “Delay fish show a greater suppression by late training.” Then answer, in your own words:

1. Suppression of exactly which measured quantity?
2. Relative to which control and Pre reference?
3. Estimated by which model and aggregation order?
4. With which confidence procedure and multiplicity family?
5. Among which fish and available observations?
6. How sensitive is it to baseline treatment, covariance, exclusion policy and trajectory assumptions?
7. What does it leave unresolved about acquisition, individuals and extinction?

If the sentence cannot be tied to these answers, it is not ready for a final figure caption. You can record the answers in this chapter’s notebook and export them for discussion.

### Open evidence gaps

- The actual fish discarded in the original February/March interactive runs are not established by the inspected source defaults.
- The original segmented frame/Bout/700-FPS inputs needed for exact last-US end-time and bout-presence reconstruction are unavailable; the saved combined recipe uses a labeled proxy.
- The exact selected full M/R CSV remains unrecovered. Displayed entries are authenticated through the selected sidecar; the complete non-rejection table has not been fabricated.
- Intended values of the corrupted protocol timestamps are unknown. No invented correction was applied.
- Final target, technical membership policy, meaningful timing threshold, calibrated trajectory/timing uncertainty and learner representation remain unapproved.

This guide adds no new freeze and selects no final analysis. Its role is to make those decisions reviewable.

{{sources:comparison-prose,review02,plan03,plan06,selection,analysis-legacy-assessment-status,analysis-raw-protocol-trace}}

## 13. Glossary and manual source checks

| Term | Meaning in this guide |
| --- | --- |
| Bout-only | Samples restricted to detected eligible movement bouts |
| Conditional intensity | Movement intensity conditional on an eligible bout being detected; not movement probability |
| Cohort | The fish included as members of the analysis population |
| Eligibility | Whether an observation or fish has the information required by a specific analysis |
| Estimand | Precisely the population quantity an analysis is meant to estimate |
| ANCOVA / baseline adjustment | Model a response with baseline as an explanatory covariate |
| Fixed effects | Population coefficients used to construct model contrasts |
| Random effects | Fish-specific deviations used to represent heterogeneity/dependence |
| BLUP / empirical Bayes | A conditional, typically shrunken estimate of a fish effect using fitted population covariance |
| Shrinkage | Pulling noisy estimates toward a shared reference; not evidence of categorical classes |
| Pointwise interval | Uncertainty targeting one specified quantity at a time |
| Simultaneous interval/band | Uncertainty constructed for a declared family together |
| Percentile bootstrap | Use quantiles of resampled estimates as interval endpoints |
| Bootstrap-t / max statistic | Standardize resampled errors and use their family maximum for a joint critical value |
| Holm | Multiplicity adjustment targeting familywise false rejection control under valid component tests |
| BH / FDR | Benjamini–Hochberg adjustment targeting false-discovery proportion under applicable conditions |
| Wald inference | Approximate inference from fitted coefficients and their covariance |
| Singular fit | A variance/covariance structure collapses under the specified numerical criterion |
| Hessian | Curvature of the fitted objective; useful in numerical assessment, not a biological validation |
| Excess kurtosis | Tail-shape summary relative to the Gaussian reference, whose value is 0 |
| AR1 / lag1 | A simple adjacent-time correlation description; not the complete residual covariance |
| LOFO | Leave-one-fish-out refitting to examine influence |
| LOO control | Leave-one-control-out reference/validation; distinguish from fully independent validation |
| Calibration | Check coverage or error behavior of the complete procedure against known simulated truth |
| Acquisition threshold | Prespecified magnitude needed for the operational acquisition definition |
| Persistence | Number of consecutive scheduled trials required by a timing rule |
| Unlocalized | The evidence does not bound the timing to the required finite range |
| Unclassified | A rule cannot give a decision with its available inputs; not “non-learner” |
| Frozen | A scoped artifact and its dependencies are recorded; not universal scientific validation |

### How to inspect the evidence without AI

Every “Evidence” link opens the source inventory entry with the original path, hash and a saved copy where appropriate. CSV files can be opened in a spreadsheet or text editor. JSON records describe configurations, source identities and result status. SVG files are editable vector artwork and can be inspected in a browser or vector editor.

For a population result, open the planned-phase CSV and check the target, phase name, estimate and interval columns. Open its result JSON to check requested/accepted bootstrap counts and approval flags. The associated manifest identifies the exact bytes. For G’s original D marks, use the selected D-interaction-tests.csv; for the black strip, use selected phase-contrasts.csv. Do not infer all test definitions from the visual star alone.

For an exclusion result, compare the recipe name in the phase table with the saved membership and per-rule files. For learners, compare flagged **and eligible** counts, then the unclassified category. Do not compare a numerator alone across input generations.

The guide’s source copies are explanatory evidence snapshots, not replacements for the immutable frozen source layout. SHA-256 identity establishes that bytes match; it does not establish that a model is statistically appropriate. If source state later changes, build a new dated guide rather than treating this snapshot as automatically current.

{{sources:analysis-manifest,frozen-inventory,analysis-legacy-sensitivity-membership,bout-classification}}
