# VLM prompting and calibration for shelf fill-level and product-match estimation (Gemini Flash-Lite class)

Research date: 2026-10-11. Source dates are given in brackets; arXiv IDs encode year-month (2505 = May 2025). Anything labelled "Inference" is my reasoning, not a cited result. Overall caveat: I found NO published study that evaluates a VLM on retail shelf or crate fill-level (empty/half/full) with in-context reference crops, so most evidence is adjacent (counting, calibration, ICL, comparison). Everything must be validated on our own LOFO labels.

## 1. Do same-area reference crops (full / empty examples) in-context improve fill estimation? What does the literature say about few-shot visual ICL for quantity estimation?

### Takeaway
Direct evidence for same-area reference crops on fill estimation does not exist in what I found; the closest evidence (many-shot multimodal ICL on Gemini 1.5 Pro, +17 points average over zero-shot on classification datasets) suggests in-context labelled examples help Gemini-family models substantially, with log-linear gains, so a few labelled "empty/half/full" exemplars per area is a well-motivated bet, to be A/B tested on LOFO.

### Cited Findings
- Many-shot multimodal ICL paper (Jiang et al., Stanford) tested GPT-4o and Gemini 1.5 Pro on 14 datasets; best many-shot gain over zero-shot averaged about +17 points for both models, e.g. Gemini 1.5 Pro +23.1 (HAM10000), +29.2 (FIVES), +37.9 (EuroSAT). Gemini improved roughly log-linearly with more examples; GPT-4o curves were less stable. Gemini was more data-efficient than GPT-4o on most datasets. [arXiv 2405.09798, May 2024, v2 later] — [Many-Shot ICL in Multimodal Foundation Models](https://arxiv.org/html/2405.09798v2)
- Same paper: open-weight models (Llama 3.2-Vision, InternLM-XComposer2.5) did not benefit from extra examples; demonstration sets were class-balanced (stratified) — [arXiv 2405.09798](https://arxiv.org/html/2405.09798v2)
- Same paper: 10 of 14 datasets were classification and only 2 VQA; none was a quantity or fill estimation task, so transfer to fill level is not established — [arXiv 2405.09798](https://arxiv.org/html/2405.09798v2)
- Reference-object prompting (text-level, "Spatial-Prompt") has been reported to help monocular size/distance estimation, but the same study argues that strong priors explain a substantial part of VLM performance on object-size questions. Only the prompt-level version is documented; I found no paper testing a reference IMAGE exemplar. [as summarised in search results, Q-Spatial Bench line of work; date not verified] — [search summary, see CompareBench listing](https://arxiv.org/pdf/2509.22737)
- Benchmarks of relative/number sense show even strong MLLMs struggle with basic counting and proportion/length comparison (VisNumBench, Mar 2025; CompareBench, Sep 2025) — [VisNumBench](https://www.alphaxiv.org/abs/2503.14939.md); [CompareBench](https://arxiv.org/pdf/2509.22737)
- A 2026 paper (Retrieving Counterfactuals Improves Visual In-Context Learning) studies how to choose in-context examples for VLMs (random vs retrieval-based); I only saw prompt templates, not quantity-estimation results — [arXiv 2603.16737, Mar 2026](https://arxiv.org/pdf/2603.16737)
- Retail shelf: I found no peer-reviewed few-shot-MLLM out-of-stock study. A vendor blog (Infosys, undated) reports a VILA VLM zero-shot often mistook empty spaces for low stock, with parameter-efficient fine-tuning reducing false alarms — [Infosys Topaz blog](https://blogs.infosys.com/topaz/nvidia/enhancing-retail-store-analytics-with-video-ai-agent-real-time-video-analytics-part-3-3.html)
- Non-LLM fill-level work (CORSMAL, Jul 2021) uses discrete levels (0, 50, 90 percent filled), supporting coarse ordinal buckets as a sensible target — [CORSMAL benchmark](https://arxiv.org/pdf/2107.12719)

### Inferences
- Because our cameras are fixed and validation frames are same-day, a same-area labelled "empty", "half", "full" exemplar triplet is cheap and well-aligned with the many-shot result; put exemplars before the query image, balanced across classes, with explicit labels. Expect gains mainly on camera-specific appearance (lighting, crate type), the failure mode that zero-shot VLMs have on empty vs low-stock.
- Gemini Flash-Lite is not Gemini 1.5 Pro; ICL gains may be smaller for a lite model, and extra images dilute attention. Test 0, 1 and 3 exemplars per class on LOFO; exclude the query frame (and ideally its temporal neighbours) from the exemplar pool to avoid leakage (project hard rule 6).
- Each 256x256 crop costs only ~258 tokens (see Q5), so 3-6 exemplar images per request is cheap in tokens.

### Gaps
- No direct experiment of same-area reference crops for fill level. No measured effect size for Flash-Lite. No evidence on how many exemplars saturate for small models.

## 2. Pairwise comparison prompts vs absolute fill percentage; ordinal ranking

### Takeaway
In image-quality assessment, pairwise judging is widely preferred to absolute scoring for consistency, but MLLM pairwise judges still have position sensitivity and the pairwise-to-absolute conversion is the hard part; for us, "compare to a same-area empty and full reference" is effectively a 3-way ordinal classification and should be tested against direct empty/half/full, with order swapping.

### Cited Findings
- GenArena switches to pairwise judging to counter calibration drift and inconsistency of absolute pointwise scores, and checks each preference with image-order swapping — [GenArena, arXiv 2602.06013, Feb 2026](https://arxiv.org/pdf/2602.06013)
- Compare2Score: relative ranking is simpler and more reliable than absolute rating, but comparisons scale quadratically and there is no standard efficient way to convert relative judgments to absolute scores — [arXiv 2405.19298, May 2024](https://arxiv.org/html/2405.19298v1)
- Q-Bench+ found only GPT-4V was more accurate on pairwise comparison than single-image evaluation (i.e. pairwise is not automatically better for weaker models) — [arXiv 2402.07116, Feb 2024](https://arxiv.org/abs/2402.07116v2)
- A comprehensive MLLM IQA study notes single-image scoring may not capture relative quality well and allowed a "comparable" option — [arXiv 2403.10854, Mar 2024](https://arxiv.org/pdf/2403.10854)
- GPT-4 absolute scores are overly generous on low-quality items; pairwise results converted to scores by voting — [DepictQA-related, arXiv 2312.08962, Dec 2023](https://arxiv.org/pdf/2312.08962)
- Gemini 2.0 Flash pairwise judgments on sharpness/noise were probabilistic rather than absolute — [IS&T EI 2026 (Mar 2026)](https://library.imaging.org/ei/articles/38/12/GENAI-178)
- A visual aesthetics benchmark of 20 frontier MLLMs: even the best picked both best and worst image correctly in only 26.5% of tasks vs 68.9% for human experts — [benchmarklist summary, secondary source, date unverified](https://benchmarklist.com/models/qwen3-vl-235b-2/)
- CompareBench (Sep 2025): strong VLMs fail at basic comparisons including counting and geometry — [arXiv 2509.22737](https://arxiv.org/pdf/2509.22737)

### Inferences
- With only 3 classes and a known per-area full and empty exemplar, a pairwise formulation reduces to "closer to the FULL reference, the EMPTY reference, or in between", which can be asked as a single multi-image ordinal question (images labelled A=empty ref, B=full ref, C=query) rather than quadratic comparisons. The quadratic-scaling objection does not apply.
- Run both orders (swap A/B) as a position-bias check; if disagreement, treat as uncertain (this is also a free confidence signal).
- Ask for an ordinal bucket (or percentage in 25-point steps) rather than a continuous percentage; continuous percentages from lite models tend to cluster on round numbers (my expectation, not cited).

### Gaps
- No study of pairwise vs absolute for fill/occupancy specifically, and none for Flash-Lite.

## 3. Self-consistency, temperature, structured output, CoT vs direct, and calibrated probabilities

### Takeaway
Do not trust verbalized confidence from VLMs: it is consistently overconfident and near-uninformative for error detection; consistency across samples/orderings and token probabilities are more reliable signals, and the final probabilities should be calibrated on our labelled set with a low-parameter method (temperature or Platt/multinomial logistic), cross-validated by frame group. CoT is not reliably helpful for perception-heavy tasks.

### Cited Findings
- Verbalized calibration in VLMs: current VLMs show notable miscalibration across tasks; thinking-with-images reasoning models are better calibrated; a two-stage "Visual Confidence-Aware Prompting" improves calibration over Top-K prompting and self-reflection baselines. [EMNLP 2025; arXiv 2505.20236, May 2025] — [Xuan et al.](https://arxiv.org/abs/2505.20236)
- LLMs and VLMs (including GPT-4V and Gemini Pro Vision) are overconfident most of the time under verbalized uncertainty — [Groot and Valdenegro-Toro, arXiv 2405.02917, May 2024](https://arxiv.org/pdf/2405.02917)
- Knowledge-boundary study of LVLMs compared probabilistic, answer-consistency and verbalized confidence: probabilistic and consistency signals are more reliable, verbalized confidence tends to overconfidence — [arXiv 2508.19111, Aug 2025](https://arxiv.org/html/2508.19111v1)
- Small VLMs (Qwen2-VL-2B, SmolVLM): stated confidence nearly constant (0.87-0.90) with error-detection AUROC near chance (~0.50); mean token probability separated correct from incorrect (AUROC 0.92-0.99 for Qwen2-VL). Under severe underexposure accuracy collapsed (0.99 to 0.22) and neither signal could be trusted. Single non-peer-reviewed preprint on small open models, not Gemini. — [arXiv 2607.22034, Jul 2026](https://arxiv.org/abs/2607.22034)
- VL-Calibration reports reducing ECE on Qwen3-VL from 0.421 to 0.098 with a decoupled calibration approach (training-based, not directly reusable via API); its related-work describes sampling-based uncertainty via aggregating multiple outputs / perturbed inputs — [arXiv 2604.09529, Apr 2026](https://arxiv.org/pdf/2604.09529)
- Text-only counterpoint: Tian et al. (2023) found verbalized confidence can beat token probabilities for RLHF-tuned LLMs (cited second-hand in the VLM papers above) — [via Xuan et al.](https://arxiv.org/pdf/2505.20236)
- Chain-of-thought and perception: step-by-step prompting can cause overthinking and sometimes lower accuracy; many VLM errors are perception failures, not reasoning failures — [Gated Perception-Reasoning Optimization, arXiv 2601.04442, Jan 2026](https://arxiv.org/html/2601.04442v1). Models improve when reasoning over their own captions on spatial/selective-attention tasks — [Caption This, Reason That, arXiv 2505.21538, May 2025](https://arxiv.org/pdf/2505.21538). Fine-grained counting is attributed to a text bottleneck — see search summary of Chain-of-Visual-Thought work (2025). I found no controlled direct-vs-CoT study on counting/fill.
- Counting failures: VLMs struggle on compositional counting and beyond ~20 objects (VLMCountBench, [arXiv 2510.04401, Oct 2025](https://arxiv.org/pdf/2510.04401)); attention interventions give only modest gains ([arXiv 2511.17722, Nov 2025, v3 Apr 2026](https://arxiv.org/html/2511.17722v3)); "The Count Is There, but Misaligned" argues the count is internally represented but mis-decoded ([arXiv 2607.09544, Jul 2026](https://arxiv.org/pdf/2607.09544)); strong models (o3, Gemini 2.5 Pro) rely on memorised priors that hurt counting (Vo et al. 2025, via search summary of [arXiv 2511.17722](https://arxiv.org/pdf/2511.17722)).
- Calibrators on small sets: practitioner guidance says Platt/temperature scaling are safer for small calibration sets, isotonic regression overfits (rule-of-thumb figures such as ~100 samples per class are unsourced); temperature scaling needs logits, which many LLM APIs do not expose in full. Sources are vignettes/blogs, not LLM studies. — [CRAN probcal vignette "Choosing a Calibrator"](https://cran.r-project.org/web/packages/probcal/vignettes/choosing-a-calibrator.html); [KDnuggets guide, undated](https://www.kdnuggets.com/a-deep-dive-into-calibration-of-language-models-platt-scaling-isotonic-regression-temperature-scaling); [Calibration Meets Reality, arXiv 2509.23665, Sep 2025](https://arxiv.org/pdf/2509.23665)
- Gemini logprobs: official docs I could read do not mention logprobs for Gemini 3.1 Flash-Lite (model page, latest update May 2026) — [Gemini 3.1 Flash-Lite model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite). A forum thread reports a 400 "Logprobs is not enabled" error for gemini-2.5-flash with a comment that 2.0 flash/flash-lite still worked; undated and anecdotal — [Google AI forum](https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989). I could not read the GenerationConfig reference section (page truncated).
- Structured output: Gemini 3.1 Flash-Lite supports structured outputs, thinking (configurable thinking_level), batch API and context caching — [model page, May 2026](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)

### Inferences
- Recommended elicitation (cheap, API-compatible): ask for a JSON schema with an enum for stock level PLUS fields for evidence cues (e.g. visible_floor_fraction as 0/25/50/75/100, product_present boolean) rather than free-form reasoning; put the short "observation" field BEFORE the verdict field so the model describes then decides, without long CoT. Treat thinking level as an A/B knob (minimal vs default), given the evidence that long reasoning does not reliably help perception.
- Probability construction without logprobs: (a) K samples at temperature around 0.7-1.0, or K prompt variants/image orderings, and use the vote fractions as p_e, p_h, p_f; (b) average with a second signal (kNN/temporal/embedding probe) already in the pipeline. Then fit a single-parameter temperature or small multinomial logistic regression on vote fractions, with frame-grouped cross-validation (project rule 6). Avoid isotonic at our label count. K=3-5 is a budget trade-off against 15 RPM (Q5); vote fractions will be coarse (multiples of 1/K), so shrink toward the class prior.
- If logprobs do work on the chosen model, read the log-prob of the first answer token over the enum and temperature-scale it; test availability with one request before designing around it. The small-VLM preprint suggests this is the better signal where available, but it was on open 2B models.
- Use temperature 0 for the deterministic baseline and cache it; use higher temperature only for the sampled ensemble (cache each sample index separately, project rule 3).

### Gaps
- No Gemini Flash-Lite-specific calibration study found. Unverified whether logprobs are available on the target model and on the free tier. No measured effect of K samples on ECE for visual ordinal tasks.

## 4. Known VLM weaknesses on fill/volume/occupancy and fresh produce; effect of crop resolution, context crops, upscaling

### Takeaway
VLMs are weak at quantity/proportion judgments and lean on language priors; small or fine details are lost at native encoder resolution, and zooming/cropping helps if the crop is well aligned. I found nothing specific to fresh produce.

### Cited Findings
- VLM counting and quantity failures and reliance on priors: see Q3 citations (arXiv 2510.04401, 2511.17722, 2607.09544). Holistic benchmark argues existing counting benchmarks miss complex failure modes — [HoloCount, arXiv 2607.06420, Jul 2026](https://www.alphaxiv.org/abs/2607.06420)
- Most mainstream VLM encoders take about 224-336 px per side so fine detail disappears; cropping/zooming is a remedy — [CropVLM, arXiv 2511.19820, Nov 2025](https://arxiv.org/pdf/2511.19820)
- Zooming beyond native resolution can recover detail ViTs miss — [Dragonfly, arXiv 2406.00977, Jun 2024](https://arxiv.org/html/2406.00977v2)
- On V*, the hard step is locating the small object; once zoomed on the right region answering is easy; misaligned prompt-based crops hurt VQA accuracy — [SPARC, arXiv 2602.06566, Feb 2026](https://arxiv.org/pdf/2602.06566)
- Top-down imagery benchmark: cropping into smaller images raises relative pixel coverage of small targets — [TDBench, arXiv 2504.03748, Apr 2025](https://arxiv.org/pdf/2504.03748)
- Zero-shot VLM on shelves reportedly confuses empty spaces with low-stock (vendor blog, undated) — [Infosys](https://blogs.infosys.com/topaz/nvidia/enhancing-retail-store-analytics-with-video-ai-agent-real-time-video-analytics-part-3-3.html)
- Image degradation (severe low light) collapses accuracy and neither stated nor internal confidence flags it (small open VLMs, Jul 2026) — [arXiv 2607.22034](https://arxiv.org/abs/2607.22034). Relevant to our pre-opening / night-blind frames.
- Dedicated (non-LLM) out-of-stock detector reports AP 86.3% for fully empty shelves and 83.7% for frontal out-of-stock; empty is the easier class — [Sensors 2024 / PMC10819825](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10819825/)

### Inferences
- Our 256x256 crop is already below Gemini's 384 px single-tile threshold (258 tokens, Q5). Upscaling to e.g. 768 px would trigger tiling at 258 tokens per tile (higher cost) and its benefit for a native-256 source is unproven since upscaling adds no information; test on LOFO before adopting. Prefer sending the crop plus the wider context crop (context helps locate the area and judge neighbours) and tell the model which one is the target.
- Fresh produce risks I'd expect (not cited): piled, irregular produce makes "half" ambiguous; fill depends on crate depth and occlusion by perspective warp. Describe the half/full boundary visually in the prompt and use exemplars to anchor it (pending mentor answer).
- Add a "visibility/quality" flag in the schema (blind down, dark, occluded) and route flagged areas to the temporal/kNN prior, since neither VLM confidence signal reliably detects degraded images.

### Gaps
- Nothing on fresh produce or crate fill-level in VLM literature. No measured Flash-Lite effect of upscaling from 256 px.

## 5. Cost and rate-limit tactics (15 RPM, batching, caching, covering 289 areas in about 20 minutes)

### Takeaway
289 single-area requests at 15 RPM is 19.3 minutes, i.e. exactly at budget with zero retries; packing several areas per request (multi-image batching, shown to work for Gemini) brings it to a few minutes and leaves headroom for sampling/ensembling. Verify the actual free-tier limit on the project, since public figures conflict.

### Cited Findings
- Official rate-limit page (last updated 2026-10-09) gives no per-model free-tier table, says limits apply per project (not per key), RPD resets at midnight Pacific, and directs users to AI Studio for active limits. Batch API limits listed are Tier 1 only. — [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- Third-party guides disagree: ranges of 5-15 RPM and 100-1,000 RPD for unspecified models (Jan 2026); a report that 2.5 Flash free tier dropped to ~20 requests/day in Dec 2025 (user reports); a snippet of an official page cached earlier listed Gemini 2.0 Flash-Lite at 30 RPM / 1,500 RPD (unverified, may be outdated). Gemini 3.1 and 3.5 Flash-Lite are listed as free-tier models. — [yingtu.ai guide, updated 2026-09-30](https://yingtu.ai/en/blog/google-gemini-api-free-tier-limits-2026); [rate-limits page snippet](https://ai.google.dev/gemini-api/docs/rate-limits?authuser=1)
- Image tokens: images with both dimensions at or below 384 px cost 258 tokens; larger images are tiled at 768x768 with 258 tokens per tile; up to 3,600 images per request; `media_resolution` (Gemini 3) caps tokens per image; for single image plus text the docs advise putting the text before the image. — [Gemini image understanding docs (fetched Oct 2026)](https://ai.google.dev/gemini-api/docs/image-understanding)
- Gemini 3.1 Flash-Lite: 1,048,576 input tokens, context caching, Batch API supported — [model page, May 2026](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
- Multi-query batching: batching up to 50 queries per call gave minimal degradation or improvement under many-shot ICL, a single query per call was usually suboptimal; per-example latency fell about 10x (HAM10000) and 2x (TerraIncognita) zero-shot with 50-query batches (Gemini 1.5 Pro); many-shot per-example cost fell from $0.84 to $0.088 (350-shot) — [arXiv 2405.09798, May 2024](https://arxiv.org/html/2405.09798v2). This was Pro, not Flash-Lite, and on classification, not fill.
- Tier-1 Batch API (not free tier): 100 concurrent batch requests, 10M enqueued tokens for Flash-Lite variants — [rate limits page](https://ai.google.dev/gemini-api/docs/rate-limits)

### Inferences
- Token budget is trivial: 289 areas x (crop + context) x 258 tokens is about 150k image tokens for one pass; the binding constraint is requests per minute, not tokens. At a verified 15 RPM, one pass of 289 single-area requests uses ~19.3 min. Packing 4-6 areas per request (each labelled "Area k: expected product X") gives ~50-72 requests, about 3.5-5 minutes per pass, allowing 3-4 passes (samples or order swaps) inside 20 minutes.
- Batching risks: cross-area contamination and positional bias in lite models; keep batches small (4-6), shuffle order between passes, request one JSON entry per area id, and fail closed per area to the fallback (project rule 2). Compare batched vs single on LOFO before committing, since the 50-query result is on Pro.
- Cache on sha256 of provider, model, prompt version, image bytes, text (already a project rule); include sample index and image order in the key. Precompute all train-frame exemplar calls offline well before validation; only the 26 validation frames are time critical.
- Implement a token-bucket limiter at 15 RPM minus margin (e.g. 13) with exponential backoff on 429; run a 1-request test to measure the real limit and logprobs support.
- Optional: Claude vision as a fallback judge if a key appears; no data found comparing it with Flash-Lite on this task.

### Gaps
- Actual free-tier RPM/RPD for the specific Flash-Lite model is not verifiable from public docs (AI Studio page is account-specific); the 15 RPM figure comes from the task brief.
- No measured effect of multi-area batching on Flash-Lite fill accuracy.

## 6. Cost-sensitive decision-making under asymmetric stock errors

### Takeaway
With the given scoring, a calibrated three-class posterior plus an expected-value argmax is the correct decision rule; because every non-correct answer costs at least 2, the rule only predicts a class when its probability is roughly above two-thirds, and under high uncertainty it biases toward LOWER stock. This is derived from the scoring spec, not from literature.

### Cited Findings
- Scoring per project brief: correct +1; predicted lower than truth -2; predicted higher than truth -3. EV(empty) = p_e - 2(p_h+p_f); EV(half) = p_h - 3p_e - 2p_f; EV(full) = p_f - 3(p_e+p_h). — project CLAUDE.md (internal, not a public source)
- Calibration literature: decisions based on miscalibrated probabilities lose the benefit of EV rules, hence calibrate first (Q3 sources: [Xuan et al. 2025](https://arxiv.org/abs/2505.20236), [Groot 2024](https://arxiv.org/pdf/2405.02917)).

### Inferences (my algebra)
- Using p_e+p_h+p_f = 1: EV(empty) = 3p_e - 2; EV(full) = 4p_f - 3; EV(half) = p_h - 3p_e - 2p_f. Empty is positive only if p_e > 2/3; full only if p_f > 3/4 (full is the hardest to justify); half with p_e=0 is 3p_h - 2, positive only if p_h > 2/3.
- Uniform uncertainty (1/3 each): EV(empty) = -1, EV(half) = -4/3, EV(full) = -5/3, so ties break toward empty. Generally, doubt should push down the ladder (full to half to empty). A model overconfident about "full" is the costliest failure; calibrate p_f most carefully (e.g. one-vs-rest Platt on p_f) and check reliability diagrams per class.
- Compliance: symmetric +1/-2, so predict "not compliant" only if P(wrong) > 0.5, using a Platt-calibrated `matches_expected` (already in plan).
- The VLM's verbalized "unsure" should not be used as a hard override; feed vote fractions and flags into the calibrated fusion instead, then apply EV.

### Gaps
- No empirical estimate of how much EV the asymmetric rule gains over argmax of probability on our data; needs LOFO measurement.

## Source reliability notes
- Strong: the arXiv papers cited by ID and the Google Gemini docs (fetched pages). Several arXiv papers were read via abstract/search summary only; effect sizes quoted come from the fetched pages of 2405.09798, 2607.22034 and 2505.20236 (abstract only).
- Weak: yingtu.ai and other third-party Gemini limit guides (conflicting, unofficial); benchmarklist.com (secondary); Infosys blog (vendor, undated); calibrator sample-size rules of thumb (blogs, unsourced thresholds).
- Date caveat: arXiv IDs 2603-2607 are 2026 papers whose full text I did not read.
