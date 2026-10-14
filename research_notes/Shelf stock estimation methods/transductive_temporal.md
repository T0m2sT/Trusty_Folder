# Transductive / semi-supervised learning and temporal state-space modelling for fixed-camera shelf sequences

Context: 14 fixed cameras, ~180 labelled-or-labellable frames from one day, 26 unlabelled later-evening test frames (289 areas). Per-area stock in {empty, half, full}, asymmetric scoring. Libraries: numpy, scikit-learn, opencv. Source dates are given inline. Many items below come from search snippets or abstracts (flagged "abstract only" where I did not open the paper); general ML papers are not shelf-specific, and I found NO peer-reviewed paper on semi-supervised or temporal stock-level estimation for fixed shelf cameras.

## Q1. Transductive and semi-supervised options for small data (self-training, label propagation, co-training, consistency regularisation): typical gains and failure modes

### Takeaway
Confidence-thresholded pseudo-labelling (FixMatch family) and graph label spreading are well documented, but the headline gains come from image-classification benchmarks with hundreds to thousands of unlabelled images and trainable deep nets. With ~26 test frames, the realistic transductive tools are (a) label spreading with a kNN kernel and (b) a thresholded self-training loop on a light classifier, both with the confirmation-bias and class-imbalance failure modes below. Co-training is only sound if the two views (VLM vs classical features) are roughly conditionally independent, which is plausible but unproven here.

### Cited Findings
- FixMatch (Sohn et al., arXiv Jan 2020, v2 Nov 2020, NeurIPS 2020) keeps a pseudo-label from a weakly augmented image only if confidence is high, then trains the model to predict it on a strongly augmented view; reports 94.93% on CIFAR-10 with 250 labels and 88.61% with 40 labels — [arXiv 2001.07685](https://arxiv.org/abs/2001.07685)
- Confirmation bias: naive pseudo-labelling overfits to its own wrong pseudo-labels; mixup and forcing a minimum number of true-labelled samples per mini-batch reduce it (Arazo et al., arXiv Aug 2019, IJCNN 2020; abstract only; results are the authors' own) — [arXiv 1908.02983](https://arxiv.org/abs/1908.02983v5)
- In FixMatch, keeping true labels in the mix (over-sampling labelled data) matters to avoid accumulating confirmation errors, mostly early in training (NLDL 2022 analysis; snippet only) — [arXiv 2201.00604](https://arxiv.org/pdf/2201.00604)
- With confusable classes and tiny label budgets, FixMatch can be no better than plain supervision; over 40% of above-threshold pseudo-labels were wrong up to a budget of 200 labels (2023 study; snippet only) — [arXiv 2308.08224](https://arxiv.org/pdf/2308.08224)
- A reimplementation study found confirmation errors especially from asymmetric pseudo-label noise, and mitigated them with entropy regularisation (snippet only) — [Papers with Code, Re: FixMatch](https://astro.paperswithcode.com/paper/re-reimplementation-of-fixmatch-and)
- Single global confidence threshold is suboptimal: FlexMatch (NeurIPS 2021) uses per-class thresholds tied to learning status (13.96% and 18.96% error reduction over FixMatch on CIFAR-100 and STL-10 with 4 labels/class); FreeMatch (2022) adds a class-fairness penalty for diverse early predictions. Neither was verified on class-imbalanced data — [FlexMatch arXiv 2110.08263](https://arxiv.org/pdf/2110.08263), [FreeMatch arXiv 2205.07246](https://arxiv.org/pdf/2205.07246v2)
- CLIP-style zero-shot pseudo-labels are class-biased, so Unsupervised Prompt Learning (2022) picks top-K per class instead of a global threshold, to avoid imbalanced pseudo-label sets — [arXiv 2204.03649](https://www.arxiv.org/pdf/2204.03649v1)
- Prototype-guided pseudo-labelling (2025) scores zero-shot VLM pseudo-label correctness via prototype and neighbourhood consistency and weights samples accordingly; CoFT (Feb 2026) uses a dual-model, cross-modal setup to judge pseudo-label quality without hand-set thresholds (abstracts only) — [arXiv 2507.22075](https://arxiv.org/pdf/2507.22075), [arXiv 2602.04337](https://arxiv.org/abs/2602.04337v1)
- Realistic SSL evaluation (Oliver et al., NeurIPS 2018): performance can degrade substantially when unlabelled data has out-of-class examples; SSL is sensitive to unlabelled-set size and class distribution; realistically small validation sets cannot reliably compare methods/hyperparameters — [arXiv 1804.09170](https://arxiv.org/pdf/1804.09170)
- Generative/EM SSL (Nigam et al., Machine Learning 2000): unlabelled data cut error by up to 30% on text tasks, but plain EM works well only when data match the model's generative assumptions; fixes were down-weighting unlabelled data and multiple mixture components per class — [CMU RI](https://www.ri.cmu.edu/publications/text-classification-from-labeled-and-unlabeled-documents-using-em)
- Co-training (Blum & Mitchell, COLT 1998): two learners on two views, each labelling data for the other; proven under conditional independence of views given the class; Abney (2002) shows a weaker assumption suffices — [CMU RI](https://www.ri.cmu.edu/publications/combining-labeled-and-unlabeled-data-with-co-training), [Abney](https://cs.cornell.edu/courses/cs775/2002fa/papers/abney-bootstrapping.pdf)
- scikit-learn (docs for 1.14): `LabelSpreading` uses a normalised graph Laplacian with soft clamping (alpha=0.2 lets 20% of the original label distribution change), and is more noise-robust than `LabelPropagation` (hard clamp); `knn` kernel is sparse and cheap, `rbf` is dense and can be prohibitively slow; unlabelled points are marked -1; gains require data structure to match class structure — [scikit-learn semi-supervised guide](https://scikit-learn.org/stable/modules/semi_supervised.html)
- `SelfTrainingClassifier` (Yarowsky-style) wraps any `predict_proba` estimator, selects by `threshold` or `k_best`, and its results depend on the base classifier's calibration — [scikit-learn semi-supervised guide](https://scikit-learn.org/stable/modules/semi_supervised.html)
- LabelSpreading originates in Zhou et al., "Learning with Local and Global Consistency" (NIPS 2003 / proceedings 2004); the search did not itself confirm the "label spreading" name link — [NIPS paper page](https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency)
- Transductive fine-tuning (Dhillon et al., ICLR 2020) beat prior few-shot SOTA on several benchmarks, but the authors note much of the gain came from a large backbone and should be compared on matched architectures — [arXiv 1909.02729](https://arxiv.org/abs/1909.02729)

### Inferences
- Stock level is ordinal and heavily class-imbalanced in this setting (likely many "full", few "empty" in the evening), so a single global threshold will pseudo-label mostly the majority class; per-class top-K or per-class thresholds (FlexMatch/UPL idea) is the safer variant.
- With only 26 test frames, FixMatch-style augmentation training of a deep net is not practical; the viable form is (1) kNN-kernel `LabelSpreading` over crop embeddings of train plus test crops, and (2) one or two rounds of self-training a logistic regression/GB on frozen embeddings, with the VLM output as a second view.
- A VLM plus a pixel/embedding classifier is a plausible co-training pair (different failure modes), but conditional independence is unlikely to hold exactly (both see the same pixels); use it as agreement-gating (add a test crop to the pseudo-labelled set only if both agree) rather than as an iterative loop.
- Because the scoring is asymmetric (over-predicting stock costs -3), pseudo-labels should be soft and pass through the existing calibrated EV decision rule, not be hard-fed back as truth; down-weight unlabelled contributions as in Nigam et al.
- Evaluation must be frame-grouped (LOFO), consistent with Oliver et al.'s warning that small validation sets make SSL gains hard to verify; treat any gain under about one standard error of the LOFO score as noise.

### Gaps
- No source quantifying SSL gains with ~10^2 unlabelled images and 3 ordinal classes; all reported numbers are CIFAR/STL-scale.
- Label spreading failure modes on kNN graphs with few labelled nodes per cluster were not found beyond the scikit-learn caveat; no empirical imbalance study found.
- Consistency regularisation beyond FixMatch (Mean Teacher, UDA) not researched in detail.

## Q2. Per-area clustering of unlabelled crops across the day to discover full/empty prototypes

### Takeaway
No retail-specific source found. The generic supporting evidence is that prototype/neighbourhood consistency is a useful way to score pseudo-label correctness, and that mixture-model SSL helps only when cluster structure matches classes (multiple components per class helps). Per-area clustering is therefore a low-cost, plausible feature but unvalidated by literature for shelves.

### Cited Findings
- Prototype and neighbourhood consistency used to estimate pseudo-label accuracy and weight samples for VLM zero-shot adaptation (2025; abstract only) — [arXiv 2507.22075](https://arxiv.org/pdf/2507.22075)
- Dual-prototype approach (DPA) aligns textual and visual prototypes and reports gains over zero-shot CLIP on 13 tasks (snippet only) — see search summary in [arXiv 2509.18938](https://arxiv.org/html/2509.18938v1) (related work)
- A lightweight classifier trained on CLIP-selected high-confidence test samples, using features from a separate pre-trained visual model, iterated in a loop, with no VLM fine-tuning (ICTAI 2025; abstract) — [arXiv 2509.18938](https://arxiv.org/html/2509.18938v1)
- Mixture-model SSL needs the cluster structure to match the classes; multiple mixture components per class and an unlabelled-data weight fix mismatch (Nigam et al., 2000) — [CMU RI](https://www.ri.cmu.edu/publications/text-classification-from-labeled-and-unlabeled-documents-using-em)
- scikit-learn notes SSL gains depend on unlabelled structure matching class structure — [scikit-learn guide](https://scikit-learn.org/stable/modules/semi_supervised.html)
- Retail patent literature treats the fixed shelf view as a background model against which foreground changes are flagged; if the planogram changes the background model may relearn the layout (US 12136061; patent, vendor-claim quality) — [USPTO 12136061](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12136061)

### Inferences
- Because cameras are fixed and each area is a stable perspective quad, within-area appearance variance is dominated by stock level, lighting, and product swaps. Within-area k-means (k=2-3) or a 2-3 component GMM on rectified-crop embeddings, over all ~194+ frames of that area (train plus test, transductive), should expose a "dense product" and an "empty shelf/background" mode. Cluster membership, distance to each centroid, and the cluster's mean labelled stock (from the few labelled members) are usable features.
- Anchor cluster identity with labels, not by cluster size: map clusters to levels using the labelled frames in that area (as in "multiple components per class"). Unanchored clusters risk matching lighting, not stock.
- Cheap robust variant: treat the empty-shelf prototype as the per-area minimum-texture mode (e.g. the frame(s) labelled empty, or pre-opening frames) and use distance-to-empty-prototype as a continuous fill proxy.
- Clusters may split by product swap (e.g. bananas vs oranges in 6215_7) rather than stock; fitting on pixel-normalised embeddings per product-identity segment avoids that.

### Gaps
- No paper found validating per-area clustering for stock estimation; no gain estimates.
- No source on choosing k or on GMM vs k-means for few (~14-30) samples per area.

## Q3. Temporal models: HMM/Kalman smoothing, forward-backward vs causal, and what temporal information exists with 26 sparse test frames

### Takeaway
Kalman/RTS and HMM forward-backward both handle missing or irregularly spaced observations naturally (the update is skipped when no observation exists, the state predicts forward), so sparse frames are not a barrier. But with 1-2 test frames per camera the "smoothing" benefit is mostly a prior from the last labelled frame earlier in the day: effectively a single prediction step with large uncertainty, and only causal (forward) filtering is possible for a test frame later than all training frames.

### Cited Findings
- Kalman filtering and smoothing work with missing observations: missing data zeroes the gain, and if all observations are missing the posterior equals the prior (arXiv 1903.11309, Dynamic Linear Models intro, March 2019) — [arXiv 1903.11309](https://arxiv.org/pdf/1903.11309)
- Standard formulas assume uniform sampling; non-uniform spacing requires propagating uncertainty over the actual time gap (cites Harvey 1991 and Durbin & Koopman 2012) — [arXiv 1903.11309](https://arxiv.org/pdf/1903.11309)
- VISTA-SSM (Oct 2024) uses Kalman filtering with RTS smoothing for varying/irregular sampling; the RTS backward pass runs after the forward pass using filtered and predicted covariances — [arXiv 2410.21527](https://arxiv.org/pdf/2410.21527)
- RTS smoother assumes a linear time-invariant model; nonlinear systems need EKF/UKF variants (MetricGate docs) — [MetricGate Kalman smoother](https://metricgate.com/docs/kalman-smoother/)
- HMM forward and backward variables, likelihood from either pass, Viterbi for best state path, and Baum-Welch for parameters are defined in Rabiner's 1989 Proc. IEEE tutorial — [Rabiner 1989 PDF](https://people.engr.tamu.edu/rgutier/web_courses/cpsc689_s07/rabiner1989tutorialHMM.pdf)
- Conceptual HMM-Kalman equivalence (discrete vs linear-Gaussian latent chain) was NOT confirmed by the sources I retrieved.

### Inferences
- Use a 3-state discrete HMM per area (empty/half/full) with an irregular-time transition matrix A(dt) built from a continuous-time generator (e.g. matrix exponential of a rate matrix) rather than a Kalman filter: the target is ordinal/discrete, and the scoring needs class probabilities for the EV rule. The generator is nearly upper-triangular in the "depleting" direction (full to half to empty) plus a small restock rate (empty/half to full), estimated from the labelled train sequence with a Dirichlet/pseudo-count prior since data are tiny (12-14 frames per camera).
- Emission model: per-frame classifier probabilities (VLM, embedding probe, kNN) converted to likelihoods by dividing by the class prior (hybrid HMM; standard practice, not sourced here). Train-frame emissions can be replaced by hard labels when present (observed), unlabelled train/test frames get soft emissions.
- Only filtering (forward) is meaningful for the test frames since all labelled frames precede them; forward-backward helps only for unlabelled intermediate frames (the unlabelled-train/test frames lying between labelled ones), which can in turn be used as extra pseudo-labelled data. If the test frames of a camera fall in a gap of the labelled series, the posterior mixes both ends (smoothing); otherwise it is a pure prediction from the last labelled state, widening with elapsed time dt, so the temporal prior should be gated by dt (the project already uses a visual-first temporal gating; this is consistent).
- With dt of hours and restock jumps, the transition prior is weak for far-apart frames and its stationary distribution (not the last label) is the right fallback. Estimating the stationary distribution per area from all labelled frames of that area is a cheap, stable prior.
- The time-of-day effect (more depletion in the evening, store closing) can be added as a time-inhomogeneous hazard (rate as a function of hour); with ~14 frames per area, share the hazard across areas of the same camera or product category.

### Gaps
- No retail/inventory paper found on HMM or state-space models for shelf stock levels; no empirical gain numbers for sparse hourly sampling.
- Equivalence/trade-off between HMM forward-backward and RTS not confirmed from fetched sources.
- Left-to-right (monotone) HMM structure: the retrieved Rabiner copies did not confirm discussion of it in this search.

## Q4. Event detection: depletion vs restock vs product replacement vs occlusion from image differences; change magnitude as a feature

### Takeaway
No peer-reviewed shelf-specific method found. Classical illumination-invariant change detection and background modelling give the building blocks (normalised/linear-dependence tests, phase correlation, level lines, block-DCT energy), and retail patents describe flagging foreground changes against a stored shelf background. Separating event types is mainly a matter of combining change magnitude with product-identity and stock-level predictions of the two frames.

### Cited Findings
- US 12136061 treats the shelf as a background model, flags foreground detections as possible out-of-stock events or planogram changes, and may relearn the background on a planogram change (patent) — [USPTO 12136061](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12136061)
- Focal Systems patent family (10949799 / 11763254 / 12229718): shelf-mounted cameras with a product-detection model for out-of-stock detection (patents; vendor-claim quality) — [Google Patents US10949799](https://patents.google.com/patent/US10949799)
- Roboflow guide: detector-based per-frame counting is a snapshot of shelf condition, not a temporal change method (vendor blog) — [Roboflow](https://blog.roboflow.com/retail-object-detection/)
- Illumination-invariant change detection: multiplicative illumination model plus a statistical colinearity test, with an MRF/Bayesian prior to cut spurious detections (Mester, Aach, Dümbgen; 2001-era paper) — [Uni Lübeck](https://research.uni-luebeck.de/en/publications/illumination-invariant-change-detection-using-a-statistical-colin/)
- Level-line change detector invariant to directed lighting for Lambertian piecewise-developable surfaces, about 85% detection at 10% false positives on easy scenes (Weiss et al., SIIMS 2011) — [Weiss 2011 PDF](https://www.math.univ-toulouse.fr/~weiss/Publis/SIIMS_2011_Weiss.pdf)
- Shadow variation from changing sun angle can overwhelm real changes; phase correlation proposed for illumination-robust change detection (ISPRS Archives 2018) — [ISPRS 2018](https://isprs-archives.copernicus.org/articles/XLII-3/1673/2018/)
- Block frequency-energy comparison (DCT) for scene-change detection (Siemens patent US7499570) — [Google Patents US7499570](https://patents.google.com/patent/US7499570)

### Inferences
- Features per area pair (frame t vs nearest earlier frame t'): (a) change magnitude after per-area photometric normalisation (e.g. 1-NCC or gradient/edge-orientation difference, computed with OpenCV on the rectified crops), (b) the change in per-frame stock probabilities, (c) the change in product-identity embedding similarity, (d) fraction of changed pixels in lower vs upper part of the quad (depletion tends to expose the back/bottom).
- A rule-of-thumb event typing using these: high change plus product-embedding drift plus stock unchanged gives replacement or planogram change; high change plus stock drops gives depletion; high change plus stock rises gives restock; high local change with a large compact mask that vanishes in the next nearby frame gives occlusion (person/trolley). Only the last needs a frame after the test frame, which is unavailable for test frames, so occlusion must be flagged from a single frame (person/occluder detector or abnormal low-texture/skin-tone blob) rather than from temporal persistence.
- Treat the "change from last labelled frame" as a gate on the temporal prior: if change magnitude is small, trust the previous state strongly; if large and unexplained, fall back to the per-frame classifier.
- Since evening illumination differs from the labelled day frames, the change signal must be computed on illumination-normalised crops or it will mostly measure lighting (ISPRS caution).

### Gaps
- No benchmark or paper reporting accuracy of event-type classification on shelf images.
- No SSIM-based shelf change-detection paper found; no recent survey of lighting-robust change detection retrieved.

## Q5. Test-time adaptation to late-evening lighting drift (per-camera normalisation using unlabelled test frames)

### Takeaway
The best-supported cheap technique is re-estimating normalisation statistics on the target (test-time) data: it gave large robustness gains with as little as 32 samples, and even one sample helped. Entropy-minimisation TTA (Tent) is fragile with small batches, mixed shifts and imbalanced labels, so for 26 images prefer statistic re-estimation or simple per-camera photometric normalisation over gradient-based TTA.

### Cited Findings
- Schneider et al. (NeurIPS 2020, arXiv Jun 2020): replacing training BN statistics with target-data statistics improved ResNet-50 mCE on ImageNet-C from 76.7% to 62.2%; works with little data (even a single sample helped; 32 samples enough to improve the then-SOTA for ResNet-50) — [arXiv 2006.16971](https://arxiv.org/abs/2006.16971)
- Tent (Wang et al., ICLR 2021, arXiv Jun 2020): minimises prediction entropy at test time, updating only BN affine parameters; 44.0% error on ImageNet-C vs 49.9% for test-time normalisation alone — [arXiv 2006.10726](https://arxiv.org/pdf/2006.10726)
- Tent failure modes: collapse to predicting one class (2023 follow-up), instability with small batches, mixed distribution shifts, and online imbalanced label distributions (2023 and 2025 follow-ups; snippets only) — [arXiv 2302.12400](https://arxiv.org/pdf/2302.12400), [arXiv 2509.04977](https://arxiv.org/html/2509.04977v1)
- Batches that are too small can give poor distribution estimates when BN statistics are re-estimated per batch (ICCV 2023 challenge report; snippet only) — [arXiv 2310.13533](https://ar5iv.labs.arxiv.org/html/2310.13533)
- MEMO, SAR and EATA extend Tent with augmentation-based single-sample adaptation, sharpness-aware minimisation, sample filtering and anchoring (snippet only) — [arXiv 2006.10726 follow-up discussion via arXiv 2509.04977](https://arxiv.org/html/2509.04977v1)
- Phase-correlation-based change detection proposed for robustness to shadows from changing sun angle (ISPRS 2018) — [ISPRS 2018](https://isprs-archives.copernicus.org/articles/XLII-3/1673/2018/)

### Inferences
- Because the model is on frozen embeddings plus a light classifier, the BN-statistics idea maps to feature-space alignment: per camera (or per area), subtract the mean (and optionally divide by the std) of embeddings computed over that camera's frames, then fit on train-time-normalised features. Using the camera's own unlabelled test frames in the mean is permissible since transduction is allowed, but a time-local mean (late-day train frames plus test frames) is better than an all-day mean because it carries the evening illumination while the all-day mean contains stock-level drift too. Risk: normalising away real stock change; limit to colour/illumination statistics (per-channel gain, gamma) estimated from static regions of the frame (structure, shelf frame, floor) rather than from the stocked product area.
- A classical alternative is per-camera gain/white-balance matching with OpenCV (grey-world or histogram matching on a static reference patch), which needs no training and no labels; validate by checking that LOFO score on the last 2-3 frames per camera (evening proxy) improves.
- Avoid Tent-style entropy minimisation here: 26 images, imbalanced stock classes, and possible one-class collapse.

### Gaps
- No evening-lighting or retail-camera-specific TTA study found.
- Effect size for per-camera feature standardisation on small embeddings not found; needs empirical LOFO check.
- Non-BN backbones (ViT/CLIP-style, layer norm) are less amenable to BN-statistic adaptation; the sources do not give a verified recipe for them.

## Source-date index (for the writer)
FixMatch Jan 2020 (NeurIPS 2020); Arazo Aug 2019; FlexMatch Oct 2021; FreeMatch May 2022; UPL Apr 2022; Oliver et al. Apr 2018; Nigam et al. 2000; Blum and Mitchell 1998; Abney 2002; Zhou et al. NIPS 2003/2004; Rabiner 1989; Schneider Jun 2020; Tent Jun 2020 (ICLR 2021); Dhillon Sep 2019 (ICLR 2020); arXiv 1903.11309 Mar 2019; VISTA-SSM Oct 2024; prototype-guided pseudo-labelling Jul 2025; "No Labels Needed" 2025; CoFT Feb 2026; scikit-learn docs (v1.14 stable page, accessed Oct 2026); retail patents (various, vendor-quality).
