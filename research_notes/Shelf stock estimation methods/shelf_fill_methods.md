# Shelf fill-level / on-shelf availability (OSA) estimation from fixed cameras

Context: 14 fixed ceiling cameras (1280x720), fruit and vegetable gondolas, ~1900 hand labels, empty/half/full per area, opencv/numpy/sklearn only by default. Research date: 2026-10-11. Source dates given inline. Searches were done in one session with about 20 tool calls; most findings come from abstracts or fetched pages summarized by a small model, so exact numbers should be re-checked in the primary paper before being quoted externally.

## 1. What do published and open methods do for shelf gap / empty-space detection (gap segmentation, IoU-style scores, depth-free occupancy)?

### Takeaway
Published OOS work is almost all supervised object detection (YOLO family) of "empty space" boxes on close-up shelf photos, scored with AP/mAP, with only a coarse two-class notion of emptiness (fully empty vs semi-empty). I found no published work that outputs a calibrated 3-level fill state for produce areas; fill percentage is instead obtained by occupancy grids (depth) or by pixel-ratio of product vs monitoring area (fixed camera, packaged goods).

### Cited Findings
- Sikic et al., Sensors, published 22 Jan 2024: 511 shelf images from four Croatian retailers, 1,210 labelled OOS instances (730 "normal" = empty front to back, 480 "front" = products removed from front, some remain at back). YOLOv5 gets AP 86.3% (normal) and 83.7% (front), mAP 85.0%; product-detection-based approaches only reach 56.9-63.8% AP on normal OOS. Two-stage training (mirroring augmentation pre-train, then fine-tune) adds about 1.6 mAP; aspect-ratio post-processing adds about 1.2-2.5 points. Authors list depth estimation as future work to cut errors; errors include poorly localized large empty areas. Dataset is restricted. — [PMC10819825](https://pmc.ncbi.nlm.nih.gov/articles/PMC10819825)
- OOS-DSD, arXiv 2510.16508, submitted 18 Oct 2025: YOLOv8 extended with OOS detection, product segmentation and depth branches (depth pseudo-labels from Depth Anything V2). Beats prior OOS detectors by 1.8% mAP; auxiliary learning adds 3.7% mAP and depth normalization 4.2% (as reported on the abstract page; dataset not named on the page I fetched). Shows monocular (depth-free hardware) depth as an auxiliary signal helps empty-space detection. — [arXiv 2510.16508](https://arxiv.org/abs/2510.16508)
- "Designing an Efficient End-to-end ML Pipeline for Real-time Empty-shelf Detection", arXiv 2205.13060, May 2022: about 1,000 annotated images, mean average F1 of 68.5%, 67 img/s on a Xeon CPU. Abstract does not define stocked/partial/empty classes. — [arXiv 2205.13060](https://arxiv.org/abs/2205.13060)
- Roboflow tutorials describe a public "Policenta" empty-shelf set of 1,050 images with one class (Empty-space) and classify shelves as stocked / partially stocked / empty by post-hoc thresholding of detected empty-space area (tutorial, not peer reviewed; date not captured). — [Roboflow blog](https://blog.roboflow.com/automate-on-shelf-monitoring/)
- Classical gap segmentation (Realogram patent family, older than 2021): pick the darkest superpixels under a shelf edge that do not belong to a recognized product as seeds, refine with GrabCut to label out-of-stock vs other. Useful as an idea only. — [USPTO 9483704](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9483704)
- Higa and Iwamoto (NEC), Sensors, 17 Jun 2019: ceiling-mounted fixed camera at 480x270, 1 fps; background subtraction finds changed regions, tracking removes shoppers, a CaffeNet classifier labels each change (taken / replenished / slight position change / false positive); OSA = ratio of product-occupied pixels to monitoring-area pixels. 89.6% of shelf-time samples within +/-0.10 error vs 64.5% for a conventional OOS method and 42.9% for counting all changes as removals. Accuracy varied per shelf (10 of 18 over 90%, worst 51.4%); errors accumulate over time, authors suggest periodic reset. Packaged beverages only. — [PMC6631981](https://pmc.ncbi.nlm.nih.gov/articles/PMC6631981/)
- Milella et al., IEEE Access 8:19353-19363, 2020: consumer depth camera, 3D point cloud, surface fitting, 2.5D occupancy grid, compared with a reference model learned in an initial training stage; targets perishable goods in countertop shelves, crates, baskets; max average discrepancy about 5.0% from true OSA on fresh products in a real retail setting (abstract only; I did not read the full paper). Requires depth hardware, which we do not have. — [DOAJ record](https://doaj.org/article/302301b555c1481b8a0ef957710e06f5); [CNR record](https://explore.cnr.it/resource/item/77679)
- An embedded planogram-compliance approach compares each grayscale frame against a stored reference "full shelf" image and triggers high-res capture when the number of changed pixels exceeds a threshold (reference-image change counting; found via search summary, original paper not opened). — [Marmara repository result](https://openaccess.marmara.edu.tr/bitstreams/0cbb49fe-df78-42b4-a176-7443845ecc78/download)

### Inferences
- Gap-detection detectors need box-level empty-space labels; our labels are area-level 3-class, so a detector is a poor fit. The usable transferable idea is "fraction of the area that is empty", which can be approximated as a continuous feature (empty-patch fraction) and then thresholded/learned against the 3 classes.
- The "half" class in Sikic's data corresponds to a different physical situation (back of a deep shelf still stocked) than half-full produce crates; "front" OOS AP is only about 2.6 points below "normal", so partial emptiness is detectable when it is spatially contiguous.
- Depth-free proxies reported to help: monocular depth as an auxiliary task (OOS-DSD) and pixel-occupancy ratio vs a monitoring area (NEC). Both can be approximated with pretrained features, but no source reports a gain for fixed ceiling cameras over produce.

### Gaps
- No peer-reviewed paper found that scores "intersection-over-product-area" for fill level specifically; the closest is NEC's occupied-pixel ratio and Roboflow's empty-area thresholding.
- Per-class (half vs full) confusion numbers were not reported by any source I found.
- Full text of Milella et al. not read, so error distribution and grid design unverified.

## 2. Change detection against a per-location reference and patch-level anomaly detection (AnomalyDINO, PatchCore, few-shot) with a per-area bank of "full" crops: use and reported gains over histogram-kNN?

### Takeaway
AnomalyDINO (WACV 2025) is a training-free, backprop-free patch kNN against a memory bank of nominal patches; it fits the "bank of full crops per area" idea mechanically, but I found no published application to shelves, no reported comparison to histogram-kNN, and its paper explicitly says it targets low-level defects not semantic changes. Reference-image change detection on shelves is well established but with packaged goods.

### Cited Findings
- AnomalyDINO (Damm, Laszkiewicz, Lederer, Fischer; arXiv 2405.14529 v3; WACV 2025): stores DINOv2 patch features of a few nominal images, scores each test patch by cosine distance to its nearest nominal patch, image score = mean of top 1% patch distances; default backbone ViT-S (about 21M params), no fine-tuning. — [arXiv HTML v3](https://arxiv.org/html/2405.14529v3); [WACV 2025 page](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)
- Image-level AUROC, MVTec-AD, 1/2/4/8/16-shot: AnomalyDINO-S(672) 96.6 / 96.9 / 97.7 / 98.2 / 98.4; PatchCore 83.4 / 86.3 / 88.8 (1/2/4-shot); SPADE 81.0 / 82.9 / 84.8; WinCLIP+ 93.1 / 94.4 / 95.2. On VisA: AnomalyDINO-S(672) 87.4 / 89.7 / 92.6 / 93.8 / 94.8 vs PatchCore 79.9 / 81.6 / 85.3. So DINOv2 patch features beat ImageNet-CNN patch features (PatchCore) by 8-13 AUROC points in the few-shot regime, on industrial data. — [arXiv HTML v3](https://arxiv.org/html/2405.14529v3)
- Resolution: detection AUROC similar at 448 and 672 but 672 improves localization, especially for small anomalies and complex scenes. Larger DINOv2 backbones (B/L/G) show no consistent gain over ViT-S. About 60 ms per image at 448 px, 1-shot, on an A40 GPU. Authors recommend coreset subsampling for larger banks. Variance is high at 1-2 shots and falls with more references. — [arXiv HTML v3](https://arxiv.org/html/2405.14529v3)
- Limitations stated by authors: targets low-level sensory defects; "Cable Swap" semantic anomaly detected at about chance (about 50% AUROC); uninformative references cause false positives; PCA foreground masking can fail when the object fills over half the patches. — [arXiv HTML v3](https://arxiv.org/html/2405.14529v3)
- Implementation: anomalib ships AnomalyDINO (timm `vit_small_patch14_dinov2`, kNN `num_neighbours` default 1, optional masking, optional coreset ratio 0.1, default input 252x252). It needs PyTorch/timm/Lightning; docs do not state CPU support. — [anomalib docs](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/anomaly_dino.html)
- Reference purity matters: "When More References Hurt" (arXiv 2608.22082, 22 Aug 2026), steel defects with DINOv2 memory banks: adding 0.5% anomalous patches to a clean bank dropped AUPRC from 0.1030 to 0.0759; naive expansion with unverified images has 9.46% anomalous patches and underperforms (0.0950) a contamination-trimmed bank at equal budget (0.1084). Fix: score candidates against a small clean seed bank, drop the worst 20%, greedy coreset. — [arXiv 2608.22082](https://arxiv.org/abs/2608.22082)
- PatchCore (Roth et al., CVPR 2022) is the origin of the patch-memory-bank approach; reported up to 99.6% image AUROC on MVTec AD with full training sets. — [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)
- Reference-image / background-subtraction change detection on shelves: NEC (2019, above) uses it with a learned change classifier; the gain of the learned classification step was large (89.6% vs 42.9% when every change is treated as a removal). Illumination changes are handled only as a "false positive" class, with no reported lighting-specific results. — [PMC6631981](https://pmc.ncbi.nlm.nih.gov/articles/PMC6631981/)
- Illumination-invariant background subtraction: a 2017 review (Kim and Jung, IEEE Access 5:8369-8384) catalogs methods; an arXiv paper notes GMM-style per-pixel models fail under significant illumination change while PCA-based ones are more robust; local ternary pattern backgrounds are another route. I did not open these in full. — [DOAJ record](https://doaj.org/article/8a60380a8b474c82a17b091eed6a7c99); [arXiv 1910.08470](https://arxiv.org/pdf/1910.08470)
- Searches combining PatchCore/DINOv2 anomaly detection with retail shelves returned no shelf application. — (negative result from searches for "PatchCore OR DINOv2 empty shelf")
- No head-to-head of DINOv2 patch features vs colour-histogram features was found. A July 2026 Reddit anecdote reports DINOv2-Giant global-embedding kNN at about 41% vs SigLIP2 about 92% on a 175-train fine-grained car set (single unreviewed data point, weak evidence). — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1uqtamz/snapshots/2026-07-09T20%3A50%3A09.14801Z)

### Inferences
- Adaptation: per area, build a bank of patch embeddings from labelled "full" crops (clean seed, perhaps one or two per daypart); score a new crop by the fraction of its patches whose nearest-bank distance exceeds a threshold ("missing product" fraction), rather than the top-1% mean used for defects. That fraction maps naturally to empty/half/full and is more informative than a single AUROC-style score. This is my adaptation, not a published result.
- Because a produce crate holds many near-identical items, "full" patches are highly repetitive, so a small bank should cover them; empty crate floors/liners are far from any "full" patch. Half is the hard case: it is a mix and depends on the threshold.
- Contamination finding suggests building the bank only from human-confirmed full labels, not from model-predicted ones; with 1900 labels this is feasible.
- A pretrained ViT-S/14 forward pass is a heavy dependency for the current stack (needs torch or an ONNX export); an opencv/numpy-only approximation (colour + texture patch descriptors, e.g. Lab means, LBP/HOG per grid cell, kNN vs full bank) would follow the same logic with no published gain figure.
- Expect no free lunch from AnomalyDINO's headline numbers: they are industrial single-object images with aligned, sparse anomalies. Fixed-camera crops from the same area are aligned, which is favourable, but time-of-day lighting changes are not covered by any reported benchmark.

### Gaps
- No reported accuracy gain of patch-bank methods over histogram-kNN on shelves or produce.
- No source evaluates AnomalyDINO/PatchCore under daylight/lighting drift.
- Whether anomalib AnomalyDINO runs acceptably on CPU is unverified (docs silent).
- I did not open the NEC paper's lighting experiments beyond the false-positive class; none were reported.

## 3. Fresh-produce specific findings (crates, bulk bins, irregular items, lighting through the day)

### Takeaway
Dedicated peer-reviewed work on produce fill level is sparse. The evidence that exists is depth-camera occupancy (about 5% error, Milella 2020), commercial deployments, and patents; no source reports a 3-class (empty/half/full) accuracy for produce from RGB.

### Cited Findings
- Milella et al. (IEEE Access, 2020): depth-based OSA for fresh products in crates/baskets/countertops, max average discrepancy about 5.0%, using a reference model learned at initialisation. — [DOAJ record](https://doaj.org/article/302301b555c1481b8a0ef957710e06f5)
- Macnica / Asus IoT supermarket trial (trade article; date not captured): cameras monitor fresh-food shelves, estimate stock level, set a per-shelf fill-level limit by product size and average daily sales; tested for several months. No accuracy given. — [inVision news](https://invision-news.de/sensorik-und-messtechnik/food-observation/)
- Lakeba SHELFIE (Azure Marketplace listing): percentage-of-shelf for fresh produce by comparison against a base image of the shelf (vendor claim, no accuracy). — [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/lakeba.shelfie)
- Patent family "Method for tracking and characterizing perishable goods in a store": estimates produce units in a display from a depth image and flags restock below a planogram minimum; also hyperspectral ripeness. Supporting evidence only. — [USPTO 10713621](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10713621)
- A refrigerator-produce tracking paper (NSF PAR) reports 74-85% per-class accuracy on apples and oranges in bins from one camera on a 301-frame video and notes that CNNs trained on large produce datasets often lose accuracy in real stores because of produce variability (small test; I did not open it). — [NSF PAR 10319207](https://par.nsf.gov/servlets/purl/10319207)
- AgriShelf dataset (Qatar University, v4 published 24 Apr 2025): 16,592 retail images of food/grocery products from smartphone and RealSense D435i, with varied lighting and angles; 2,416 labelled samples with centroid annotations for OSA estimation and counting; CC BY 4.0. No fill-level class labels listed. — [Mendeley AgriShelf](https://data.mendeley.com/datasets/s3vc2552sf)
- Beverage stock detection with fixed cameras, a CNN to reject obstructed images and YOLOv2 counting, over 98% accuracy (2024, abstract only; packaged goods). — [DOAJ record](https://doaj.org/article/35c287d2c47b4376b8df83cd6df23cbb)

### Inferences
- Occlusion/obstruction filtering (shoppers, hands) is a recurring component (NEC tracking, the beverage CNN gate); our system's temporal neighbours or a confidence gate play the same role.
- Produce piles are irregular and not countable per item, so published counting/detection pipelines (YOLO per item) do not transfer; area-level texture/coverage cues (fraction of crate floor visible) do.
- Lighting: no produce source quantifies daylight drift. The change-detection literature suggests illumination-robust representations (local texture, normalized chromaticity, learned features) over raw per-pixel intensity statistics.

### Gaps
- No produce-specific benchmark, no empty/half/full accuracy, no lighting-through-the-day study found.
- Vendor and patent items are claims, not validated results.

## 4. Datasets and benchmarks; what they say about half vs full ambiguity

### Takeaway
Large retail datasets target product detection/recognition, not fill level. The few shelf-emptiness sets are small, single-class (empty-space) and box-annotated; none provides a calibrated half/full label, so there is no external benchmark for the half-vs-full boundary.

### Cited Findings
- SKU-110K (Goldman et al., CVPR 2019): dense retail scenes, one "retail item" class, 8,233 train images / 1,210,431 boxes and 2,941 test images / 432,312 boxes per the paper (a dataset catalog lists different totals); introduced a Soft-IoU layer for precise detection. Predates the 2021 window and has no stock-level labels. — [arXiv 1904.00853](https://arxiv.org/pdf/1904.00853); [datasetninja](https://datasetninja.com/sku110k)
- RPC (Wei et al., arXiv 1901.07249, Jan 2019; journal version Science China Info. Sci. 2022): 200 categories, 83,739 images, for checkout recognition; not shelf fill. — [arXiv 1901.07249](https://arxiv.org/pdf/1901.07249)
- Grozi-3.2k: 3,235 images of Swiss retail shelves with reference product images (product recognition). Surveys: Melek et al. 2024 (grocery shelf product recognition datasets and methods); Santra and Mukherjee 2019; Wei et al. 2020. — [arXiv 2411.10591](https://arxiv.org/html/2411.10591v1); [Bilgi repository](https://openaccess.bilgi.edu.tr/items/ec2407dc-1ce8-42af-9a62-d157cfb1f1cf/full)
- Empty-shelf sets: Policenta (Roboflow, 1,050 images, class "Empty-space"); Sikic et al. 2024 (511 images, restricted access); AgriShelf (2025, centroids, no fill level). — [Roboflow](https://blog.roboflow.com/automate-on-shelf-monitoring/); [PMC10819825](https://pmc.ncbi.nlm.nih.gov/articles/PMC10819825); [Mendeley](https://data.mendeley.com/datasets/s3vc2552sf)
- A 2021 Sensors paper notes the largest computer-vision retail datasets do not annotate store products in the way OSA needs, so labelling effort is large (as summarised by search tool; original not opened). — [DOAJ record](https://doaj.org/article/36876bfa69cc458c8aa2bc66d941bc8d)
- In Sikic et al., the semi-empty class ("front") is defined by physical position (front removed, back remains) and detected at AP 83.7 vs 86.3 for fully empty, i.e. partial emptiness with a spatial structure is only modestly harder. Authors' errors include poorly localized large empty areas and hanging products at shelf edges. — [PMC10819825](https://pmc.ncbi.nlm.nih.gov/articles/PMC10819825)

### Inferences
- Half vs full is an annotator-boundary problem; none of the benchmarks measure inter-annotator agreement on it. With our asymmetric scoring (predict lower than truth -2, higher -3), the practical lever is calibrated probabilities plus the EV rule, not a better backbone alone.
- Our own ~1900 labels are likely the largest fill-level-labelled produce set available; external datasets are useful only for pretraining features, not for the half/full boundary.

### Gaps
- No inter-annotator agreement or confusion matrix for partial vs full stock in any source.
- SKU-110K/RPC headline AP/checkout numbers were not captured and are not relevant to fill level.

### Inferences: which methods work without training a large network
- Reference-image change features (per-area "full" reference, changed-pixel fraction, edge/texture difference, NEC-style occupancy ratio): numpy/opencv only; published evidence is on packaged goods (NEC 2019: 89.6% within +/-0.10).
- Pretrained patch features + memory-bank kNN (AnomalyDINO style): training-free but needs a ViT-S/14 forward pass (torch/timm, or exported model); strongest few-shot evidence (+8-13 AUROC points over PatchCore on MVTec/VisA) but only on industrial data. Keep the bank clean (arXiv 2608.22082).
- Monocular depth (Depth Anything V2) as an auxiliary signal: +3.7-4.2 mAP in a trained detector (OOS-DSD, Oct 2025); needs a network and training, so likely out of scope.
- Supervised YOLO empty-space detectors: best AP (about 85) but need box labels we do not have.
