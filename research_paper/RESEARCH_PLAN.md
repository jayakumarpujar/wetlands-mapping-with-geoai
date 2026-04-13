# WetMamba: Research Plan & Decision Rationale

**Authors**: Jayakumar Pujar, Qiusheng Wu
**Date**: 2026-04-13
**Status**: Pre-implementation planning

---

## 1. Problem Selection: Why Sub-Hectare Prairie Pothole Wetlands?

### Philosophy: "Target small problem, best solution ever"

We deliberately narrowed from "wetland mapping" (broad, saturated) to **sub-hectare prairie pothole delineation and Cowardin classification at 1-meter resolution**.

### Why this specific problem?

**Ecological urgency:**
- PPR contains millions of depressional wetlands — primary breeding habitat for North American migratory waterfowl
- These wetlands provide $3.2B/year in ecosystem services (flood attenuation, carbon sequestration, water filtration)
- 50-90% of PPR wetlands have been drained since European settlement
- NWI (National Wetlands Inventory) was created from 1980s aerial photo interpretation — 40 years outdated, misses newly formed wetlands, still labels drained ones

**Technical difficulty (what makes it publishable):**
- Sub-hectare targets (often <0.5 ha, some <0.1 ha) — invisible to Sentinel-2 (10m) or Landsat (30m)
- Requires 1m resolution (NAIP) — creates massive rasters (109K x 22K pixels) that break standard architectures
- Seasonal inundation (dry in summer, wet in spring) means single-date classification fails
- 6-class Cowardin taxonomy (not just binary wet/dry) demands fine-grained spectral + topographic discrimination
- No ground truth available at scale — must use weak labels from imperfect NWI

**Strategic advantages for our team:**
- Wu 2019 RSE studied the EXACT same area → direct published baseline comparison
- Wu created geoai package → we build natively on it, ensuring reproducibility
- We already have the full data pipeline (10-band composite, weak labels, training tiles) — 2,437 lines, 248 tests, HPC-validated
- PPR North Dakota is well-characterized hydro-geomorphically → reviewers can validate claims

### What we rejected and why

| Alternative problem | Why rejected |
|---|---|
| Global wetland mapping | Too broad, requires multi-sensor fusion across biomes — 3-year project |
| Sentinel-2 wetland mapping | Saturated space (Igwe 2026, many others). 10m can't see sub-hectare potholes |
| Binary wet/dry classification | Too simple for a top venue. 6-class Cowardin adds novelty + practical value |
| Change detection only | ChangeMamba already exists. We want classification + dynamics |
| Coastal wetlands | Different geomorphology, no depression prior, not Wu's domain |

**Decision**: PPR sub-hectare Cowardin classification at 1m. Small enough to execute in months. Hard enough that no one has solved it well. Directly comparable to Wu 2019.

---

## 2. Architecture Decision: Why Prithvi + Mamba + DAG?

### The landscape we surveyed

We researched 60+ papers (2024-2026) across four categories:

#### A. Foundation Models for Remote Sensing

| Model | Params | Pretraining | Key Result |
|---|---|---|---|
| **Prithvi-EO-2.0** (NASA/IBM, Dec 2024) | 600M | 4.2M HLS temporal scenes | 75.6% GEO-Bench (+8% over others) |
| SkySense (CVPR 2024) | 1B+ | 21.5M multi-modal sequences | 93.99% mF1 ISPRS Potsdam |
| SpectralGPT (TPAMI 2024) | — | 3D spectral pretraining | Strong on hyperspectral tasks |
| Clay v1.5 | — | 70M images, Apache-2.0 | Open-source alternative |
| DOFA (2024) | — | Dynamic multi-sensor | Good cross-sensor transfer |

**Key insight**: Prithvi-EO-2.0 is the clear choice because:
1. Best GEO-Bench score among open models
2. Has temporal encoding built in (critical for our multi-epoch NAIP)
3. ViT architecture → naturally produces multi-scale features for decoder
4. Open-weight on HuggingFace → reproducible
5. 300M and 600M variants → can ablate model scale
6. NASA/IBM backing → credibility in RS community

#### B. Mamba/SSM for Remote Sensing

| Model | Year | Architecture | Result |
|---|---|---|---|
| RS3Mamba | Apr 2024 | Dual-branch VSS+CNN | 82.78% mIoU Vaihingen |
| UNetMamba | Aug 2024 | CNN encoder + Mamba decoder | SOTA on LoveDA (6/7 classes) |
| CM-UNet | May 2024 | CNN encoder + Mamba decoder | Strong on building extraction |
| ChangeMamba | TGRS 2024 | Mamba for change detection | Beats CNN/Transformer on 5 datasets |
| **RoMA** | NeurIPS 2025 | Mamba FM scaling laws | Proves Mamba FMs outperform ViT FMs in efficiency AND accuracy |
| SatMamba | Feb 2025 | Mamba FM | 66.46% mIoU OpenEarthMap (beats ViTMAE) |

**Key insight**: Mamba is proven for RS segmentation but ALL existing papers use CNN encoders. No one has paired a **pretrained FM encoder** with a **Mamba decoder**. This is our architectural gap.

**Why Mamba over Transformer decoder?**
- O(n) complexity vs O(n^2) → critical at 1m resolution (our rasters are 109K x 22K = 2.4 billion pixels)
- RoMA (NeurIPS 2025) proved Mamba follows RS scaling laws AND outperforms ViT at same compute
- SSM naturally models sequential state transitions → perfect fit for temporal wetland dynamics
- UNetMamba already proved Mamba decoder works for RS segmentation — we extend it with FM encoder

#### C. Wetland-Specific DL Papers

| Paper | Architecture | Resolution | Classes | Best F1 |
|---|---|---|---|---|
| Igwe 2026 (RSASE) | U-Net++/DeepLabV3+ | 10m Sentinel-2 | 7 Cowardin | 91.3% |
| WetMapFormer 2023 | CNN+ViT local attention | 10m | Multi-class | 0.94-1.00 |
| Wet-ConViT 2024 | Conv attention + transformer | Various | Multi-class | ~95% OA |
| Biesbosch 2025 | SSL pretraining + fine-tune | HR aerial | Wetland types | 88.23% (from 60.35% w/o SSL) |

**Key insight**: NO wetland paper uses foundation models. NO wetland paper uses Mamba. This intersection is completely unclaimed.

#### D. Weak Supervision + Foundation Models

| Approach | Key Finding |
|---|---|
| PEFT/LoRA for geospatial FMs (2025) | LoRA matches full fine-tuning, better generalization |
| GeoSAM (ECAI 2025) | SAM fine-tuned with auto-prompts, +5% mIoU unseen regions |
| Biesbosch SSL (2025) | SSL pretraining on HR imagery: 60% → 88% accuracy (dramatic) |

**Key insight**: Foundation model pretraining + LoRA fine-tuning is the proven recipe. Our NWI weak labels + Prithvi pretraining is a natural fit.

### Architecture synthesis: How WetMamba was designed

Each component was chosen to fill a specific gap:

```
Problem requirement          →  Architecture choice         →  Justification
─────────────────────────────────────────────────────────────────────────────
Need pretrained RS knowledge →  Prithvi-EO-2.0 encoder     →  Best open FM, temporal encoding
Need O(n) for 1m rasters    →  Mamba SSM decoder           →  Linear complexity, proven for RS
Wetlands = depressions       →  DAG (Depression-Aware Gate) →  Physics prior, completely novel
Multi-epoch dynamics         →  Temporal SSM fusion         →  SSM = state transitions = phenology
Limited labels               →  LoRA + weak supervision     →  PEFT proven for geospatial FMs
```

**Nothing is added for novelty alone.** Each component solves a real constraint.

---

## 3. Depression-Aware Gating (DAG): The Core Novel Module

### The insight

Wu 2019 discovered that LiDAR-derived topographic depressions are the strongest predictor of wetland presence — wetlands physically cannot exist outside depressions in the PPR landscape.

In our current pipeline (and in Wu 2019), this is used as a **post-processing filter**: classify first, then mask out predictions outside depressions.

**Problem with post-processing**: The model wastes capacity learning to suppress false positives in non-depression areas. It never "knows" about topography during feature learning.

**Our solution**: Make the model depression-aware **during** feature extraction. The DAG module:

1. Takes depression depth map as auxiliary input
2. Passes through learned convolutions → spatial attention map
3. Multiplicatively gates decoder feature maps at each scale

```
Depression depth (H x W x 1)
    → Conv2d(1, C, 3x3) → BatchNorm → ReLU
    → Conv2d(C, C, 3x3) → BatchNorm → Sigmoid
    → gate ∈ [0, 1] for each spatial location
    → element-wise multiply with decoder features
```

### Why this is novel (we checked thoroughly)

- **Coordinate attention** (Hou et al. 2021): encodes positional info → our DAG encodes geomorphological info
- **Terrain-aware networks**: Some papers use DEM as input channel — but NO paper uses it as a learned gating mechanism
- **Physics-informed neural networks (PINNs)**: embed physics equations → our DAG embeds geomorphological constraint (wetlands ∈ depressions)
- **Attention mechanisms**: Self-attention learns what to attend to from data → DAG provides domain-specific spatial prior

**DAG is the bridge between Wu's geomorphological insight and modern DL architecture design.** It elevates a heuristic post-processing step into a learned, differentiable, end-to-end module.

### Why not just add depression as another input channel?

We already do that (band 10 in our composite). But:
- An input channel is processed identically to spectral channels — the network must learn from scratch that "depression = where wetlands can exist"
- DAG explicitly encodes this as a gating operation — multiplicative zero outside depressions, pass-through inside
- This is a much stronger inductive bias, requiring less data to learn
- **Ablation will prove this**: WetMamba w/o DAG (depression as input only) vs w/ DAG (gating)

---

## 4. Multi-Temporal SSM Fusion: Why State-Space for Phenology?

### The insight

Prairie pothole wetlands are dynamic:
- Spring: snowmelt fills depressions → maximum inundation
- Summer: evapotranspiration → many potholes dry up
- Fall: some refill from precipitation
- Permanent vs seasonal vs ephemeral — this distinction matters for ecology and regulation

Our 2-epoch NAIP (2015 Sep, 2017 Aug) captures two snapshots of this cycle. Currently, we concatenate these as channels (bands 5-8: NDVI/NDWI per epoch).

**Problem with channel concatenation**: Treats temporal signal as spatial features. The model has no notion of "this is time step 1, this is time step 2" — it must infer temporal relationships from correlation patterns.

### Why SSM fits perfectly

State-Space Models (SSMs) were designed to model sequential state transitions:

```
x(t+1) = A·x(t) + B·u(t)    ← state transition
y(t)   = C·x(t) + D·u(t)    ← observation
```

Wetland phenology IS a state transition system:
- State x(t) = wetland condition (inundated, vegetated, dry)
- Input u(t) = spectral observation at time t
- Transition A = seasonal dynamics (how wetlands change between epochs)

Mamba's selective SSM processes features from each epoch as a temporal sequence, naturally capturing:
- Which wetlands are **permanent** (state unchanged across epochs)
- Which are **seasonal** (state transitions between epochs)
- Which are **ephemeral** (brief inundation, may be missed)

### Why not a temporal Transformer?

- O(T^2) attention over T epochs — overhead for T=2 is small, but we design for extensibility to T=5+ epochs (Wu 2019 used 6 NAIP years: 2009-2017)
- SSM has built-in state memory — each epoch's representation is conditioned on all previous epochs via hidden state, naturally encoding temporal order
- Mamba's selective mechanism learns which temporal features to remember/forget — analogous to which spectral changes indicate real wetland dynamics vs noise

### Current design (2 epochs) → Future extensibility

With 2 epochs, temporal SSM advantage over concatenation may be modest. The ablation will quantify this honestly. But the architecture scales cleanly to 3-6 epochs without modification, which is the real value:

| Approach | 2 epochs | 6 epochs |
|---|---|---|
| Channel concatenation | 10 bands → 22 bands (messy) | 34 bands (impractical) |
| Temporal SSM | 2-step sequence (clean) | 6-step sequence (clean) |

---

## 5. Benchmark Strategy: Why These 8 Models?

### Design principle: Systematic comparison across paradigm generations

Each baseline represents a specific paradigm. Together they tell a complete story:

| Model | Paradigm | What comparison proves |
|---|---|---|
| Wu 2019 k-means | Unsupervised (2019) | DL >> unsupervised for this task |
| U-Net++ (ResNet-50) | CNN + skip connections (2018) | Current SOTA baseline for wetlands (Igwe 2026) |
| DeepLabV3+ (ResNet-50) | CNN + atrous convolution (2018) | Alternative CNN paradigm |
| SegFormer-B2 | Pure Transformer (2021) | Does global attention help for wetlands? |
| Swin-UNet | Hierarchical Transformer (2022) | Window attention for dense prediction |
| UNetMamba | CNN + Mamba decoder (2024) | Value of Mamba decoder alone (no FM) |
| Prithvi + linear head | FM encoder only (2024) | Value of FM pretraining alone (no Mamba) |
| **WetMamba (ours)** | FM + Mamba + DAG (2026) | Full system vs all individual components |

### Why these specific models and not others?

**Included:**
- U-Net++: Igwe 2026 showed it's best CNN for wetlands → fairest baseline
- DeepLabV3+: Second-best in Igwe → covers ASPP paradigm
- SegFormer: Clean ViT segmentation → proves transformer comparison
- Swin-UNet: Hierarchical windows → best transformer segmentation architecture
- UNetMamba: Published Mamba segmentation → isolates "what does FM encoder add?"
- Prithvi + linear: Published FM approach → isolates "what does Mamba decoder add?"

**Excluded (and why):**
- ResNet/VGG baselines: Too old, not competitive, waste of compute
- PSPNet/FPN: Similar paradigm to DeepLabV3+ → redundant
- SAM-based: Different task (promptable segmentation) → not directly comparable
- SkySense: Closed model, can't reproduce → unfair comparison

### Ablation design: Every component justified

| Ablation variant | Tests | Expected finding |
|---|---|---|
| w/o Prithvi (random init encoder) | Value of FM pretraining | Big drop (Biesbosch 2025 showed 60→88% from SSL) |
| w/o Mamba (ViT decoder instead) | Value of SSM vs attention | Modest improvement from Mamba + major efficiency gain |
| w/o DAG module | Value of depression gating | Meaningful drop, especially for small/ambiguous wetlands |
| w/o temporal SSM (concat channels) | Value of temporal modeling | Modest with 2 epochs, proves concept for >2 epochs |
| w/o LoRA (full fine-tune) | PEFT efficiency | Similar accuracy, much less memory → practical finding |
| w/o weak label filtering | Value of label pipeline | Significant drop — noisy NWI labels hurt training |

**Every ablation has a clear hypothesis and expected outcome.** No fishing.

---

## 6. Why LoRA (Not Full Fine-Tuning)?

### The evidence

- **PEFT for Geospatial FMs (2025)**: LoRA matches or exceeds full fine-tuning while enhancing generalization on out-of-distribution data
- **Practical**: Prithvi 600M has ~600M encoder params. Full fine-tuning requires storing full gradients → ~4.8 GB optimizer state alone
- LoRA adds ~2-4M trainable params (rank 16-32) → 99.5% fewer params to tune
- Prevents catastrophic forgetting of pretrained RS knowledge

### Our strategy

- Freeze Prithvi encoder
- Add LoRA adapters (rank=16) to attention Q/V projections
- Train only: LoRA adapters + Mamba decoder + DAG module + segmentation head
- Ablation: LoRA vs full fine-tune vs frozen (no adaptation)

---

## 7. Dataset & Evaluation Strategy

### Training data: PPR weak labels (existing pipeline)

- Source: NWI rasterized → depression filtered → temporal stability filtered → object confidence filtered
- ~100K tiles at 256x256 (1m/pixel = 256m x 256m coverage per tile)
- 6 classes: Upland, Water, Emergent, Forested, Scrub-Shrub, Other
- 60:40 train/val split, stratified by ecoregion

**No new data collection needed.** Entire weak label pipeline is built and tested (Phase 2 of existing repo).

### Evaluation metrics (standard for RS segmentation)

| Metric | Purpose |
|---|---|
| Overall Accuracy (OA) | Simple pixel-level correctness |
| Mean IoU (mIoU) | Class-balanced segmentation quality (PRIMARY metric) |
| Per-class F1 | Identifies which classes benefit most from each component |
| Per-class IoU | Standard RS segmentation metric |
| Precision/Recall per class | Distinguishes over-prediction vs under-prediction |

### Cross-dataset generalization (secondary benchmarks)

| Dataset | Why included |
|---|---|
| LoveDA (7-class, 0.3m) | Most relevant RS segmentation benchmark — urban/rural, similar resolution |
| ISPRS Vaihingen (6-class, 0.09m) | Classic aerial benchmark, very high res — stress-tests architecture |

**Note**: DAG module is PPR-specific (requires depression depth). For cross-dataset experiments, we remove DAG and test the Prithvi+Mamba core. This proves the architecture generalizes beyond wetlands.

---

## 8. Target Venue Analysis

### Primary: Remote Sensing of Environment (RSE)

| Factor | Assessment |
|---|---|
| Impact Factor | ~13.5 (top RS journal) |
| Wu's history | Wu 2019 published here → reviewer familiarity with PPR study area |
| Scope fit | Perfect — RS methodology + environmental application |
| Review time | 3-6 months typical |
| Competition | High, but novelty is strong (first FM+Mamba for wetlands) |

### Backup: IEEE TGRS

| Factor | Assessment |
|---|---|
| Impact Factor | ~8.2 |
| Scope fit | Strong — geoscience + RS methodology |
| Review time | 4-8 months |
| Advantage | More architecture-focused papers accepted here |

### Alternative: ISPRS J. Photogrammetry and Remote Sensing

| Factor | Assessment |
|---|---|
| Impact Factor | ~12.7 |
| Scope fit | Good — photogrammetry + RS |
| Advantage | Accepts longer papers with extensive experiments |

---

## 9. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Prithvi input bands don't match NAIP | Medium | High | Prithvi accepts variable input via projection layer; test early |
| Mamba decoder training unstable | Low | Medium | UNetMamba proved stability; follow their training recipe |
| DAG shows marginal improvement | Medium | Medium | Even small improvement is publishable if architecture is novel; honest ablation |
| Temporal SSM shows no gain with 2 epochs | High | Low | Expected — document as "architecture designed for >2 epochs" + show concat baseline |
| Weak labels too noisy for FM fine-tuning | Low | High | Already have 4-step filtering pipeline; Biesbosch 2025 showed SSL works with imperfect labels |
| Prithvi input resolution mismatch (HLS=30m, NAIP=1m) | Medium | Medium | Use Prithvi as feature extractor, not direct application; LoRA adapts representations |
| Reviewer says "just another Mamba paper" | Low | High | DAG module is genuinely novel (physics-informed gating); cross-dataset experiments show generalization |

---

## 10. What Makes This Paper Get Accepted

### Novelty checklist (reviewer perspective)

1. **First FM + Mamba for wetland mapping** — unclaimed intersection, verified via thorough literature search
2. **Depression-Aware Gating** — new module, no precedent in any RS or CV paper
3. **Temporal SSM for wetland phenology** — new application of SSM to ecological dynamics
4. **Weak supervision + FM** — proven separately, never combined for wetlands

### Rigor checklist (reviewer perspective)

1. **8-model comparison** spanning 4 paradigm generations (unsupervised → CNN → Transformer → FM+Mamba)
2. **7-variant ablation** isolating each component's contribution
3. **Cross-dataset generalization** on LoveDA + Vaihingen
4. **Direct comparison to Wu 2019** on identical study area
5. **Efficiency analysis** — FLOPs, params, inference time

### Practical impact checklist (reviewer perspective)

1. **Zero annotation cost** — NWI weak labels are free
2. **Reproducible** — open-source geoai, open-weight Prithvi, public NAIP/NWI/3DEP data
3. **Scalable** — Mamba O(n) enables 1m resolution at continental scale
4. **Directly applicable** — USFWS needs NWI updates, this provides methodology

---

## 11. Decisions (Resolved 2026-04-13)

1. **Prithvi variant**: Start with **300M**. Escalate to 600M only if 300M underperforms.
2. **Number of epochs**: Start with **2 (2015, 2017)**. Expand to 2009/2010/2012/2014 based on results.
3. **Secondary datasets**: **LoveDA + Vaihingen** post-training. Add OpenEarthMap/GEO-Bench if time permits.
4. **Paper length**: **Regular article (~8000 words)**. Document everything in research plan, rewrite for paper later.
5. **Code release**: **Same repo** (geoai extension). Restructure later if needed.
6. **Experiment tracking**: **JSON logs + WandB**. JSON for HPC, WandB for visualization.

---

## 12. Summary: The Story in One Paragraph

Prairie pothole wetlands are disappearing faster than we can map them. The national inventory is 40 years old. Existing ML approaches use 2018-era CNNs at 10m resolution — too coarse to see sub-hectare potholes. We propose WetMamba, the first architecture combining a geospatial foundation model (Prithvi-EO-2.0, pretrained on 4.2M Earth observation scenes) with a Mamba state-space decoder (linear complexity, scalable to 1m resolution) and a novel Depression-Aware Gating module that embeds geomorphological physics (wetlands only exist in topographic depressions) directly into the network architecture. Using free NWI weak labels refined by LiDAR depression filtering, we train WetMamba to classify 6 Cowardin wetland types at 1-meter resolution across the Prairie Pothole Region. Comprehensive benchmarks against 7 baselines spanning unsupervised, CNN, Transformer, and Mamba paradigms, plus 7-variant ablation studies, demonstrate that embedding domain physics into foundation model architectures achieves state-of-the-art wetland mapping without any manual annotation.
