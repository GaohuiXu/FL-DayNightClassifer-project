# Near-Term Research Roadmap — Analysis & Recommendations

## Your Three Ideas — Assessment

---

### Idea 1: Representation-space analysis of backdoors

**Verdict: Strong research direction. This is the most thesis-relevant of the three ideas.**

Your intuition is well-supported by literature:

- **Activation Clustering** (Chen et al., 2019) — backdoored models produce distinguishable activation patterns for triggered vs clean inputs
- **Spectral Signatures** (Tran et al., 2018) — backdoor activations have separable spectral properties in the penultimate layer
- **Feature-space trojans** — recent work shows backdoors occupy a low-rank subspace in representation space

The progression you describe is sound:
1. Extract hidden representations for clean, triggered, and poisoned inputs
2. Visualize with t-SNE/UMAP — do triggered inputs cluster differently?
3. Compare across attack types (pixel trigger vs DBA vs model replacement)
4. If common patterns emerge → design attacks that directly manipulate representations

**Why this matters for your thesis**: It transitions from "apply existing attacks and measure ASR" to "understand *why* attacks work mechanically." This is the kind of analysis that turns engineering baselines into research contributions. It also connects directly to your CLAUDE.md hypothesis #1 (move toward optimization-based attacks) — an attack designed in representation space IS an optimization-based attack.

**What you'd need on the platform side**:
- A way to extract intermediate features from the model (the current CNN `forward()` doesn't expose them)
- Analysis utilities: feature extraction, dimensionality reduction, visualization
- These should live in a separate `analysis/` module — not inside the training loop

**Risk to watch**: Don't let analysis tooling become a rabbit hole. Define a clear first experiment (e.g., "t-SNE of penultimate-layer features for clean vs triggered inputs on the CNN") and get results before building elaborate infrastructure.

---

### Idea 2: Stronger backbone (ResNet18) + Vision LLMs

**ResNet18: Yes, do this soon. It's a short-term platform task.**

- Validates whether findings on the simple CNN generalize to deeper architectures
- Standard in FL backdoor literature (most papers use ResNet18 on CIFAR-10/GTSRB)
- Your current CNN has 3 conv layers → 2048-dim features → 256-dim → 43 logits. ResNet18 would give richer representations, which makes Idea 1 more meaningful
- Engineering effort is low: define a new model class, make it selectable via config

**Vision LLMs (ViT, CLIP, etc.): Not yet. But worth planning for.**

Honest assessment:
- For GTSRB (32x32 images, 43 classes): ViT is overkill and won't train well from scratch at this resolution without heavy augmentation or pre-training
- For FL: large models dramatically increase communication cost (ViT-B is ~86M params vs ResNet18's ~11M vs your CNN's ~0.6M). This changes the FL dynamics fundamentally
- For your thesis direction (autonomous driving): multimodal/ViT backbones ARE highly relevant — autonomous driving perception increasingly uses ViT and fusion architectures

**My recommendation**: Use ViT/CLIP as a longer-term thesis chapter, not a near-term platform task. When you get there, interesting questions emerge:
- Do attention-based architectures exhibit different backdoor representation patterns than CNNs?
- Are backdoors in pre-trained ViTs (fine-tuned in FL) qualitatively different from trained-from-scratch CNNs?
- Multimodal fusion (camera + LiDAR) creates NEW attack surfaces at the fusion layer — this is largely unexplored

---

### Idea 3: More attack baselines (model replacement, DBA)

**Verdict: Necessary platform work. Do it, but in the right order.**

**Model replacement (Bagdasaryan et al., 2020)**:
- Fundamentally different from data poisoning — the attacker directly crafts model weights
- Directly relevant to Idea 1: the attacker manipulates the model's representation space by design
- Harder to defend with simple norm clipping (the attacker can scale their update to match normal norms)
- Implementation complexity: medium — you need to replace the client's training with a constrained optimization that produces a model replacement update

**DBA (Distributed Backdoor Attack, Xie et al., 2020)**:
- Trigger is split across multiple clients — each client only injects a sub-trigger
- More realistic in FL (no single client holds the full backdoor)
- Tests whether defenses can detect coordinated but individually-innocent updates
- Implementation complexity: medium — modify `PixelBackdoorDataset` to accept sub-trigger specifications per client

**Priority**: Model replacement first — it connects to Idea 1 more directly and is the stronger attack baseline. DBA second.

---

## What Should Be the Next Steps — Recommended Priority

### Phase A: Finish current baselines

1. **Complete the full experiment matrix** with the current CNN:
   - Clean baseline (running now)
   - Pixel backdoor + no defense
   - Pixel backdoor + each defense (norm clipping, FedMedian, Krum, etc.)
   - Record: test_acc, ASR, target_class_clean_accuracy across rounds
2. **Document findings** — these are your baseline numbers for everything that follows

### Phase B: ResNet18 + representation hooks

3. **Add ResNet18 as a selectable backbone** (`model-type` config param)
4. **Add a `forward_features()` method** to both CNN and ResNet18 that returns intermediate representations (feature maps before the classifier head). This is the minimal engineering needed for Idea 1.
5. **Re-run the experiment matrix on ResNet18** — compare ASR and clean accuracy against CNN

### Phase C: Representation-space analysis (together with Phase B)

6. **Build analysis utilities** (new `analysis/` directory):
   - Feature extraction: run a dataset through the model, collect penultimate-layer features
   - Dimensionality reduction: t-SNE/UMAP visualization
   - Statistical comparison: clean vs triggered feature distributions (e.g., L2 distance between cluster centroids, cosine similarity distributions)
7. **First experiment**: For the trained CNN and ResNet18 (with backdoor), visualize:
   - Clean inputs features (colored by true class)
   - Triggered inputs features (same inputs with trigger)
   - Where do triggered inputs land? Near the target class cluster? In a separate region?
8. **Compare across attack types** once model replacement is implemented

### Phase D: Model replacement attack (can overlap with B+C)

9. **Implement Bagdasaryan model replacement** as a new attack module
10. **Repeat representation analysis** for model replacement — how does it differ from pixel trigger?
11. This comparison (heuristic trigger vs model replacement in representation space) could be a key thesis contribution

### Phase E: DBA + deeper analysis

12. Implement DBA
13. Investigate whether there are shared "backdoor signatures" across attack types in representation space
14. If signatures exist → explore defense ideas based on detecting them (connects to your TTA hypothesis)

---

## Other Constructive Suggestions

### On research framing

Your thesis direction (CLAUDE.md hypothesis) goes: heuristic attacks → optimization attacks → server defense limits → client-side defense → TTA-inspired defense. The representation-space analysis fits perfectly as the **bridge between step 1 and step 2**: understanding WHY heuristic attacks work in representation space leads naturally to designing optimization-based attacks that operate in that space.

Consider framing your thesis story as: "Understanding backdoor attacks through the lens of representation competition → designing stronger attacks → evaluating whether current defenses are sufficient → proposing representation-aware defenses."

### On coding structure

For the upcoming work, I'd suggest this directory structure:

```
fl_v2/src/fl_v2/
├── analysis/          # NEW — post-hoc analysis, not in the training loop
│   ├── feature_extraction.py   # extract representations from trained models
│   ├── visualization.py        # t-SNE, UMAP, cluster plots
│   └── statistics.py           # distribution comparisons
├── models/
│   ├── classifier.py           # current CNN
│   └── resnet.py               # NEW — ResNet18 wrapper
├── attacks_defenses/
│   └── attacks/
│       ├── pixel_backdoor.py   # existing
│       ├── model_replacement.py # NEW
│       └── dba.py              # NEW (later)
```

Key principle: `analysis/` is for offline investigation (run after training, load saved model). It should NOT be coupled into the FL training loop. This keeps the training pipeline clean and the analysis flexible.

### On the ViT/LLM question — a more specific future direction

When you eventually move to autonomous driving, the interesting question isn't "can we use ViT as backbone" (yes, trivially). It's: **"does the fusion layer between modalities (camera + LiDAR) create a new attack surface that single-modality backdoor analysis misses?"** This is where your representation-space analysis could become truly novel — studying how backdoors propagate through fusion layers. But this is a late-thesis question, not a near-term one.
