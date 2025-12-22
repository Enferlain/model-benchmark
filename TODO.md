# Model Benchmark Explorer - TODO

## Current Status

Working prototype with:

- ✅ Local model scanning and image generation
- ✅ CLIP score (prompt adherence)
- ✅ LPIPS diversity (intra-prompt variety)
- ✅ Configurable generation settings (sampler, steps, CFG, seed, resolution)
- ✅ Separate Generate/Analyze workflows with cancellation
- ✅ Metric info modals with detailed explanations

---

## Metrics To Implement

### High Priority

- [ ] **VQA / TIFA-style scoring** - Question-answering based prompt faithfulness
  - Requires: BLIP-2, LLaVA, or InstructBLIP model
  - Can also use gemini api with multimodal gemma 3 27b model (uses siglip2)
  - Parse prompts into yes/no questions, run VQA, score accuracy

### Medium Priority

- [ ] **MS-SSIM** - Alternative to LPIPS for diversity measurement
  - Available in `torchmetrics` or `pytorch-msssim`

### Lower Priority (Complex)

- [ ] **GenEval-style detector** - Object/attribute counting
  - Requires: YOLO/DETR object detector
  - Check if "2 red apples" actually has 2 red apples

### Aesthetic Scorers or other stuff maybe

### DINOv3 smalL/large semantic consistency score, dino lpips instead of alex/vgg, object halo (saliency sparsity)

|Metric Name|Using Model...|Tells|
|---|---|---|
Semantic  Match|vitl16 (Global)|"""Is this actually what I asked for visually?"""|
Deep Diversity|vitb16 (Layers)|"""Is the model just repeating itself?"""|
Object Integrity|vitl16 (Attention)|"""Did the AI mess up the body/structure?"""|
Batch Quality|convnext-base|"""Does this whole folder of images look 'natural'?"""|

---

## Features To Add

### UI/UX

- [ ] **Image gallery viewer** - View generated images per model
- [ ] **Drag/upload images for prompts** - Drag or upload images from any place for a tagger to build a prompt
- [ ] **Prompt editor** - Edit/manage test prompts in UI
- [ ] **Model comparison view** - Side-by-side image comparison, maybe with sli slider
- [ ] **img arena** - conoyare random gens fo same seed and prompt between two models to compute user score 
- [ ] **Export results** - CSV/JSON export of benchmark data
- [ ] **Model fetching** - Fix model fetch/downloader

### Backend

- [ ] **Negative prompt support** - Per-generation negative prompts
- [ ] **Batch generation** - Queue multiple models for overnight runs
- [ ] **Cache metrics** - Don't recompute if images haven't changed
- [ ] **LoRA support** - Test LoRA models (not just checkpoints)

### Data Management

- [ ] **Prompt categories** - Group prompts by type (portrait, landscape, etc.)
- [ ] **Prompt difficulty** - Tag prompts as easy/medium/hard
- [ ] **Reference images** - Compare against ground truth
- [ ] **Model downloading** - Downloading models from hf and civit

---

## Known Issues

- [ ] VQA score currently mocked (returns random values)
- [ ] Diversity metric label still says "Diversity (⚠️ WIP)" - cross-prompt version
- [ ] Old image naming (`gen_000.png`) not recognized by new system

---

## Quick Fixes Needed

