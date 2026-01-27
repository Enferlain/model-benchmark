# Model Benchmark Explorer - TODO

## Current Status

Working prototype with:

- ✅ Local model scanning and image generation
- ✅ CLIP score (prompt adherence)
- ✅ LPIPS diversity (intra-prompt variety)
- ✅ Configurable generation settings (sampler, steps, CFG, seed, resolution)
- ✅ Separate Generate/Analyze workflows with cancellation
- ✅ Metric info modals with detailed explanations
- ✅ **New Transfer List Interface** (Library/Queue) with filters and presets
- ✅ **Dedicated Analytics Page**

---

## High Priority / Quick Fixes

---

## Features To Add

### UI/UX & Dashboard

- [x] **Image gallery viewer** - View generated images per model
- [x] **Drag/upload images for prompts** - Drag or upload images from any place for a tagger to build a prompt
- [x] **Prompt editor** - Edit/manage test prompts in UI
- [x] **Prompt sidebar buttons** - Shuffle prompts, enable/disable all
- [x] **Model comparison view** - Side-by-side image comparison
- [x] **Model selection for benchmarking** - Implemented via new Transfer List
- [x] **Visible database tab** - Implemented as "Analytics" page
- [ ] **Img Arena**
  - Compare random gens of same seed/prompt between models for user voting
  - Host on remote shareable link (generations transient)
  - Import prompts from boorus/external links with auto-tagging
  - _Fix:_ "model arena" header whitespace
- [ ] **Export results** - CSV/JSON export of benchmark data
- [ ] **Share comparison** - Build grid plots and upload to imgur/imgsli
- [ ] **Prompt set name option** - Ability to name/save prompt sets
- [ ] **Database models style** - Badge styling for prediction types (Red: vpred, Grey: eps)
- [ ] **Benchmark runs db details** - Show model count/list in history even if models deleted
- [ ] **Benchmark runs actions** - Delete/Export benchmark runs
- [ ] **Metrics dropdown** - Consistent metric selection UI across all pages
- [ ] **Diversity metric label** - Rename "Diversity (⚠️ WIP)" once validated

### Backend & Core

- [ ] **Negative prompt support** - Per-generation negative prompts
- [ ] **Noise/Color scorers** - Forensic noise, PCA, color distribution metrics
- [ ] **Batch generation** - Queue multiple models for overnight runs
- [x] **Cache metrics** - Don't recompute if images haven't changed
- [ ] **LoRA support** - Test LoRA models (not just checkpoints)
- [ ] **Unique identifiers for prompts** - Immutable IDs to prevent regeneration collisions
- [ ] **Queue management** - Interrupt/Cancel specific items in download queue

### Data Management

- [ ] **Prompt categories** - Group prompts by type (portrait, landscape, etc.)
- [ ] **Prompt difficulty** - Tag prompts as easy/medium/hard
- [ ] **Reference images** - Compare against ground truth
- [x] **Model downloading** - Downloading models from HF and Civitai
- [ ] **Support SHA256 Hashing** - Logic for standard SHA256 (HF/CivitAI compatibility)
- [ ] **Shared model hash lookup** - Map SHA256 → Metadata (Community `known_models.json`)
- [ ] **Remote Arena Results** - Sync voting results to server

---

## Metrics Research

### High Priority

- [ ] **VQA / TIFA-style scoring** - Question-answering based prompt faithfulness (BLIP-2/LLaVA)

### Medium Priority

- [ ] **MS-SSIM** - Alternative to LPIPS for diversity

### Lower Priority / Experimental

- [ ] **GenEval-style detector** - Object/attribute counting (YOLO/DETR)
- [ ] **DINOv3 / Semantic Consistency** - Advanced semantic matching
- [ ] **Aesthetic Scorers**

| Metric Name      | Using Model...     | Purpose                                            |
| ---------------- | ------------------ | -------------------------------------------------- |
| Semantic Match   | vitl16 (Global)    | "Is this actually what I asked for visually?"      |
| Deep Diversity   | vitb16 (Layers)    | "Is the model just repeating itself?"              |
| Object Integrity | vitl16 (Attention) | "Did the AI mess up the body/structure?"           |
| Batch Quality    | convnext-base      | "Does this whole folder of images look 'natural'?" |

---

## Known Issues

- [ ] VQA score currently mocked
- [ ] Old image naming (`gen_000.png`) not recognized by new system
