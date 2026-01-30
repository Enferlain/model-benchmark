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
- ✅ **Performance Overhaul** (Virtualization, Thumbnailing, Pre-fetching)

---

## High Priority / Quick Fixes

- [ ] **Arena/benchmarks** - Generated images are tied to benchmark runs. Arena can be called on arbitrary references/categorical references based on the category, pulled from websites (like danbooru or e621 again depends on category or if there's any set) or from local image/prompt pairs, or from simply images post tagger implementation. Arena can also be called on benchmark runs, in which case the images generated from it will be used for rating. I'm not sure yet, we need more discussions
- [ ] **Tab header consistency** - check if headers are uniform, gallery seems not to be
- [ ] **Arena gaps** - Fix and make consistent the gaps between top bar, header, prompt bar, the images, and the buttons. Should be dynamic from the prompt down based on the arena battle contents.
- [ ] **Generation** - Need to plug the queue into the generate and analyze buttons. Need validation/guidance for pressing, and updating the backend logic if needed for interaction with the new database
- [ ] **Image full view modals** - In arena (and wherever else) make the zoomed model consistent with existing designs (like the one from comparison) same for gallery and prompts if needed.

---

## Features To Add

### UI/UX & Dashboard

- [ ] **Database editing** - Editable entries in ui that reflects in the backend/database. For example attaching prediction types to models and similar. Add a note option to the actions menu.
- [x] **Image gallery viewer** - View generated images per model
- [x] **Drag/upload images for prompts** - Drag or upload images from any place for a tagger to build a prompt
- [x] **Prompt editor** - Edit/manage test prompts in UI
- [x] **Prompt sidebar buttons** - Shuffle prompts, enable/disable all
- [x] **Model comparison view** - Side-by-side image comparison
- [x] **Model selection for benchmarking** - Implemented via new Transfer List
- [x] **Visible database tab** - Implemented as "Analytics" page
- [x] **Img Arena**
  - [x] Compare random gens of same seed/prompt between models for user voting
  - [ ] Host on remote shareable link (generations transient)
  - [ ] Import prompts from boorus/external links with auto-tagging
  - [x] _Fix:_ "model arena" header whitespace
- [x] **Global Data Caching & State Persistence** - Centralized `DataContext` for instant page switching and saved UI states.
- [ ] **Export results** - CSV/JSON export of benchmark data
- [ ] **Share comparison** - Build grid plots and upload to imgur/imgsli
- [ ] **Prompt set name option** - Ability to name/save prompt sets
- [ ] **Database models style** - Badge styling for prediction types (Red: vpred, Grey: eps)
- [ ] **Benchmark runs db details** - Show model count/list in history even if models deleted
- [ ] **Benchmark runs actions** - Delete/Export benchmark runs
- [ ] **Metrics dropdown** - Consistent metric selection UI across all pages
- [ ] **Diversity metric label** - Rename "Diversity (⚠️ WIP)" once validated
- [ ] **Prompt tab is effectively a prompt library** - Maybe add useful features with that in mind, like categories and such, or multiple reference images maybe with source labels (website vs ai gen from what)

### 🔗 External Integration (Long Term Goal)

- [ ] **A1111/ComfyUI API Support** - Connect as generation providers (replacing internal inference) <!-- id: 18 -->
- [ ] **Provider Configuration UI** - Settings to manage API endpoints and keys <!-- id: 19 -->
- [ ] **Shared model path mapping** - Map local files to external API paths <!-- id: 20 -->

### Backend & Core

- [ ] **Negative prompt support** - Per-generation negative prompts
- [ ] **Noise/Color scorers** - Forensic noise, PCA, color distribution metrics
- [ ] **Batch generation** - Queue multiple models for overnight runs
- [ ] **Legacy: Internal Inference** - Complete/Maintain `sd-scripts` wrapper (to be deprecated by APIs) <!-- id: 21 -->
- [x] **Cache metrics** - Don't recompute if images haven't changed
- [ ] **LoRA support** - Test LoRA models (not just checkpoints)
- [x] **Unique identifiers for prompts** - Immutable IDs to prevent regeneration collisions
- [ ] **Queue management** - Interrupt/Cancel specific items in download queue
- [ ] **Prompt alias database** - Check if these are recorded in the database
- [ ] **Prompt search improvements** - Multiple search terms, tags, etc.

### Data Management

- [ ] **Prompt categories** - Group prompts by type (portrait, landscape, etc.)
- [ ] **Prompt difficulty** - Tag prompts as easy/medium/hard
- [ ] **Reference images** - Compare against ground truth
- [x] **Model downloading** - Downloading models from HF and Civitai
- [x] **Support SHA256 Hashing** - Logic for standard SHA256 (HF/CivitAI compatibility)
- [ ] **Shared model hash lookup** - Map SHA256 → Metadata (Community `known_models.json`)
- [ ] **Remote Arena Results** - Sync voting results/database to a remote server

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
