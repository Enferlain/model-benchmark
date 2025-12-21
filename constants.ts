import { ModelData, MetricOption } from './types';

export const INITIAL_MODELS: ModelData[] = [
  { 
    id: '1', 
    name: 'IllustriousXL_v01', 
    accuracy: 0.88, 
    diversity: 0.65, 
    rating: 4.8, 
    source: 'Civitai',
    url: 'https://civitai.com/models/illustrious-xl' 
  },
  { 
    id: '2', 
    name: 'NoobAI-XL', 
    accuracy: 0.92, 
    diversity: 0.55, 
    rating: 4.5, 
    source: 'Civitai',
    url: 'https://civitai.com/models/noobai-xl'
  },
  { 
    id: '3', 
    name: 'Flux.1-dev', 
    accuracy: 0.74, 
    diversity: 0.45, 
    rating: 4.9, 
    source: 'HuggingFace',
    url: 'https://huggingface.co/black-forest-labs/FLUX.1-dev'
  },
  { 
    id: '4', 
    name: 'Pony Diffusion V6', 
    accuracy: 0.82, 
    diversity: 0.78, 
    rating: 4.7, 
    source: 'Civitai',
    url: 'https://civitai.com/models/pony-diffusion-v6'
  },
  { 
    id: '5', 
    name: 'SDXL Base', 
    accuracy: 0.65, 
    diversity: 0.85, 
    rating: 4.2, 
    source: 'HuggingFace',
    url: 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0'
  }
];

export const METRIC_OPTIONS: MetricOption[] = [
  { 
    value: 'accuracy', 
    label: 'Accuracy (Prompt Adherence)', 
    description: 'How closely the image matches the prompt', 
    direction: 'higher',
    extendedDescription: `**CLIP Score** measures how well the generated image aligns with the text prompt.

**How it works:**
Uses OpenAI's CLIP model to encode both the image and text prompt into embeddings, then computes their cosine similarity.

**Implementation:**
- Tokenizes prompt (with chunking for long prompts >77 tokens)
- Encodes image through CLIP vision encoder
- Computes similarity between text and image embeddings

**What it means for the model:**
- **Higher is better** (0-1 scale, typically 0.2-0.4 for good results)
- A model with high accuracy follows prompts faithfully
- Low scores may indicate the model ignores keywords or has weak text understanding`
  },
  { 
    value: 'diversity', 
    label: 'Diversity (⚠️ WIP)', 
    description: '⚠️ Currently measures cross-prompt variety, not true intra-prompt diversity', 
    direction: 'higher',
    extendedDescription: `**⚠️ NOT YET PROPERLY IMPLEMENTED**

**Current behavior:**
Measures semantic difference between images generated from DIFFERENT prompts. This is not very useful since any model will produce different outputs for different prompts.

**What it SHOULD measure (TODO):**
"Intra-prompt diversity" - variety when generating the same prompt multiple times with different seeds.

**Standard implementations:**
- Average pairwise LPIPS (most common)
- 1 - MS-SSIM over image pairs
- Embedding variance in CLIP/Inception space

**Requires:**
- \`images_per_prompt > 1\` in generation
- Grouping images by prompt before computing diversity
- LPIPS or MS-SSIM computation within groups`
  },
  { 
    value: 'rating', 
    label: 'User Rating', 
    description: 'Community score / 5.0', 
    direction: 'higher',
    extendedDescription: `**User Rating** is the community-provided score from the model's source platform.

**How it works:**
Scraped or fetched from Civitai/HuggingFace model pages.

**What it means for the model:**
- **Higher is better** (scale of 1-5)
- Reflects user satisfaction and popularity
- May not correlate with technical quality`
  },
  { 
    value: 'vqa_score', 
    label: 'VQA Score', 
    description: 'Visual Question Answering faithfulness', 
    direction: 'higher',
    extendedDescription: `**VQA Score** (Visual Question Answering) evaluates if the image contains what the prompt asked for.

**How it works:**
Uses a VQA model to ask questions about the image based on the prompt keywords.

**Implementation:**
Currently uses a placeholder/mock value. Full implementation would:
- Parse prompt for key objects/attributes
- Generate yes/no questions ("Is there a cat?")
- Score based on correct answers

**What it means for the model:**
- **Higher is better**
- Tests semantic accuracy beyond embedding similarity
- Catches cases where CLIP score is high but content is wrong`
  },
  { 
    value: 'lpips_loss', 
    label: 'LPIPS Diversity', 
    description: 'Intra-prompt variety (higher = more diverse outputs for same prompt)', 
    direction: 'higher',
    extendedDescription: `**LPIPS Diversity** measures how varied the model's outputs are when generating the same prompt multiple times.

**How it works:**
Uses LPIPS (Learned Perceptual Image Patch Similarity) to compute perceptual distance between images generated for the same prompt with different seeds.

**Implementation:**
- Generates N images per prompt (configurable via \`images_per_prompt\`)
- Groups images by prompt index
- Computes pairwise LPIPS distance within each group
- Returns average LPIPS score across all groups

**What it means for the model:**
- **Higher is better** (0-1 scale)
- High LPIPS = Creative model that produces varied outputs
- Low LPIPS = Model produces similar/repetitive images ("same face syndrome")
- Requires \`images_per_prompt > 1\` in scan options to get meaningful results`
  },
];
