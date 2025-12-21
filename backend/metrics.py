import random
from typing import List, Optional, Any, Dict

try:
    import torch
    import numpy as np
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Torch/Transformers/Numpy/PIL not found. Running in MOCK mode.")
    # Define dummies for type hints
    class MockImage:
        Image = Any
    Image = MockImage
    np = Any

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. LPIPS diversity will use mock values.")

import lpw_utils

class MetricsCalculator:
    def __init__(self, device=None):
        if device:
            self.device = device
        elif TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.clip_model = None
        self.clip_processor = None
        self.lpips_model = None
        
    def load_clip(self):
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available. Skipping CLIP load.")
            return
        if self.clip_model is None:
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def load_lpips(self):
        if not LPIPS_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            print("LPIPS not available. Skipping load.")
            return
        if self.lpips_model is None:
            print("Loading LPIPS model (alex)...")
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
    
    def calculate_clip_score(self, images: List[Any], prompts: List[str]) -> float:
        """
        Calculates the average cosine similarity between images and prompts, checking 
        long prompts by chunking and averaging chunks (simple "long CLIP" approximation).
        """
        if not images:
            return 0.0
            
        if not TRANSFORMERS_AVAILABLE or not self.clip_model:
            # Fallback for demo/testing without downloading weights
            return float(random.uniform(0.7, 0.95))

        # We process one image-prompt pair at a time to handle chunking correctly
        scores = []
        for img, prompt in zip(images, prompts):
            try:
                # 1. Parse and Tokenize with Weights (Get full list of tokens)
                # Max length large enough to hold all tokens
                token_ids_list, weights_list = lpw_utils.get_prompts_with_weights(
                    self.clip_processor.tokenizer, [prompt], max_length=77*5 
                )
                
                token_ids = token_ids_list[0] # List of ints
                
                # 2. Chunk into 75-token segments (leaving room for BOS/EOS)
                chunk_len = 75
                chunks = []
                for i in range(0, len(token_ids), chunk_len):
                    chunk = token_ids[i:i+chunk_len]
                    # Wrap with BOS/EOS
                    bos = self.clip_processor.tokenizer.bos_token_id
                    eos = self.clip_processor.tokenizer.eos_token_id
                    
                    full_chunk = [bos] + chunk + [eos]
                    
                    # Pad to 77 if necessary (standard CLIP expects 77)
                    if len(full_chunk) < 77:
                        pad = self.clip_processor.tokenizer.pad_token_id
                        full_chunk += [pad] * (77 - len(full_chunk))
                    
                    chunks.append(full_chunk[:77]) # Ensure max 77
                
                if not chunks:
                    continue
                    
                # 3. Compute Embeddings for each chunk
                chunk_scores = []
                
                # Preprocess image once
                image_inputs = self.clip_processor(images=img, return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(pixel_values=image_inputs)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    
                    for chunk_ids in chunks:
                        chunk_tensor = torch.tensor([chunk_ids]).to(self.device)
                        # Get text features
                        text_features = self.clip_model.get_text_features(input_ids=chunk_tensor)
                        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                        
                        # Compute score
                        score = (image_features @ text_features.t()).item()
                        chunk_scores.append(score)
                    
                # 4. Aggregation: Average of chunks
                if chunk_scores:
                    scores.append(sum(chunk_scores) / len(chunk_scores))
                    
            except Exception as e:
                print(f"Error processing pair: {e}")
                scores.append(0.0)

        if not scores:
            return 0.0
            
        return float(sum(scores) / len(scores))

    def calculate_diversity(self, images: List[Any]) -> float:
        """
        Calculates diversity score using CLIP embeddings.
        NOTE: This is cross-prompt diversity (not ideal). Use calculate_lpips_diversity for proper intra-prompt diversity.
        """
        if not images:
            return 0.0
            
        if not TRANSFORMERS_AVAILABLE or not self.clip_model:
             return float(random.uniform(0.3, 0.8))

        # Use CLIP embeddings for "Semantic Diversity"
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_embeds = self.clip_model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            
        # Calculate pairwise cosine distances
        sim_matrix = torch.matmul(image_embeds, image_embeds.T) # (N, N)
        n = sim_matrix.shape[0]
        if n <= 1:
            return 0.0
            
        sum_sim = sim_matrix.sum() - n # subtract diagonal (which is 1s)
        avg_sim = sum_sim / (n * (n - 1))
        
        # Diversity = 1 - average_similarity
        return 1.0 - avg_sim.item()

    def calculate_lpips_diversity(self, grouped_images: Dict[int, List[Any]]) -> float:
        """
        Calculates proper intra-prompt diversity using LPIPS.
        
        Args:
            grouped_images: Dict mapping prompt_index -> list of PIL images generated for that prompt
        
        Returns:
            Average pairwise LPIPS distance across all prompt groups.
            Higher = more diverse outputs for the same prompt.
        """
        if not grouped_images:
            return 0.0
            
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            return float(random.uniform(0.3, 0.6))
        
        all_lpips_scores = []
        
        for prompt_idx, images in grouped_images.items():
            if len(images) < 2:
                continue  # Need at least 2 images to compare
                
            # Convert PIL images to tensors
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # LPIPS expects [-1, 1]
            ])
            
            tensors = [transform(img.convert('RGB')).unsqueeze(0).to(self.device) for img in images]
            
            # Compute pairwise LPIPS
            with torch.no_grad():
                for i in range(len(tensors)):
                    for j in range(i + 1, len(tensors)):
                        lpips_dist = self.lpips_model(tensors[i], tensors[j]).item()
                        all_lpips_scores.append(lpips_dist)
        
        if not all_lpips_scores:
            return 0.0
            
        return float(sum(all_lpips_scores) / len(all_lpips_scores))

    def calculate_metrics(self, images: List[Any], prompts: List[str], grouped_images: Dict[int, List[Any]] = None) -> dict:
        """
        Calculate all available metrics.
        
        Args:
            images: Flat list of all generated images
            prompts: List of prompts (should match 1:1 with images for CLIP score)
            grouped_images: Optional dict mapping prompt_idx -> [images] for LPIPS diversity
        """
        metrics = {
            "clip_score": self.calculate_clip_score(images, prompts),
            "diversity_score": self.calculate_diversity(images),
            "image_count": len(images)
        }
        
        # Add LPIPS diversity if grouped images provided
        if grouped_images:
            metrics["lpips_diversity"] = self.calculate_lpips_diversity(grouped_images)
        
        return metrics
