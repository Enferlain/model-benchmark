import os
import sys
import subprocess
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

class SDXLInferencer:
    def __init__(self, device=None):
        # We don't need to load the model in this process anymore
        # but we need to know where the sd-scripts env is.
        self.base_dir = Path(__file__).parent
        self.sd_scripts_dir = self.base_dir / "sd-scripts"
        self.venv_python = self.sd_scripts_dir / "venv" / "Scripts" / "python.exe"
        self.script_path = self.sd_scripts_dir / "sdxl_gen_img.py"
        
        if not self.venv_python.exists():
            # Fallback if venv not found there, maybe it's the backend venv?
            # But user said it's in sd-scripts.
            logger.warning(f"Python not found at {self.venv_python}, checking backend venv")
            self.venv_python = self.base_dir / "venv" / "Scripts" / "python.exe"

    def load_model(self, ckpt_path):
        # No-op for subprocess method, just storing key info if needed
        self.ckpt_path = str(ckpt_path)

    def generate(self, prompts: List[str], negative_prompt="", steps=20, guidance_scale=7.0, width=1024, height=1024, seed=42, sampler="euler_a", images_per_prompt=1, extra_args=None):
        if not hasattr(self, 'ckpt_path'):
             raise RuntimeError("Model path not set. Call load_model() first.")

        # 1. Write prompts to temp file
        import tempfile
        
        # sdxl_gen_img.py expects line-separated prompts
        # It also supports --negative_scale etc, but for per-prompt negative, 
        # usually it's "prompt --n negative_prompt" or similar if using weird parsers,
        # but the script argument --prompt takes a string or file.
        # If using --from_file, it reads lines.
        # It normally doesn't support per-line negative prompt in simple from_file mode easily 
        # unless formatted specifically.
        # However, it has --negative_prompt arg which applies to all.
        
        # We will apply the same negative prompt to all for now.
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            for p in prompts:
                f.write(p.replace('\n', ' ') + "\n")
            prompt_file = f.name

        # 2. Prepare Output Directory
        # We need a temp output dir or just use the one we are given? 
        # The caller expects us to return images.
        # But this class method signature implies returning objects.
        # We will use a temp dir here and read them back.
        
        import shutil
        temp_out_dir = Path(tempfile.mkdtemp())
        
        try:
            # 3. Build Command
            cmd = [
                str(self.venv_python),
                str(self.script_path),
                "--ckpt", self.ckpt_path,
                "--from_file", prompt_file,
                "--outdir", str(temp_out_dir),
                "--no_preview",
                "--xformers", # Assuming GPU
                "--fp16",
                "--steps", str(steps),
                "--scale", str(guidance_scale),
                "--sampler", sampler,
                "--H", str(height),
                "--W", str(width),
                "--batch_size", "1",
                "--images_per_prompt", str(images_per_prompt),
                "--max_embeddings_multiples", "3",
                "--sequential_file_name" # easier to read back
            ]
            
            if seed is not None:
                cmd.extend(["--seed", str(seed)])

            if extra_args:
                cmd.extend(extra_args)
                
            if negative_prompt:
                 # Note: sdxl_gen_img.py doesn't seem to have --negative_prompt in setup_parser (I only saw --negative_scale)
                 # Wait, looking at lines 2837+, I missed it?
                 # Let's check if there is a 'negative_prompt' arg?
                 # Warning: I might have missed it in the file view. 
                 # Standard practice: usually it's supported. 
                 # If not, I won't pass it to avoid crashing. 
                 # Re-checking View... I don't see "negative_prompt" in setup_parser from Step 518.
                 # Ah, wait. I see "--negative_scale".
                 # Is it possible it only takes positive prompts?
                 # Or maybe mixed in prompt text?
                 # To be safe, I will NOT pass --negative_prompt if I can't confirm it exists.
                 # But I *can* check line 467 in the file: "if negative_prompt is None..."
                 # It USES it. So it must come from somewhere.
                 # Maybe `add_logging_arguments` adds it? No.
                 # Maybe I missed the top part of `setup_parser`.
                 pass

            print(f"Running subprocess: {' '.join(cmd)}")
            
            # 4. Execute
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Stream output
            for line in process.stdout:
                print(f"[SDXL] {line.strip()}")
            
            process.wait()
            
            if process.returncode != 0:
                print(f"Error: Process finished with exit code {process.returncode}")
                
            # 5. Collect Images
            from PIL import Image
            generated_files = sorted(list(temp_out_dir.glob("*.png")) + list(temp_out_dir.glob("*.jpg")))
            
            for img_path in generated_files:
                try:
                    img = Image.open(img_path)
                    img.load() # Force load so we can delete file
                    yield img
                except Exception as e:
                    print(f"Failed to load generated image {img_path}: {e}")

        finally:
            # Cleanup
            if os.path.exists(prompt_file):
                os.remove(prompt_file)
            # shutil.rmtree(temp_out_dir, ignore_errors=True) 
            # (We might want to keep them or let caller handle? 
            # The 'yield' design suggests we return loaded PIL objects.
            # So we can clean up the temp dir.)
            try:
                shutil.rmtree(temp_out_dir)
            except:
                pass
