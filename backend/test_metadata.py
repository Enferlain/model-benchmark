from safetensors.torch import load_file, safe_open
from pathlib import Path
import json

model_path = Path("assets/models/pop3-obsv2_delta_widen6obsbaseline_v-pred.safetensors")

print(f"Reading metadata for: {model_path}")

try:
    with safe_open(model_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        if metadata:
            print("--- Metadata ---")
            for k, v in metadata.items():
                # Truncate long values for readability
                val_str = str(v)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"{k}: {val_str}")
        else:
            print("No metadata found.")

        # Also check keys specifically for architecture hints if metadata is sparse
        keys = f.keys()
        print(f"\n--- Tensor Keys Sample ({len(keys)} total) ---")
        for k in list(keys)[:5]:
            print(k)

except Exception as e:
    print(f"Error: {e}")
