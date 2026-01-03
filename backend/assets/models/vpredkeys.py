# Version: 1.0.0

from safetensors.torch import load_file, save_file
import torch
import os

def modify_and_save_model(input_filepath: str, output_directory: str, suffix: str = "_vzk") -> None:
    """
    Modifies a safetensors model by adding 'v_pred' and 'ztsnr' empty tensors
    and saves it to a new file with an optional suffix.

    Args:
        input_filepath (str): The full path to the input .safetensors file.
        output_directory (str): The directory where the modified file will be saved.
        suffix (str): The suffix to add to the base filename before the extension.
                      Defaults to "_vzk".
    """
    try:
        # Load the state dictionary from the input file
        print(f"Loading model: {input_filepath}")
        state_dict = load_file(input_filepath)
        print("Model loaded successfully!")

        # Add the 'v_pred' and 'ztsnr' empty tensors
        # Using torch.tensor([]) creates an empty tensor with no dimensions.
        # This is typically what's desired for placeholder or absent data.
        state_dict['v_pred'] = torch.tensor([])
        state_dict['ztsnr'] = torch.tensor([])
        print("Added 'v_pred' and 'ztsnr' tensors.")

        # Construct the new filename
        # Get the base name (e.g., "1bbwm-21821-catobs05-merge_layers-it_4_best-bf16")
        base_name = os.path.basename(input_filepath)
        # Split the base name into name and extension (e.g., "1bbwm...", ".safetensors")
        name_without_ext, ext = os.path.splitext(base_name)
        # Construct the new filename with the suffix
        new_filename = f"{name_without_ext}{suffix}{ext}"
        # Construct the full output path
        output_filepath = os.path.join(output_directory, new_filename)

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Save the modified state dictionary to the new file
        print(f"Saving modified model to: {output_filepath}")
        save_file(state_dict, output_filepath)
        print(f"Successfully modified and saved: {new_filename}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
    except Exception as e:
        print(f"An error occurred while processing {input_filepath}: {e}")

if __name__ == "__main__":
    # Define the directory where your input models are located
    # Make sure to change this to the actual path on your system!
    input_models_directory = r'D:\Projects\model-benchmark-explorer\backend\assets\models'

    # Define the output directory for the modified models
    # This is where the new files will be saved.
    output_models_directory = r'D:\Projects\model-benchmark-explorer\backend\assets\models' # <--- IMPORTANT: Change this if needed!

    # List of model filenames you want to process
    # The script will look for these in `input_models_directory`.
    model_filenames = [
        r'pop3-obsv2_delta_widen6obsbaseline.safetensors'
    ]

    # Process each model in the list
    for filename in model_filenames:
        full_input_path = os.path.join(input_models_directory, filename)
        modify_and_save_model(full_input_path, output_models_directory)
        print("-" * 50) # Separator for readability