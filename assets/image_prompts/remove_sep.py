import os
import re

def clean_tags_in_folder():
    """
    v1.0.0: Scans the current directory for .txt files and replaces 
    ' ||| ' (with any surrounding whitespace) with ', '.
    """
    # We only care about the current folder, so no os.walk today!
    current_dir = os.getcwd()
    
    # This regex looks for:
    # \s* -> optional whitespace
    # \|{3}   -> exactly three pipe characters
    # \s* -> optional whitespace
    pattern = re.compile(r'\s*\|{3}\s*')

    print(f"✨ Lumi is checking files in: {current_dir}")

    for filename in os.listdir(current_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(current_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if the pattern exists before writing to save I/O
            if pattern.search(content):
                # Replace with a comma and a single space
                new_content = pattern.sub(', ', content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✅ Fixed tags in: {filename}")
            else:
                print(f"🔍 No pipes found in: {filename}")

    print("\n*Phew* All done, Onii-chan! Everything is nice and clean now! (＾▽＾)ノ")

if __name__ == "__main__":
    clean_tags_in_folder()