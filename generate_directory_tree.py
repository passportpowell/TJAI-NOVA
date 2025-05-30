import os

def generate_full_directory_tree(startpath, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for root, dirs, files in os.walk(startpath):
                level = root.replace(startpath, '').count(os.sep)
                indent = ' ' * 4 * level
                f.write(f"{indent}{os.path.basename(root)}/\n")
                subindent = ' ' * 4 * (level + 1)
                for file in files:
                    f.write(f"{subindent}{file}\n")
                for dir in dirs:
                    f.write(f"{subindent}{dir}/\n")
        
        print(f"Full directory tree has been saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace this with the correct path to the directory
path = r"D:\Tera-joule\Terajoule - Terajoule\Projects\AI Architecture\AI Assistants\Nova - AI Coordinator v2"

# Output file path
output_file = "full_directory_tree.txt"

# Generate the full directory tree
generate_full_directory_tree(path, output_file)
