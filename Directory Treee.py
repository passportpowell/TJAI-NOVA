import os
import json

def create_directory_tree(start_path, exclude_dirs=None, output_json=None, output_txt=None):
    """
    Recursively generates the directory tree structure starting from the given path
    and saves it as JSON and TXT files.

    Args:
        start_path (str): The root directory to start the tree structure.
        exclude_dirs (list, optional): List of directory names to exclude. Defaults to ['__pycache__'].
        output_json (str, optional): Path to save the directory tree as a JSON file.
        output_txt (str, optional): Path to save the directory tree as a TXT file.
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__']

    tree = {}

    for root, dirs, files in os.walk(start_path):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        relative_path = os.path.relpath(root, start_path)
        current_dir = tree
        if relative_path != '.':
            for part in relative_path.split(os.sep):
                current_dir = current_dir.setdefault(part, {})
        current_dir.update({file: None for file in files})

    # Save as JSON
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(tree, json_file, indent=4)

    # Save as TXT
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as txt_file:
            def write_tree(subtree, indent_level=0):
                for key, value in subtree.items():
                    txt_file.write('    ' * indent_level + f"{key}/\n" if isinstance(value, dict) else '    ' * indent_level + f"{key}\n")
                    if isinstance(value, dict):
                        write_tree(value, indent_level + 1)

            write_tree(tree)

if __name__ == "__main__":
    # Replace with your desired starting directory
    start_directory = r'd:/Tera-joule/Terajoule - AI Architecture/AI Assistants/Nova - AI Coordinator v2/'
    json_output_path = r'd:/Tera-joule/Terajoule - AI Architecture/AI Assistants/Nova - AI Coordinator v2/directory_tree.json'
    txt_output_path = r'd:/Tera-joule/Terajoule - AI Architecture/AI Assistants/Nova - AI Coordinator v2/directory_tree.txt'

    create_directory_tree(start_directory, output_json=json_output_path, output_txt=txt_output_path)