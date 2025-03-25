import os

def list_files(startpath):
    with open("project_tree.txt", "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(startpath):
            # Exclude common unwanted directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'env', 'venv', 'bayesnet_py310']]
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")

list_files(".")
