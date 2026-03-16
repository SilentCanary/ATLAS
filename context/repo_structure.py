import os

def build_repo_structure(repo_path):
    structure = {}

    for root, dirs, files in os.walk(repo_path):
        rel_path = os.path.relpath(root, repo_path)

        structure[rel_path] = {
            "folders": dirs,
            "files": files
        }

    return structure

def format_structure(structure):
    text = ""

    for path, content in structure.items():
        text += f"\n📁 {path}\n"

        for f in content["files"]:
            text += f"  - {f}\n"

    return text