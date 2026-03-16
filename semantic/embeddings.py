import os
import time
import chromadb
from chromadb.config import Settings
from typing import Dict
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "semantic_memory")
client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="code_embeddings"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

def get_embeddings(text: str):
    return embedder.encode(text).tolist()


def build_semantic_memmory(parsed_repo: Dict[str, Dict]):
    doc_id = 0

    for file_path, data in parsed_repo.items():

        folder = data.get("folder", "")
        module = data.get("module", "")

        # ----------------- FILE LEVEL EMBEDDING -----------------
        file_text = ""
        file_text += "\n".join(data.get("imports", [])) + "\n"
        file_text += f"# module: {module} folder: {folder}\n"

        for func_name, func_data in data.get("functions", {}).items():
            file_text += func_data.get("code", func_name) + "\n"

        for class_name, class_data in data.get("classes", {}).items():
            # class_data may not have 'code' if parser doesn't store full class code
            class_code = class_data.get("code", "")
            file_text += class_code + "\n"

        try:
            embedding = get_embeddings(file_text)
            collection.add(
                documents=[file_text],
                embeddings=[embedding],
                metadatas=[{
                    "type": "file",
                    "path": file_path,
                    "folder": folder,
                    "module": module
                }],
                ids=[f"doc_{doc_id}"]
            )
            doc_id += 1
        except Exception as e:
            print(f"Embedding failed for file {file_path}: {e}")

        # ----------------- FUNCTION LEVEL EMBEDDING -----------------
        for func_name, func_data in data.get("functions", {}).items():
            func_code = func_data.get("code", func_name)
            try:
                embedding = get_embeddings(func_code)
                collection.add(
                    documents=[func_code],
                    embeddings=[embedding],
                    metadatas=[{
                        "type": "function",
                        "path": file_path,
                        "folder": folder,
                        "module": module,
                        "name": func_name
                    }],
                    ids=[f"doc_{doc_id}"]
                )
                doc_id += 1
            except Exception as e:
                print(f"Embedding failed for function {func_name}: {e}")

        # ----------------- CLASS LEVEL EMBEDDING -----------------
        for class_name, class_data in data.get("classes", {}).items():
            class_code = class_data.get("code", class_name)  # fallback to name
            try:
                embedding = get_embeddings(class_code)
                collection.add(
                    documents=[class_code],
                    embeddings=[embedding],
                    metadatas=[{
                        "type": "class",
                        "path": file_path,
                        "folder": folder,
                        "module": module,
                        "name": class_name
                    }],
                    ids=[f"doc_{doc_id}"]
                )
                doc_id += 1
            except Exception as e:
                print(f"Embedding failed for class {class_name}: {e}")

        # slight throttle to avoid overloading
        time.sleep(0.1)

    print("Semantic memory stored in ChromaDB")
    print("Collection info:", collection.count())
    return collection