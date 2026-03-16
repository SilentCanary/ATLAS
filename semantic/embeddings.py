import os
import time
import chromadb
from typing import Dict
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "semantic_memory")
client = chromadb.PersistentClient(path=CHROMA_PATH)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)


def get_embeddings(text: str):
    return embedder.encode(text).tolist()


def build_semantic_memmory(parsed_repo: Dict[str, Dict]):
    # Clear any stale data from previous runs, then recreate with cosine metric
    try:
        client.delete_collection("code_embeddings")
    except Exception:
        pass
    collection = client.create_collection(
        name="code_embeddings",
        metadata={"hnsw:space": "cosine"}
    )

    doc_id = 0

    for file_path, data in parsed_repo.items():
        folder = data.get("folder", "")
        module = data.get("module", "")

        # ----------------- FILE LEVEL -----------------
        file_text = "\n".join(data.get("imports", [])) + "\n"
        file_text += f"# module: {module} folder: {folder}\n"
        for func_name, func_data in data.get("functions", {}).items():
            file_text += func_data.get("code", func_name) + "\n"
        for class_name, class_data in data.get("classes", {}).items():
            file_text += class_data.get("code", "") + "\n"

        try:
            collection.add(
                documents=[file_text],
                embeddings=[get_embeddings(file_text)],
                metadatas=[{"type": "file", "path": file_path,
                            "folder": folder, "module": module}],
                ids=[f"doc_{doc_id}"]
            )
            doc_id += 1
        except Exception as e:
            print(f"Embedding failed for file {file_path}: {e}")

        # ----------------- FUNCTION LEVEL -----------------
        for func_name, func_data in data.get("functions", {}).items():
            func_code = func_data.get("code", func_name)
            try:
                collection.add(
                    documents=[func_code],
                    embeddings=[get_embeddings(func_code)],
                    metadatas=[{"type": "function", "path": file_path,
                                "folder": folder, "module": module, "name": func_name}],
                    ids=[f"doc_{doc_id}"]
                )
                doc_id += 1
            except Exception as e:
                print(f"Embedding failed for function {func_name}: {e}")

        # ----------------- CLASS LEVEL -----------------
        for class_name, class_data in data.get("classes", {}).items():
            class_code = class_data.get("code", class_name)
            try:
                collection.add(
                    documents=[class_code],
                    embeddings=[get_embeddings(class_code)],
                    metadatas=[{"type": "class", "path": file_path,
                                "folder": folder, "module": module, "name": class_name}],
                    ids=[f"doc_{doc_id}"]
                )
                doc_id += 1
            except Exception as e:
                print(f"Embedding failed for class {class_name}: {e}")

            # ----------------- METHOD LEVEL -----------------
            for method_name, method_data in class_data.get("methods", {}).items():
                method_code = method_data.get("code", method_name)
                try:
                    collection.add(
                        documents=[method_code],
                        embeddings=[get_embeddings(method_code)],
                        metadatas=[{"type": "method", "path": file_path,
                                    "folder": folder, "module": module,
                                    "name": method_name, "class_name": class_name}],
                        ids=[f"doc_{doc_id}"]
                    )
                    doc_id += 1
                except Exception as e:
                    print(f"Embedding failed for method {class_name}.{method_name}: {e}")

        time.sleep(0.05)

    print(f"Semantic memory built: {collection.count()} embeddings stored")
    return collection
