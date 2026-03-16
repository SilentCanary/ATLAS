# ATLAS - Intelligent Codebase Reasoning Agent (BDH Architecture)
## 🧠 Project Overview

This project aims to build an **intelligent coding assistant** that can understand, reason about, and modify an existing codebase.

Unlike traditional code generators, this system does not rely only on prompts.  
Instead, it builds a **structured understanding of the repository** using:

- ⚙️ **Static code analysis (AST parsing)**  
- 🕸️ **Graph-based relationships**  
- 🧠 **Semantic memory (vector embeddings)**  
- 🔁 **Hebbian-style memory updates**  

---

### 🎯 Goal
Enable **context-aware code generation** where the system can:

- ✔️ Understand existing code  
- ✔️ Retrieve relevant logic  
- ✔️ Suggest where new code should go  
- ✔️ Generate meaningful additions without breaking the project


## ⚙️ System Architecture

The assistant is designed as a pipeline of interconnected components:

Repo Parser
     ↓
Structure Graph
     ↓
Semantic Memory (ChromaDB)
     ↓
Retriever (Hybrid: Semantic + Hebbian)
     ↓
Context Builder
     ↓
Planner (LLM)
     ↓
Action Executor
     ↓
Repository Modification


### 🔍 Flow Explained
1. **Repo Parser** → Parses `.py` files using AST to extract functions, classes, imports, and calls.  
2. **Structure Graph** → Builds a graph of relationships (calls, contains, imports).  
3. **Semantic Memory (ChromaDB)** → Stores embeddings for similarity-based retrieval.  
4. **Retriever** → Combines semantic similarity, graph neighbors, and Hebbian weights.  
5. **Context Builder** → Prepares structured input for the LLM (snippets, imports, repo structure).  
6. **Planner (LLM)** → Decides what code to generate and where to place it.  
7. **Action Executor** → Safely writes new files and applies modifications.  
8. **Repository Modification** → Evolves the project incrementally with new features.

## 🔍 Components Explained

### 1. 📂 Repo Parser
Parses all `.py` files using Python AST.  
**Extracts:**
- Functions  
- Classes  
- Methods  
- Imports  
- Function calls  

**Output Example:**
```json
{
  "file_path": "...",
  "functions": {...},
  "classes": {...},
  "imports": [...]
}
```

### 2. 🕸️ Structure Graph
Converts parsed data into a graph.  
**Nodes:** file / class / method / function  
**Edges:** calls / contains / imports  

**Example:**
- Function A → calls → Function B  
- Class → contains → Method  
- File → imports → Module  

---

### 3. 🧠 Semantic Memory
Stores embeddings of code snippets using **ChromaDB**.  
Enables similarity-based retrieval.  

**Example:**  
Query: `"hebbian learning"` → retrieves relevant functions even if names differ  

---

### 4. 🔁 Hebbian Memory Graph
Inspired by Hebbian Learning (*“neurons that fire together wire together”*).  
Strengthens connections between frequently co-accessed nodes.  

**Purpose:**  
- Improve retrieval over time  
- Simulate learning from usage  

---

### 5. 🔎 Retriever
Hybrid retrieval combining:
- Semantic similarity (embeddings)  
- Graph relationships (neighbors, dependencies)  
- Hebbian weights  

**Output:** Top relevant nodes (functions/methods/files)  

---

### 6. 🧱 Context Builder
Builds structured input for LLM:  
- Relevant code snippets  
- Suggested imports  
- Target folder  
- Repo structure  

📌 *This step is crucial — it decides how smart the LLM feels.*  

---

### 7. 🧠 Planner (LLM)
Uses LLM (via API) to:  
- Understand task  
- Decide where to add code  
- Generate new file / function  

**Example:**  
Task: *Create memory graph logger* → LLM generates new utility file  

---

### 8. ⚡ Action Executor
- Takes LLM output  
- Writes new files safely (no overwrite)  
- Handles file creation  

---

### 9. 📁 Repository Modification
- New features are added to the repo  
- System evolves incrementally  

---

## 🚀 Current Features
- ✅ Parse any Python repo  
- ✅ Build code graph (tested on 600+ nodes)  
- ✅ Semantic search over code  
- ✅ Hebbian adaptive memory  
- ✅ Retrieve relevant functions  
- ✅ Generate new code using LLM  
- ✅ Suggest target folder & imports

## 🐉 BDH Integration — Baby Dragon Hatchling Architecture

### 🧠 What is BDH?
The **Baby Dragon Hatchling (BDH)** concept represents a system that:
- Starts with basic knowledge  
- Learns continuously from interactions  
- Strengthens useful patterns over time  
- Gradually becomes better at reasoning and decision-making  

📌 In simple terms:  
The system behaves like a **young agent that learns how to code by exploring a repository**.

## 🐉 How BDH is Implemented

BDH ideas are mapped directly to system components, making the assistant behave like a learning agent:

1. 🐣 **Initial Understanding**  
   *Repo Parser + Graph Builder* → Reads the repository, extracts structure, builds relationships.  
   *BDH Mapping:* Hatchling “seeing the world for the first time.”

2. 🧠 **Memory Formation**  
   *Semantic Memory (ChromaDB) + Hebbian Graph* → Stores meaning of code and strengthens frequently used connections.  
   *BDH Mapping:* “Things used together are remembered together.”

3. 🔁 **Learning from Interaction**  
   *MemoryGraph Updates* → Queries activate nodes, strengthen links, decay unused ones.  
   *BDH Mapping:* Learns by experience.

4. 🔎 **Attention & Recall**  
   *Retriever* → Combines semantic similarity, graph neighbors, and Hebbian weights.  
   *BDH Mapping:* Focuses attention on relevant past knowledge.

5. 🧱 **Contextual Reasoning**  
   *Context Builder* → Forms working memory with snippets, imports, and structure.  

6. 🧠 **Decision Making**  
   *Planner (LLM)* → Decides what to generate and where to place it.  

7. ⚡ **Action & Adaptation**  
   *Executor* → Applies changes, creates new files/features.  

---

### 🔄 Continuous Learning Loop
```
Query → Retrieve → Context → Plan → Execute → Update Memory
↑________________________________________________________|
```
This loop ensures the system improves over time, retrieves frequently used logic faster, and deepens code understanding.

---

### 💡 Why BDH Matters
Unlike stateless coding assistants, BDH builds **long-term memory**, learns **project-specific patterns**, and improves with usage.  
It transforms a coding assistant from a **tool** into a **learning agent**.

**Example:**  
Query: *"Where is Hebbian learning?"*  
System: Retrieves MemoryGraph methods, strengthens those connections, and improves future retrievals.  
📌 That’s BDH in action.

## 🛠️ Tech Stack
- **Python**  
- **AST** (static parsing)  
- **NetworkX** (graph representation)  
- **ChromaDB** (vector database for semantic memory)  
- **HuggingFace / AWS Bedrock(LLM inference)**  
- **JSON-based intermediate storage**  

---

## 🤝 Contribution
This project is under **active development**.  

**Focus areas:**
- Retrieval improvement  
- Better planning logic  
- Robust execution  
