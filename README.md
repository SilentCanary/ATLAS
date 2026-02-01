# ATLAS - AI-Powered Codebase Understanding System

## Problem Statement
Developers struggle to understand large or unfamiliar codebases. Finding where a feature is implemented, understanding dependencies, and onboarding to new projects is time-consuming and error-prone.

## Core Idea
ATLAS allows users to upload or link a GitHub repository and then ask natural language questions about the codebase. The AI builds a structural and semantic understanding of the repository and answers questions like:

- "Where is authentication handled?"
- "Which files are affected if I change X?"
- "How does the payment flow work?"

## Key Features

### Repository-Level Understanding
Parse the entire codebase and build a structured mental model.

### Natural Language Q&A
Users can ask architectural and functional questions in plain English.

### Impact Analysis
Show which files, functions, and modules are affected by a change.

### Architecture Visualization
Automatically generate system and dependency diagrams.

### Onboarding Assistant
Explain the codebase to new developers with summaries.

## System Architecture (High Level)

1. **Frontend**: Web UI for uploading repositories and chatting
2. **Code Ingestion Layer**: Clones/parses repos using AST / Tree-sitter
3. **Knowledge Representation Layer**:
   - Structural code graph (Neo4j / NetworkX)
   - Semantic embeddings (Hugging Face) stored in vector DB (OpenSearch / FAISS)
4. **Reasoning Layer**: LangGraph orchestrates queries over graph + embeddings
5. **LLM Integration**: AWS Bedrock or Hugging Face generates answers
6. **Output Layer**: Highlighted code, diagrams, and impact results

## Getting Started

[Development setup instructions will go here]

## Tech Stack

- Frontend: [TBD]
- Backend: [TBD]
- Database: Neo4j (graph), OpenSearch/FAISS (vector)
- AI/ML: Hugging Face, AWS Bedrock, LangGraph
- Code Parsing: Tree-sitter