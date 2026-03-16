ATLAS x BDH: Autonomous Code Intelligence Agent
Context
ATLAS is a codebase reasoning agent that parses repos into graphs, embeds code semantically, and uses an LLM (Mistral via Bedrock) for explanation/generation. However, it has critical gaps: the 529-line Hebbian MemoryGraph is completely unused at runtime, actual code bodies are never passed to the LLM, there's no agent loop (single-shot only), the executor is never called, and generated code is never validated.

BDH (Baby Dragon Hatchling) is a biologically-inspired neural architecture where inference state lives on synaptic edges (Hebbian fast weights) rather than a passive KV cache. Its key properties — monosemantic synapses, emergent modularity, sparse positive activations, and interpretable reasoning — make it uniquely suited for code intelligence.

The vision: Combine BDH's interpretable neural reasoning with ATLAS's structured code knowledge to build an agent that autonomously understands, reasons about, and writes code — with transparent, inspectable thought processes at every layer.

Architecture: Three Layers of Knowledge
Layer 3: BDH Neural Model (trained on code)
   Monosemantic synapses → code pattern recognition
   Evolving sigma state → working memory during reasoning
   Interpretable: you can READ what concepts it's activating

Layer 2: Hebbian MemoryGraph (existing, currently dead)
   Code-concept level learning across sessions
   Strengthens connections between co-used functions/classes
   Decays unused pathways over time

Layer 1: Code Graph + Semantic Index (existing, working)
   NetworkX DiGraph → structural relationships
   ChromaDB → semantic similarity
   Together → multi-signal retrieval
All three layers use the same principle (Hebbian learning) at different scales — neural, conceptual, and structural. This is the novel contribution.

Phase 1: Activate ATLAS's Dead Systems (Foundation Fix)
Goal: Wire up everything that exists but doesn't work. No new architecture yet.

1.1 Activate the MemoryGraph in the Query Pipeline
Files: main2.py, semantic/retrieval.py

In explore_repo(), after retrieval results come back, call memory.update_from_concepts() with the top retrieved nodes and the user's query as context
In explore_repo(), inject memory.get_summary() into the LLM prompt so the model sees what concepts have been historically important
After every query, call memory.decay_all() so temporal decay actually happens
In CodeRetriever.retrieve(), after ChromaDB + graph expansion, also query memory.get_related_nodes() for the top semantic hits and merge those results in (with a weight factor like 0.3x the Hebbian relevance score)
1.2 Pass Actual Code to the LLM
Files: utils/planner.py

explain_repo() and plan_code() currently pass only ', '.join(list(context.get('retrieved_code', {}).keys())) — just names
Change both to pass the actual code snippets (truncated to fit context window):
code_section = ""
for name, code in context.get('retrieved_code', {}).items():
    code_section += f"\n### {name}\n```python\n{code}\n```\n"
Increase max_tokens from 512 to 4096 for code generation
1.3 Build the Agent Loop
New file: agent/agent_loop.py

Replace the single-shot menu with an iterative agent:

while task not complete:
    1. UNDERSTAND  → retrieve context (semantic + graph + Hebbian memory)
    2. PLAN        → decompose task into subtasks via LLM
    3. GENERATE    → write code for current subtask
    4. VALIDATE    → syntax check (ast.parse), import check, type hints
    5. REFLECT     → LLM self-critique: "does this code do what was asked?"
    6. EXECUTE     → write to disk (finally call executor.py!)
    7. LEARN       → update MemoryGraph with co-activated concepts
    8. NEXT        → move to next subtask or finish
1.4 Fix Remaining Infrastructure
Files: Various

Merge duplicate llm_call.py → single utils/llm_call.py, update all imports
Add method-level embeddings in semantic/embeddings.py
Add depth limit (default 3) to graph_store.py BFS expansion
Call executor.py from generate_code() in main2.py
Delete unused utils/context_builder.py and context/final_context.py
Phase 2: Train BDH on Code
Goal: Adapt BDH from character-level Shakespeare to a code-understanding model.

2.1 Tokenizer Adaptation
New file: bdh/tokenizer.py

BDH currently uses character-level (vocab_size=256). For code:

Implement a BPE tokenizer trained on Python code (use tokenizers library from HuggingFace)
Train on a Python corpus (~50K files from The Stack or curated GitHub repos)
Vocab size: 8192–16384 (small enough for BDH's architecture, large enough for code tokens)
Ensure Python keywords, common identifiers, indentation patterns are single tokens
2.2 Code Training Data Pipeline
New file: bdh/data_pipeline.py

Three data sources, mixed during training:

Raw Python code (60%) — files from popular open-source repos
Structured code representations (30%) — ATLAS's own parsed output:
[FILE] path/to/file.py
[FUNC] function_name(args) -> calls: [a, b, c]
[CLASS] ClassName inherits: [Base]
[METHOD] method_name(self, args) -> calls: [x, y]
[IMPORTS] os, sys, json
This teaches BDH the structural relationships ATLAS already knows about
Graph-annotated code (10%) — code with inline graph context:
# UPSTREAM: database.connect(), config.load()
# DOWNSTREAM: api.respond(), logger.info()
def process_request(req):
    ...
2.3 BDH Model Configuration for Code
Modified file: bdh/bdh.py (clone from BDH repo, modify)

@dataclass
class CodeBDHConfig:
    n_layer: int = 8              # more iterations for code reasoning
    n_embd: int = 512             # larger embedding for code vocabulary
    dropout: float = 0.1
    n_head: int = 8               # more heads for parallel concept tracking
    mlp_internal_dim_multiplier: int = 64  # N = 64*512/8 = 4096 neurons
    vocab_size: int = 8192        # BPE code vocabulary
    block_size: int = 1024        # longer context for code
This gives ~50M parameters — trainable on a single GPU in hours.

2.4 Training Script
Modified file: bdh/train.py

Adapt nanoGPT training loop for the code dataset
Train on mixed data sources (raw code + structured + graph-annotated)
Checkpoint saving every 500 iterations
Validation on held-out code repos (perplexity + code completion accuracy)
2.5 Synapse Analysis Tools
New file: bdh/synapse_inspector.py

The killer feature: BDH's monosemantic synapses let you SEE what it learned:

After training, run various code snippets through the model
Record which synapses activate for each concept
Build a synapse→concept mapping:
Synapse #1247 activates for class definitions
Synapse #3891 activates for error handling patterns
Synapse #502 activates for import statements
This mapping becomes the bridge between BDH and ATLAS
Phase 3: Integrate BDH as Code Concept Reasoner
Goal: BDH augments (not replaces) Mistral. BDH handles concept routing and working memory; Mistral handles generation.

3.1 BDH as Retrieval Router
New file: agent/bdh_router.py

When a user asks "add authentication to the API":

Feed the query through BDH
Read BDH's activated synapses
Map activated synapses to code concepts via the synapse→concept mapping
Use those concepts to GUIDE retrieval (not just raw embedding similarity)
class BDHRouter:
    def __init__(self, bdh_model, synapse_map, retriever):
        self.model = bdh_model
        self.synapse_map = synapse_map  # synapse_id -> concept_label
        self.retriever = retriever

    def route(self, query: str) -> List[str]:
        # Run query through BDH, read activated synapses
        tokens = self.tokenizer.encode(query)
        logits, state = self.model(tokens)

        # Extract top activated synapses
        activated = self.get_top_synapses(state, threshold=0.3)

        # Map to code concepts
        concepts = [self.synapse_map[s] for s in activated]

        # Use concepts to enhance retrieval
        enhanced_query = query + " " + " ".join(concepts)
        return self.retriever.retrieve(enhanced_query)
3.2 BDH Working Memory for Multi-Step Tasks
New file: agent/working_memory.py

BDH's evolving sigma state naturally accumulates context:

Step 1 of a task: BDH processes "create user model" → synapses for ORM patterns activate
Step 2: BDH processes "add authentication" → previous ORM synapses REMAIN active (Hebbian persistence) + auth pattern synapses activate
Step 3: BDH processes "connect to API" → all three concept clusters are active simultaneously
This is working memory — the model maintains context across multi-step reasoning without explicit prompt engineering. ATLAS's MemoryGraph does this at the code-concept level; BDH does it at the neural level.

class BDHWorkingMemory:
    def __init__(self, bdh_model):
        self.model = bdh_model
        self.accumulated_state = None  # BDH sigma state

    def process_step(self, step_text: str):
        """Process one step, accumulating state."""
        tokens = self.tokenizer.encode(step_text)
        logits, new_state = self.model(tokens, initial_state=self.accumulated_state)
        self.accumulated_state = new_state
        return self.get_active_concepts()

    def get_active_concepts(self) -> List[str]:
        """Read current working memory as human-readable concepts."""
        activated = self.get_top_synapses(self.accumulated_state)
        return [self.synapse_map[s] for s in activated]
3.3 Dual-Hebbian Feedback Loop
Modified file: graph/memory_graph.py, agent/agent_loop.py

After each successful code generation/validation cycle:

BDH's synaptic state tells us which neural concepts were co-active
ATLAS's MemoryGraph strengthens connections between the corresponding code nodes
Next time a similar task appears, BOTH layers are primed:
BDH activates the right neural patterns faster
MemoryGraph surfaces the right code nodes with higher weight
def feedback_loop(bdh_state, memory_graph, generated_nodes):
    """Synchronize BDH neural state with ATLAS Hebbian memory."""
    # Get BDH's active concepts
    active_concepts = bdh_working_memory.get_active_concepts()

    # Map to code graph nodes
    relevant_nodes = [map_concept_to_node(c) for c in active_concepts]

    # Strengthen connections in MemoryGraph
    memory_graph.update_from_concepts(
        concepts=relevant_nodes + generated_nodes,
        relations=[("co_activated", n1, n2) for n1, n2 in pairs(relevant_nodes)],
        context_node=f"task:{current_task_id}"
    )
Phase 4: Full Autonomous Agent
Goal: Everything works together for end-to-end autonomous coding.

4.1 The Complete Agent Loop
User: "Add JWT authentication to the Flask API"
                    │
                    ▼
    ┌───────────────────────────────┐
    │  1. TASK DECOMPOSITION        │
    │  BDH processes the request    │──→ BDH synapses activate:
    │  Mistral decomposes into      │    auth_patterns, jwt_tokens,
    │  subtasks                     │    flask_decorators, middleware
    └───────────┬───────────────────┘
                │
    ┌───────────▼───────────────────┐
    │  2. CONCEPT ROUTING           │
    │  BDH-activated concepts       │──→ Enhanced retrieval query:
    │  guide retrieval              │    "JWT auth" + [auth, decorator,
    │  MemoryGraph adds historical  │     middleware, token_verify]
    │  co-activation boost          │
    └───────────┬───────────────────┘
                │
    ┌───────────▼───────────────────┐
    │  3. CONTEXT ASSEMBLY          │
    │  Code graph → structural deps │──→ Full context with ACTUAL CODE,
    │  ChromaDB → similar code      │    graph relationships, memory
    │  MemoryGraph → past patterns  │    summaries, BDH concept state
    │  BDH state → active concepts  │
    └───────────┬───────────────────┘
                │
    ┌───────────▼───────────────────┐
    │  4. CODE GENERATION           │
    │  Mistral generates code with  │──→ auth_middleware.py
    │  full context (code bodies +  │    jwt_utils.py
    │  graph + memory + concepts)   │    updated routes.py
    └───────────┬───────────────────┘
                │
    ┌───────────▼───────────────────┐
    │  5. VALIDATION                │
    │  ast.parse() → syntax check   │──→ If fails → back to step 4
    │  Import resolution check      │    with error context
    │  BDH pattern check            │
    └───────────┬───────────────────┘
                │
    ┌───────────▼───────────────────┐
    │  6. SELF-CRITIQUE             │
    │  Mistral reviews own output   │──→ "Does this implement JWT
    │  against original request     │    correctly? Missing refresh
    │  BDH concept coverage check   │    token handling."
    └───────────┬───────────────────┘
                │
    ┌───────────▼───────────────────┐
    │  7. WRITE & LEARN             │
    │  Write files to disk          │──→ MemoryGraph strengthened:
    │  Update MemoryGraph           │    jwt↔flask, decorator↔route
    │  BDH state persists           │    BDH session state saved
    └───────────┴───────────────────┘
4.2 Interpretability Dashboard
New file: agent/dashboard.py

Real-time visibility into agent reasoning:

Current Task: "Add JWT authentication"
Subtask 2/4: "Create token validation middleware"

BDH Active Synapses:
  #1247 (class_definition) ████████░░ 0.82
  #3891 (error_handling)   ██████░░░░ 0.61
  #502  (import_stmt)      █████░░░░░ 0.54
  #2103 (decorator_pattern)████████░░ 0.79

MemoryGraph Hot Paths:
  flask.Blueprint ──(0.9)──→ route_decorator ──(0.7)──→ auth_required
  jwt.decode ──(0.8)──→ token_verify ──(0.6)──→ user_lookup

Retrieved Context:
  app/routes.py::register_routes (score: 0.89)
  app/models.py::User (score: 0.76)
  app/middleware.py::require_auth (score: 0.72)
Implementation Order & Files
Phase 1 (Foundation) — ~3-4 days
#	Task	Files
1	Merge duplicate llm_call.py	utils/llm_call.py, context/llm_call.py, all importers
2	Pass actual code to LLM prompts	utils/planner.py
3	Increase max_tokens to 4096	utils/llm_call.py
4	Wire MemoryGraph into retrieval	main2.py, semantic/retrieval.py
5	Add depth limit to BFS	graph/graph_store.py
6	Add method-level embeddings	semantic/embeddings.py
7	Build agent loop skeleton	agent/__init__.py, agent/agent_loop.py
8	Add validation layer	agent/validator.py
9	Clean up dead code	utils/context_builder.py, context/final_context.py
Phase 2 (Train BDH) — ~1-2 weeks
#	Task	Files
1	Clone BDH repo, adapt model config	bdh/bdh.py
2	Build BPE tokenizer for code	bdh/tokenizer.py
3	Build data pipeline (raw + structured + graph)	bdh/data_pipeline.py
4	Train code-BDH model	bdh/train.py
5	Build synapse inspector	bdh/synapse_inspector.py
6	Build synapse→concept mapping	bdh/concept_map.py
Phase 3 (Integration) — ~1 week
#	Task	Files
1	BDH concept router	agent/bdh_router.py
2	BDH working memory	agent/working_memory.py
3	Dual-Hebbian feedback loop	agent/agent_loop.py, graph/memory_graph.py
4	Enhanced retrieval (semantic + graph + BDH + Hebbian)	semantic/retrieval.py
Phase 4 (Autonomy) — ~1 week
#	Task	Files
1	Full agent loop with all 7 steps	agent/agent_loop.py
2	Task decomposition via LLM	agent/task_planner.py
3	Self-critique loop	agent/critic.py
4	Interpretability dashboard	agent/dashboard.py
5	New main entry point	main3.py or updated main2.py
Verification
Phase 1
Run ATLAS against a test repo (e.g., VIGILUM)
Confirm MemoryGraph.get_summary() output appears in LLM prompt
Confirm actual code snippets appear in LLM prompt
Confirm agent loop iterates and writes code to disk
Confirm validation catches syntax errors
Phase 2
Train BDH on Python corpus, check loss curve matches BDH paper's scaling law
Generate code completions, verify they are syntactically valid Python
Run synapse inspector, confirm monosemantic synapses map to code patterns
Validate that different code constructs activate different synapse clusters
Phase 3
Query "add error handling" → BDH should activate try/except synapses → retrieval should prioritize error-handling code
Multi-step task: confirm BDH working memory retains context across steps
Confirm MemoryGraph weights increase for co-activated code nodes after successful generation
Phase 4
End-to-end: "Add user registration to this Flask app" → agent autonomously:
Decomposes into model/route/template subtasks
Retrieves relevant existing code
Generates, validates, and writes each file
Self-critiques for completeness
BDH synapses and MemoryGraph both updated
Dashboard shows real-time reasoning state