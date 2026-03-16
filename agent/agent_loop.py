import os
from typing import Optional
from utils.llm_call import ask_llm
from utils.executor import execute_plan
from agent.validator import CodeValidator
from agent.dashboard import Dashboard


class AgentLoop:
    """
    Iterative agent loop for autonomous code understanding and generation.

    Steps per task:
    1. UNDERSTAND  - retrieve context (semantic + graph + Hebbian memory)
    2. PLAN        - decompose task into subtasks via LLM
    3. GENERATE    - write code for current subtask
    4. VALIDATE    - syntax check, import check
    5. REFLECT     - LLM self-critique
    6. EXECUTE     - write to disk
    7. LEARN       - update MemoryGraph + BDH working memory
    """

    def __init__(
        self,
        retriever,
        graph_store,
        memory,
        parsed,
        repo_structure,
        max_retries: int = 3,
        bdh_router=None,
        working_memory=None,
    ):
        self.retriever = retriever
        self.graph_store = graph_store
        self.memory = memory
        self.parsed = parsed
        self.repo_structure = repo_structure
        self.validator = CodeValidator()
        self.max_retries = max_retries

        # Optional BDH components
        self.bdh_router = bdh_router
        self.working_memory = working_memory

        # Dashboard
        self.dashboard = Dashboard(
            memory_graph=memory,
            working_memory=working_memory,
            bdh_router=bdh_router,
        )

    def _retrieve_context(self, query: str) -> dict:
        """Step 1: Retrieve relevant context for the query."""
        # If BDH router is available, use concept-enhanced retrieval
        if self.bdh_router:
            results = self.bdh_router.route(query)
        else:
            results = self.retriever.retrieve(query)

        top_nodes = [node for node, _ in results[:5]]

        snippets = {}
        for n in top_nodes:
            parts = n.split("::")
            if len(parts) >= 2:
                file_data = self.parsed.get(parts[0], {})
                if len(parts) == 2:
                    code = file_data.get("functions", {}).get(parts[1], {}).get("code", "")
                elif len(parts) == 3:
                    code = (file_data.get("classes", {})
                            .get(parts[1], {})
                            .get("methods", {})
                            .get(parts[2], {})
                            .get("code", ""))
                else:
                    code = ""
                if code:
                    snippets[n] = code

        folders = [self.graph_store.graph.nodes[n].get("folder") for n in top_nodes
                   if self.graph_store.graph.nodes[n].get("folder")]
        target_folder = max(set(folders), key=folders.count) if folders else "generated"

        import_nodes = set()
        for n in top_nodes:
            import_nodes.update(self.graph_store.get_full_upstream(n, types=["file", "module"]))

        return {
            "retrieved_code": snippets,
            "target_folder": target_folder,
            "imports": list(import_nodes),
            "repo_structure": self.repo_structure,
            "memory_summary": self.memory.get_summary(max_nodes=10),
            "top_nodes": top_nodes,
            "results": results[:10],
        }

    def _decompose_task(self, task: str, context: dict) -> list:
        """Step 2: Break a task into subtasks using LLM."""
        code_section = ""
        for name, code in context.get("retrieved_code", {}).items():
            code_section += f"\n- {name}"

        # Include BDH concept hints if available
        concept_hint = ""
        if self.working_memory:
            concepts = self.working_memory.get_concept_keywords(threshold=0.3)
            if concepts:
                concept_hint = f"\nRelevant code patterns detected: {', '.join(concepts)}"

        prompt = f"""
You are a software architect. Break this task into concrete subtasks.

Task: {task}
{concept_hint}

Available code in the repo:
{code_section}

Repo structure:
{context.get('repo_structure', 'Not provided.')}

Return ONLY a numbered list of subtasks (2-5 items). Each subtask should be a single
file or function to create/modify. Be specific. Example:
1. Create auth/jwt_utils.py with token generation and validation functions
2. Add login endpoint to routes/api.py
"""
        response = ask_llm(prompt, max_tokens=1024)

        subtasks = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                task_text = line.lstrip("0123456789.)- ").strip()
                if task_text:
                    subtasks.append(task_text)

        return subtasks if subtasks else [task]

    def _generate_code(self, subtask: str, context: dict) -> str:
        """Step 3: Generate code for a single subtask."""
        code_section = ""
        for name, code in context.get("retrieved_code", {}).items():
            code_section += f"\n### {name}\n```python\n{code}\n```\n"

        prompt = f"""
You are writing Python code for a repository.

# Subtask:
{subtask}

# Target folder: {context.get('target_folder', 'generated')}
# Available imports: {context.get('imports', [])}

# Existing code for reference:
{code_section}

# Repo structure:
{context.get('repo_structure', '')}

Write ONLY the Python code. No explanations, no markdown fences.
Include proper imports, docstrings, and error handling.
"""
        return ask_llm(prompt, max_tokens=4096, temperature=0.3)

    def _reflect(self, subtask: str, code: str) -> tuple:
        """Step 5: LLM self-critique of generated code."""
        prompt = f"""
Review this generated Python code for correctness and completeness.

# Original task:
{subtask}

# Generated code:
```python
{code}
```

Answer with:
PASS - if the code correctly implements the task
FAIL: <reason> - if there are issues

Be concise. One line only.
"""
        response = ask_llm(prompt, max_tokens=256, temperature=0.2)
        response = response.strip()

        if response.upper().startswith("PASS"):
            return True, "Self-critique passed"
        else:
            return False, response

    def _learn(self, subtask: str, top_nodes: list):
        """Step 7: Update MemoryGraph + BDH working memory."""
        # Update Hebbian MemoryGraph
        self.memory.update_from_concepts(
            concepts=top_nodes,
            context_node=f"task:{subtask[:50]}"
        )
        self.memory.decay_all()

        # Update BDH working memory if available (dual-Hebbian feedback)
        if self.working_memory:
            self.working_memory.process_step(subtask)

    def _clean_code(self, raw_code: str) -> str:
        """Strip markdown fences if the LLM wrapped the code."""
        code = raw_code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        return code

    def run(self, task: str, auto_save: bool = False, show_dashboard: bool = True) -> dict:
        """
        Execute the full agent loop for a task.

        Returns a dict with results for each subtask.
        """
        # Reset BDH working memory for new task
        if self.working_memory:
            self.working_memory.reset()
            self.working_memory.process_step(task)

        print(f"\n{'='*60}")
        print(f"  AGENT: {task}")
        print(f"{'='*60}")

        # Step 1: Retrieve context
        print("\n[1/7] Retrieving context...")
        context = self._retrieve_context(task)

        print(f"  Found {len(context['retrieved_code'])} code snippets")
        print(f"  Target folder: {context['target_folder']}")

        # Step 2: Decompose task
        print("\n[2/7] Decomposing task...")
        subtasks = self._decompose_task(task, context)
        print(f"  Subtasks:")
        for i, st in enumerate(subtasks, 1):
            print(f"    {i}. {st}")

        results = {"task": task, "subtasks": []}

        for idx, subtask in enumerate(subtasks, 1):
            print(f"\n{'─'*40}")
            print(f"  Subtask {idx}/{len(subtasks)}: {subtask}")
            print(f"{'─'*40}")

            # Show dashboard if enabled
            if show_dashboard:
                self.dashboard.print_dashboard(
                    task=task,
                    subtask=subtask,
                    subtask_idx=idx,
                    total_subtasks=len(subtasks),
                    retrieved=context.get("results", []),
                    phase="GENERATE",
                )

            success = False
            code = ""
            issues = []
            for attempt in range(1, self.max_retries + 1):
                # Step 3: Generate
                print(f"\n  [3/7] Generating code (attempt {attempt})...")
                raw_code = self._generate_code(subtask, context)
                code = self._clean_code(raw_code)

                # Step 4: Validate
                print("  [4/7] Validating...")
                valid, issues = self.validator.validate(code)
                if not valid:
                    print(f"  Validation failed: {issues}")
                    if attempt < self.max_retries:
                        subtask = f"{subtask}\n\nPrevious attempt had errors: {issues}\nFix them."
                    continue

                # Step 5: Self-critique
                print("  [5/7] Self-critiquing...")
                passed, critique_msg = self._reflect(subtask, code)
                if not passed:
                    print(f"  Critique: {critique_msg}")
                    if attempt < self.max_retries:
                        subtask = f"{subtask}\n\nSelf-critique feedback: {critique_msg}\nRevise the code."
                        continue

                success = True
                break

            if success:
                # Step 6: Execute
                if auto_save:
                    print("  [6/7] Saving to disk...")
                    file_name = self._suggest_filename(subtask)
                    path = execute_plan(code, context["target_folder"], file_name)
                    print(f"  Saved: {path}")
                else:
                    print("  [6/7] Code ready (auto-save disabled)")
                    print(f"  Preview:\n{code[:500]}...")

                # Step 7: Learn
                print("  [7/7] Updating memory...")
                self._learn(subtask, context.get("top_nodes", []))

                results["subtasks"].append({
                    "subtask": subtask,
                    "code": code,
                    "status": "success",
                })
            else:
                print(f"  Failed after {self.max_retries} attempts")
                results["subtasks"].append({
                    "subtask": subtask,
                    "code": code,
                    "status": "failed",
                    "issues": issues,
                })

        succeeded = sum(1 for s in results["subtasks"] if s["status"] == "success")
        print(f"\n{'='*60}")
        print(f"  DONE: {succeeded}/{len(subtasks)} subtasks completed")
        print(f"{'='*60}")

        return results

    def _suggest_filename(self, subtask: str) -> str:
        """Suggest a filename from the subtask description."""
        words = subtask.lower().split()
        for w in words:
            if w.endswith(".py"):
                return w
        clean = "".join(c if c.isalnum() or c == " " else "" for c in subtask.lower())
        name = "_".join(clean.split()[:3])
        return f"{name}.py" if name else "generated.py"
