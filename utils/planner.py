# utils/planner.py
from .llm_call import ask_llm
def plan_code(context: dict, user_query: str):
    prompt = f"""
You are adding a new feature to a Python repo.

Target folder: {context['target_folder']}
Suggested imports: {context['imports']}
Relevant code snippets: {list(context['relevant_code_snippets'].keys())}
Task: {user_query}

Write a Python file skeleton (do not overwrite existing files), include proper imports,
class/function definitions, and reference the relevant functions/methods as needed.
"""
    
    return ask_llm(prompt)  # your LLM interface