from .llm_call import ask_llm

from .llm_call import ask_llm

def explain_repo(context: dict, user_query: str):
    prompt = f"""
You are an expert codebase analyst.

Your job is to HELP the user understand the repository based on context.

# Repository Overview:
{context.get('repo_summary', 'No summary available.')}

# Module Summaries:
{chr(10).join([f"- {c['summary']}" for c in context.get('module_summaries', [])])}

# Repository Structure:
{context.get('repo_structure', 'Not provided.')}

# Relevant Code Snippets (names only):
{', '.join(list(context.get('retrieved_code', {}).keys()))}

# User Question:
{user_query}

Instructions:
- Answer the question clearly and directly
- Explain flow (how data moves, which functions are used)
- Reference relevant functions/modules when needed
- Keep it structured but not too long
- DO NOT generate code unless explicitly asked
"""

    return ask_llm(prompt)




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
    
    return ask_llm(prompt)  
from .llm_call import ask_llm

def plan_code(context: dict, user_query: str):
    prompt = f"""
You are adding a new feature to a Python repository.

# Repository Overview:
{context.get('repo_summary', 'No summary available.')}

# Module Summaries:
{', '.join([c['summary'] for c in context.get('module_summaries', [])])}

# Repository Structure:
{context.get('repo_structure', 'Not provided.')}

# Target folder for new code: {context.get('target_folder', 'new_folder')}
# Suggested imports: {context.get('imports', [])}

# Relevant code snippets:
{', '.join(list(context.get('retrieved_code', {}).keys()))}

# Task:
{user_query}

Instructions:
- Write a Python file skeleton (do NOT overwrite existing files)
- Include proper imports, class/function definitions
- Reference the relevant functions/methods from the retrieved snippets
- Make the code integrate with existing repo structure
- Do not write explanations, only code
"""
    return ask_llm(prompt)