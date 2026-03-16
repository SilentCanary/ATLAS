from .llm_call import ask_llm


def _format_code_snippets(code_dict: dict, max_chars: int = 12000) -> str:
    """Format code snippets for prompt injection, truncating if too large."""
    if not code_dict:
        return "No code snippets available."
    sections = []
    total = 0
    for name, code in code_dict.items():
        snippet = f"### {name}\n```python\n{code}\n```"
        if total + len(snippet) > max_chars:
            sections.append(f"... ({len(code_dict) - len(sections)} more snippets truncated)")
            break
        sections.append(snippet)
        total += len(snippet)
    return "\n\n".join(sections)


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

# Relevant Code:
{_format_code_snippets(context.get('retrieved_code', {}))}

# Memory Context:
{context.get('memory_summary', 'No prior memory state.')}

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
You are adding a new feature to a Python repository.

# Repository Overview:
{context.get('repo_summary', 'No summary available.')}

# Module Summaries:
{', '.join([c['summary'] for c in context.get('module_summaries', [])])}

# Repository Structure:
{context.get('repo_structure', 'Not provided.')}

# Target folder for new code: {context.get('target_folder', 'new_folder')}
# Suggested imports: {context.get('imports', [])}

# Existing Code (for reference and integration):
{_format_code_snippets(context.get('retrieved_code', {}))}

# Memory Context:
{context.get('memory_summary', 'No prior memory state.')}

# Task:
{user_query}

Instructions:
- Write a Python file skeleton (do NOT overwrite existing files)
- Include proper imports, class/function definitions
- Reference the relevant functions/methods from the code above
- Make the code integrate with existing repo structure
- Do not write explanations, only code
"""
    return ask_llm(prompt)
