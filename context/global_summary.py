from .llm_call  import ask_llm
def build_global_prompt(cluster_summaries, repo_structure):
    prompt = "You are given a codebase summary.\n\n"

    prompt += "Repository Structure:\n"
    prompt += str(repo_structure) + "\n\n"

    prompt += "Module Summaries:\n"
    for c in cluster_summaries:
        prompt += f"- {c['summary']}\n"

    prompt += "\nExplain what this project does and how it is organized."

    return prompt

def generate_global_summary(cluster_summaries, structure):
    prompt = build_global_prompt(cluster_summaries, structure)
    return ask_llm(prompt)