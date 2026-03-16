from .llm_call import ask_llm
def build_cluster_prompt(cluster, contexts, snippets):
    text = "You are analyzing a codebase.\n\n"

    text += "Functions in this group:\n"
    for node in cluster:
        text += f"- {node}\n"

    text += "\nContext:\n"
    for ctx in contexts:
        if ctx["node"] in cluster:
            text += f"\nNode: {ctx['node']}\n"
            text += f"Upstream: {ctx['upstream']}\n"
            text += f"Downstream: {ctx['downstream']}\n"

    text += "\nCode Snippets:\n"
    for node in cluster:
        if node in snippets:
            text += f"\n{snippets[node]}\n"

    text += "\nSummarize what this group of functions does in 1-2 lines."

    return text

def summarize_clusters(clusters, contexts, snippets):
    summaries = []

    for cluster in clusters:
        prompt = build_cluster_prompt(cluster, contexts, snippets)
        summary = ask_llm(prompt)

        summaries.append({
            "cluster": cluster,
            "summary": summary
        })

    return summaries