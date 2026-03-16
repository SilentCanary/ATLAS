def build_final_context(global_summary, cluster_summaries, structure, snippets):
    return {
        "repo_summary": global_summary,
        "module_summaries": cluster_summaries,
        "repo_structure": structure,
        "retrieved_code": snippets
    }