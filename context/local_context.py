def get_local_context(graph,node,types=None):
    return{
        "node":node,
        "upstream":graph.get_full_upstream(node,types=types),
        "downstream":graph.get_full_downstream(node,types=types)
    }

def build_context_for_nodes(graph,nodes,types=None):
    contexts=[]
    for node in nodes:
        ctx=get_local_context(graph,node,types=types)
        contexts.append(ctx)

    return contexts
