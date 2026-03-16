from graph.graph_store import GraphStore

def build_context(retriever,graph_store:GraphStore,query:str,top_k:int=5):
    results=retriever.retrieve(query)
    folders=[]
    for node,_ in results:
        preds=graph_store.get_upstream_dependencies(node,types=["file"])
        if preds:
            folders.append(preds[0])
    target_folder=max(set(folders),key=folders.count) if folders else "new_folder"

    imports=set()
    for node,_ in results:
        preds=graph_store.get_upstream_dependencies(node,types=["file"])
        imports.update(preds)

    relevant_code={}
    for node,_ in results[:top_k]:
        node_data=graph_store.graph.nodes.get(node)
        if node_data and "code" in node_data:
            relevant_code[node]=node_data["code"]
    
    return {
        "target_folder":target_folder,
        "imports":imports,
        "relevant_code_snippets":relevant_code
    }