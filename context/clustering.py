from collections import defaultdict

def cluster_by_files(nodes):
    clusters=defaultdict(list)
    for node in nodes:
        file_path=node.split("::")[0]
        clusters[file_path].append(node)
    
    return list(clusters.values())

def cluster_scc(graph, nodes):
    visited = set()
    clusters = []

    def dfs(node, current_cluster):
        visited.add(node)
        current_cluster.append(node)

        for nei in graph.get_neighbors(node):
            if nei not in visited:
                dfs(nei, current_cluster)

    for node in nodes:
        if node not in visited:
            cluster = []
            dfs(node, cluster)
            clusters.append(cluster)

    return clusters