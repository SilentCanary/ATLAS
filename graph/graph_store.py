class GraphStore:
    def __init__(self, graph):
        self.graph = graph

    # Simple direct successors
    def get_downstream_impact(self, node, types=None):
     
        if node not in self.graph:
            return []

        neighbors = list(self.graph.successors(node))
        if types:
            neighbors = [n for n in neighbors if self.graph.nodes[n].get("type") in types]
        return neighbors

    # Simple direct predecessors
    def get_upstream_dependencies(self, node, types=None):
        if node not in self.graph:
            return []

        neighbors = list(self.graph.predecessors(node))
        if types:
            neighbors = [n for n in neighbors if self.graph.nodes[n].get("type") in types]
        return neighbors

    # Recursive downstream (BFS)
    def get_full_downstream(self, node, types=None):
        if node not in self.graph:
            return []

        visited = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if types:
            return [n for n in visited if self.graph.nodes[n].get("type") in types]
        return list(visited)

    # Recursive upstream (BFS)
    def get_full_upstream(self, node, types=None):
        if node not in self.graph:
            return []

        visited = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for neighbor in self.graph.predecessors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if types:
            return [n for n in visited if self.graph.nodes[n].get("type") in types]
        return list(visited)

    # Extra utility: get all functions in downstream recursively
    def get_function_downstream(self, node):
        return self.get_full_downstream(node, types=["function"])

    # Extra utility: get all methods in downstream recursively
    def get_method_downstream(self, node):
        return self.get_full_downstream(node, types=["method"])
    
    # get all folders downstream
    def get_folder_downstream(self, node):
        return self.get_full_downstream(node, types=["folder"])

    # get all modules downstream
    def get_module_downstream(self, node):
        return self.get_full_downstream(node, types=["module"])