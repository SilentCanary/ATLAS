from collections import deque


class GraphStore:
    def __init__(self, graph):
        self.graph = graph

    def get_downstream_impact(self, node, types=None):
        """Simple direct successors."""
        if node not in self.graph:
            return []
        neighbors = list(self.graph.successors(node))
        if types:
            neighbors = [n for n in neighbors if self.graph.nodes[n].get("type") in types]
        return neighbors

    def get_upstream_dependencies(self, node, types=None):
        """Simple direct predecessors."""
        if node not in self.graph:
            return []
        neighbors = list(self.graph.predecessors(node))
        if types:
            neighbors = [n for n in neighbors if self.graph.nodes[n].get("type") in types]
        return neighbors

    def get_full_downstream(self, node, types=None, max_depth=3):
        """Recursive downstream BFS with depth limit."""
        if node not in self.graph:
            return []

        visited = set()
        queue = deque([(node, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        if types:
            return [n for n in visited if self.graph.nodes[n].get("type") in types]
        return list(visited)

    def get_full_upstream(self, node, types=None, max_depth=3):
        """Recursive upstream BFS with depth limit."""
        if node not in self.graph:
            return []

        visited = set()
        queue = deque([(node, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self.graph.predecessors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        if types:
            return [n for n in visited if self.graph.nodes[n].get("type") in types]
        return list(visited)

    def get_function_downstream(self, node):
        return self.get_full_downstream(node, types=["function"])

    def get_method_downstream(self, node):
        return self.get_full_downstream(node, types=["method"])

    def get_folder_downstream(self, node):
        return self.get_full_downstream(node, types=["folder"])

    def get_module_downstream(self, node):
        return self.get_full_downstream(node, types=["module"])
