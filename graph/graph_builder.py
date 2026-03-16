import networkx as nx

class CodeGraphBuilder:

    def __init__(self):
        self.graph=nx.DiGraph()

    def build_graph(self,parsed_repo):

        function_lookup={}
        method_lookup={}

        for file_path, data in parsed_repo.items():

            for func_name in data.get("functions",{}):
                node=f"{file_path}::{func_name}"
                function_lookup[func_name]=node 

            for cls_name,cls_data in data.get("classes",{}).items():
                for method_name in cls_data.get("methods",{}):
                    key=f"{cls_name}.{method_name}"
                    node=f"{file_path}::{cls_name}::{method_name}"
                    method_lookup[key]=node
            
        for file_path, data in parsed_repo.items():
            self.graph.add_node(file_path,type="file")


            for imp in data.get("imports",[]):
                self.graph.add_node(imp,type="module")
                self.graph.add_edge(file_path,imp,relation="imports")


            for func_name, func_data in data.get("functions",{}).items():
                func_node=f"{file_path}::{func_name}"
                self.graph.add_node(
                    func_node,
                    type="function",
                    code=func_data["code"],
                    args=func_data["args"],
                    start_line=func_data["start_line"],
                    end_line=func_data["end_line"]
                )
                self.graph.add_edge(file_path,func_node,relation="defines")


            for cls_name, cls_data in data.get("classes", {}).items():

                cls_node = f"{file_path}::{cls_name}"

                self.graph.add_node(cls_node, type="class")
                self.graph.add_edge(file_path, cls_node, relation="defines")

                # Inheritance
                for parent in cls_data.get("inherits", []):
                    self.graph.add_node(parent, type="class")
                    self.graph.add_edge(cls_node, parent, relation="inherits")

                # Methods
                for method_name, method_data in cls_data.get("methods", {}).items():

                    method_node = f"{file_path}::{cls_name}::{method_name}"

                    self.graph.add_node(
                        method_node,
                        type="method",
                        code=method_data["code"],
                        args=method_data["args"],
                        start_line=method_data["start_line"],
                        end_line=method_data["end_line"]
                    )

                    self.graph.add_edge(cls_node, method_node, relation="defines")


        for file_path,data in parsed_repo.items():

            for func_name,func_data in data.get("functions",{}).items():
                func_node = f"{file_path}::{func_name}"

                for call in func_data.get("calls", []):

                    target = function_lookup.get(call)

                    if target:
                        self.graph.add_edge(func_node, target, relation="calls")
                    else:
                        self.graph.add_node(call, type="external_function")
                        self.graph.add_edge(func_node, call, relation="calls")

            for cls_name, cls_data in data.get("classes", {}).items():

                for method_name, method_data in cls_data.get("methods", {}).items():

                    method_node = f"{file_path}::{cls_name}::{method_name}"

                    for call in method_data.get("calls", []):

                        key = f"{cls_name}.{call}"
                        target = method_lookup.get(key) or function_lookup.get(call)

                        if target:
                            self.graph.add_edge(method_node, target, relation="calls")
                        else:
                            self.graph.add_node(call, type="external_function")
                            self.graph.add_edge(method_node, call, relation="calls")

        return self.graph