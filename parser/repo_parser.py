import ast
import os

from parser.language_parsers import detect_language, get_parser, SUPPORTED_EXTENSIONS

class FunctionCallCollector(ast.NodeVisitor):
    #used to collect all the function and method call inside a function or method

    def __init__(self):
        self.calls=[]

    def visit_Call(self, node):
        if isinstance(node.func,ast.Name):
            self.calls.append(node.func.id)

        elif isinstance(node.func,ast.Attribute):
            name_chain=[]
            current=node.func

            while isinstance(current,ast.Attribute):
                name_chain.append(current.attr)
                current=current.value
            
            if isinstance(current,ast.Name):
                name_chain.append(current.id)

            full_name=".".join(reversed(name_chain))
            self.calls.append(full_name)
        self.generic_visit(node)

class CodeParser:

    def get_source_segment(self,file_path,node):
        with open(file_path,"r",encoding='utf-8') as f:
            source=f.read()
        return ast.get_source_segment(source,node)
    
    def extract_function_data(self,file_path,node):
        collector=FunctionCallCollector()
        collector.visit(node)

        code=self.get_source_segment(file_path,node)
        args=[arg.arg for arg in node.args.args]
        return {
            "args":args,
            "calls":collector.calls,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "code": code
        }
    
    def parse_file(self,file_path):

        with open(file_path,"r",encoding='utf-8') as f:
            source=f.read()
        
        try:
            tree=ast.parse(source)
        
        except SyntaxError:
            print(f"skipping file with syntax error ..  {file_path}")
            return None
        functions={}
        classes={}
        imports=[]

        for node in tree.body:
            if isinstance(node,ast.FunctionDef):
                functions[node.name]= self.extract_function_data(file_path,node)

            elif isinstance(node,ast.ClassDef):
                method_dict={}

                for item in node.body:
                    if isinstance(item,ast.FunctionDef):
                        method_dict[item.name]= self.extract_function_data(file_path,item)

                bases=[]
                for base in node.bases:
                    if isinstance(base,ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        name_chain = []
                        current = base
                        while isinstance(current, ast.Attribute):
                            name_chain.append(current.attr)
                            current = current.value
                        if isinstance(current, ast.Name):
                            name_chain.append(current.id)
                        bases.append(".".join(reversed(name_chain)))

                classes[node.name]={
                    "inherits":bases,
                    "methods":method_dict
                }
            elif isinstance(node,ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            
            elif isinstance(node,ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
            
        return {
            "file_path":file_path,
            "functions":functions,
            "classes":classes,
            "imports":imports
        }
    
    def parse_repository(self,repo_path):
        parsed_data={}

        for root, _, files in os.walk(repo_path):
            for file in files:
                full_path=os.path.join(root,file)
                _, ext = os.path.splitext(file)
                ext_lower = ext.lower()

                # Python: use existing high-fidelity AST parser
                if ext_lower == ".py":
                    result=self.parse_file(full_path)
                    if result:
                        result["folder"] = os.path.relpath(root, repo_path)
                        result["module"] = file.replace(".py", "")
                        result["language"] = "python"
                        parsed_data[full_path]=result

                # Other supported languages: dispatch to language parser
                elif ext_lower in SUPPORTED_EXTENSIONS:
                    # Skip binary files (null bytes in first 8KB)
                    try:
                        with open(full_path, "rb") as bf:
                            chunk = bf.read(8192)
                            if b"\x00" in chunk:
                                continue
                    except Exception:
                        continue

                    # Skip very large files
                    try:
                        if os.path.getsize(full_path) > 500_000:
                            continue
                    except Exception:
                        continue

                    language = detect_language(full_path)
                    lang_parser = get_parser(language)
                    result = lang_parser.parse_file(full_path)
                    if result:
                        result["folder"] = os.path.relpath(root, repo_path)
                        result["module"] = file.rsplit(".", 1)[0]
                        parsed_data[full_path] = result

        return parsed_data