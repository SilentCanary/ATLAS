"""
Multi-language parser support for ATLAS.

Uses tree-sitter for C, C++, Go, JavaScript, TypeScript, Rust, Java.
Uses regex for Assembly (NASM/GAS).
Falls back to raw file storage for unsupported languages.

All parsers produce the same dict format consumed by graph_builder and embeddings.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

EXTENSION_MAP: Dict[str, str] = {
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".go": "go",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust",
    ".java": "java",
    ".sol": "solidity",
    ".asm": "assembly", ".s": "assembly", ".S": "assembly",
}

SUPPORTED_EXTENSIONS: set = set(EXTENSION_MAP.keys())


def detect_language(file_path: str) -> Optional[str]:
    """Return language name from file extension, or None."""
    _, ext = os.path.splitext(file_path)
    return EXTENSION_MAP.get(ext.lower())


# ---------------------------------------------------------------------------
# Base parser
# ---------------------------------------------------------------------------

class BaseLanguageParser(ABC):
    """All language parsers must produce the standard ATLAS dict format."""

    language: str = "unknown"

    @staticmethod
    def _make_func_dict(name: str, args: List[str], calls: List[str],
                        start_line: int, end_line: int, code: str) -> dict:
        """Build a function/method dict that satisfies the downstream contract."""
        return {
            "args": args or [],
            "calls": calls or [],
            "start_line": start_line,
            "end_line": end_line,
            "code": code or "",
        }

    def _empty_result(self, file_path: str) -> dict:
        return {
            "file_path": file_path,
            "language": self.language,
            "imports": [],
            "functions": {},
            "classes": {},
        }

    @abstractmethod
    def parse_file(self, file_path: str) -> Optional[dict]:
        ...


# ---------------------------------------------------------------------------
# Tree-sitter base
# ---------------------------------------------------------------------------

class TreeSitterParser(BaseLanguageParser):
    """Shared logic for all tree-sitter-based parsers."""

    ts_language: str = ""

    _parser_cache: dict = {}  # class-level cache

    def _get_parser(self):
        if self.ts_language not in TreeSitterParser._parser_cache:
            import tree_sitter_languages
            TreeSitterParser._parser_cache[self.ts_language] = (
                tree_sitter_languages.get_parser(self.ts_language)
            )
        return TreeSitterParser._parser_cache[self.ts_language]

    @staticmethod
    def _node_text(node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _extract_calls(self, node, source_bytes: bytes) -> List[str]:
        calls = []
        self._walk_calls(node, source_bytes, calls)
        return calls

    def _walk_calls(self, node, source_bytes: bytes, calls: List[str]):
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                calls.append(self._node_text(func_node, source_bytes))
        for child in node.children:
            self._walk_calls(child, source_bytes, calls)

    def parse_file(self, file_path: str) -> Optional[dict]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception:
            return None

        source_bytes = source.encode("utf-8")

        try:
            parser = self._get_parser()
            tree = parser.parse(source_bytes)
        except Exception:
            # Fallback: store entire file as one function
            result = self._empty_result(file_path)
            lines = source.split("\n")
            result["functions"]["__file__"] = self._make_func_dict(
                "__file__", [], [], 1, len(lines), source)
            return result

        result = self._empty_result(file_path)
        result["imports"] = self._extract_imports(tree, source_bytes)
        self._extract_functions(tree, source_bytes, result)
        self._extract_classes(tree, source_bytes, result)
        return result

    def _extract_imports(self, tree, source_bytes: bytes) -> List[str]:
        return []

    def _extract_functions(self, tree, source_bytes: bytes, result: dict):
        pass

    def _extract_classes(self, tree, source_bytes: bytes, result: dict):
        pass


# ---------------------------------------------------------------------------
# C Parser
# ---------------------------------------------------------------------------

class CParser(TreeSitterParser):
    language = "c"
    ts_language = "c"

    def _extract_imports(self, tree, source_bytes):
        imports = []
        for child in tree.root_node.children:
            if child.type == "preproc_include":
                path_node = child.child_by_field_name("path")
                if path_node:
                    imports.append(
                        self._node_text(path_node, source_bytes).strip('"<>'))
        return imports

    def _extract_functions(self, tree, source_bytes, result):
        for child in tree.root_node.children:
            if child.type == "function_definition":
                declarator = child.child_by_field_name("declarator")
                name = self._get_func_name(declarator, source_bytes)
                if name:
                    params = self._get_params(declarator, source_bytes)
                    calls = self._extract_calls(child, source_bytes)
                    code = self._node_text(child, source_bytes)
                    result["functions"][name] = self._make_func_dict(
                        name, params, calls,
                        child.start_point[0] + 1,
                        child.end_point[0] + 1,
                        code)

    def _get_func_name(self, declarator, source_bytes):
        if declarator is None:
            return None
        if declarator.type == "function_declarator":
            name_node = declarator.child_by_field_name("declarator")
            if name_node:
                return self._node_text(name_node, source_bytes)
        for child in declarator.children:
            result = self._get_func_name(child, source_bytes)
            if result:
                return result
        return None

    def _get_params(self, declarator, source_bytes):
        params = []
        if declarator is None:
            return params
        if declarator.type == "function_declarator":
            param_list = declarator.child_by_field_name("parameters")
            if param_list:
                for p in param_list.children:
                    if p.type == "parameter_declaration":
                        decl = p.child_by_field_name("declarator")
                        if decl:
                            params.append(self._node_text(decl, source_bytes))
        return params

    def _extract_classes(self, tree, source_bytes, result):
        """C structs map to classes."""
        for child in tree.root_node.children:
            self._find_structs(child, source_bytes, result)

    def _find_structs(self, node, source_bytes, result):
        if node.type in ("struct_specifier", "type_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._node_text(name_node, source_bytes)
                if name not in result["classes"]:
                    result["classes"][name] = {"inherits": [], "methods": {}}
        for child in node.children:
            if child.type in ("struct_specifier",):
                self._find_structs(child, source_bytes, result)


# ---------------------------------------------------------------------------
# C++ Parser
# ---------------------------------------------------------------------------

class CppParser(CParser):
    language = "cpp"
    ts_language = "cpp"

    def _extract_classes(self, tree, source_bytes, result):
        super()._extract_classes(tree, source_bytes, result)
        for child in tree.root_node.children:
            if child.type == "class_specifier":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                name = self._node_text(name_node, source_bytes)
                inherits = []
                methods = {}

                body = child.child_by_field_name("body")
                if body:
                    for item in body.children:
                        if item.type == "function_definition":
                            m_decl = item.child_by_field_name("declarator")
                            m_name = self._get_func_name(m_decl, source_bytes)
                            if m_name:
                                m_params = self._get_params(m_decl, source_bytes)
                                m_calls = self._extract_calls(item, source_bytes)
                                m_code = self._node_text(item, source_bytes)
                                methods[m_name] = self._make_func_dict(
                                    m_name, m_params, m_calls,
                                    item.start_point[0] + 1,
                                    item.end_point[0] + 1, m_code)

                result["classes"][name] = {
                    "inherits": inherits, "methods": methods}


# ---------------------------------------------------------------------------
# Go Parser
# ---------------------------------------------------------------------------

class GoParser(TreeSitterParser):
    language = "go"
    ts_language = "go"

    def _extract_imports(self, tree, source_bytes):
        imports = []
        self._walk_imports(tree.root_node, source_bytes, imports)
        return imports

    def _walk_imports(self, node, source_bytes, imports):
        if node.type == "import_spec":
            path_node = node.child_by_field_name("path")
            if path_node:
                imports.append(
                    self._node_text(path_node, source_bytes).strip('"'))
        for child in node.children:
            self._walk_imports(child, source_bytes, imports)

    def _extract_functions(self, tree, source_bytes, result):
        for child in tree.root_node.children:
            if child.type == "function_declaration":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                name = self._node_text(name_node, source_bytes)
                params = self._go_params(child, source_bytes)
                calls = self._extract_calls(child, source_bytes)
                code = self._node_text(child, source_bytes)
                result["functions"][name] = self._make_func_dict(
                    name, params, calls,
                    child.start_point[0] + 1,
                    child.end_point[0] + 1, code)

            elif child.type == "method_declaration":
                name_node = child.child_by_field_name("name")
                receiver_node = child.child_by_field_name("receiver")
                if not name_node or not receiver_node:
                    continue
                method_name = self._node_text(name_node, source_bytes)
                receiver_type = self._go_receiver_type(receiver_node, source_bytes)
                params = self._go_params(child, source_bytes)
                calls = self._extract_calls(child, source_bytes)
                code = self._node_text(child, source_bytes)

                if receiver_type not in result["classes"]:
                    result["classes"][receiver_type] = {
                        "inherits": [], "methods": {}}
                result["classes"][receiver_type]["methods"][method_name] = (
                    self._make_func_dict(method_name, params, calls,
                                         child.start_point[0] + 1,
                                         child.end_point[0] + 1, code))

    def _go_params(self, func_node, source_bytes):
        params = []
        param_list = func_node.child_by_field_name("parameters")
        if param_list:
            for p in param_list.children:
                if p.type == "parameter_declaration":
                    name_node = p.child_by_field_name("name")
                    if name_node:
                        params.append(self._node_text(name_node, source_bytes))
        return params

    def _go_receiver_type(self, receiver_node, source_bytes):
        for child in receiver_node.children:
            if child.type == "parameter_declaration":
                type_node = child.child_by_field_name("type")
                if type_node:
                    return self._node_text(type_node, source_bytes).lstrip("*")
        return "Unknown"

    def _extract_classes(self, tree, source_bytes, result):
        for child in tree.root_node.children:
            if child.type == "type_declaration":
                for spec in child.children:
                    if spec.type == "type_spec":
                        name_node = spec.child_by_field_name("name")
                        type_node = spec.child_by_field_name("type")
                        if name_node and type_node and type_node.type in (
                                "struct_type", "interface_type"):
                            name = self._node_text(name_node, source_bytes)
                            if name not in result["classes"]:
                                result["classes"][name] = {
                                    "inherits": [], "methods": {}}


# ---------------------------------------------------------------------------
# JavaScript Parser
# ---------------------------------------------------------------------------

class JavaScriptParser(TreeSitterParser):
    language = "javascript"
    ts_language = "javascript"

    def _extract_imports(self, tree, source_bytes):
        imports = []
        self._walk_js_imports(tree.root_node, source_bytes, imports)
        return imports

    def _walk_js_imports(self, node, source_bytes, imports):
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source:
                imports.append(
                    self._node_text(source, source_bytes).strip("'\""))
        for c in node.children:
            self._walk_js_imports(c, source_bytes, imports)

    def _extract_functions(self, tree, source_bytes, result):
        for child in tree.root_node.children:
            self._walk_js_functions(child, source_bytes, result)

    def _walk_js_functions(self, node, source_bytes, result):
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._node_text(name_node, source_bytes)
                params = self._js_params(node, source_bytes)
                calls = self._extract_calls(node, source_bytes)
                code = self._node_text(node, source_bytes)
                result["functions"][name] = self._make_func_dict(
                    name, params, calls,
                    node.start_point[0] + 1, node.end_point[0] + 1, code)

        elif node.type == "export_statement":
            for child in node.children:
                self._walk_js_functions(child, source_bytes, result)
                self._walk_js_classes(child, source_bytes, result)

        elif node.type == "lexical_declaration":
            for decl in node.children:
                if decl.type == "variable_declarator":
                    name_node = decl.child_by_field_name("name")
                    value_node = decl.child_by_field_name("value")
                    if (name_node and value_node
                            and value_node.type == "arrow_function"):
                        name = self._node_text(name_node, source_bytes)
                        params = self._js_arrow_params(value_node, source_bytes)
                        calls = self._extract_calls(value_node, source_bytes)
                        code = self._node_text(node, source_bytes)
                        result["functions"][name] = self._make_func_dict(
                            name, params, calls,
                            node.start_point[0] + 1,
                            node.end_point[0] + 1, code)

    def _js_params(self, func_node, source_bytes):
        params = []
        param_node = func_node.child_by_field_name("parameters")
        if param_node:
            for p in param_node.children:
                if p.type in ("identifier", "required_parameter",
                              "optional_parameter"):
                    params.append(self._node_text(p, source_bytes))
        return params

    def _js_arrow_params(self, arrow_node, source_bytes):
        params = []
        param_node = arrow_node.child_by_field_name("parameters")
        if param_node:
            for p in param_node.children:
                if p.type in ("identifier", "required_parameter"):
                    params.append(self._node_text(p, source_bytes))
        elif arrow_node.child_by_field_name("parameter"):
            params.append(self._node_text(
                arrow_node.child_by_field_name("parameter"), source_bytes))
        return params

    def _extract_classes(self, tree, source_bytes, result):
        for child in tree.root_node.children:
            self._walk_js_classes(child, source_bytes, result)

    def _walk_js_classes(self, node, source_bytes, result):
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            name = self._node_text(name_node, source_bytes)
            inherits = []
            for child in node.children:
                if child.type == "class_heritage":
                    for c in child.children:
                        if c.type == "identifier":
                            inherits.append(
                                self._node_text(c, source_bytes))

            methods = {}
            body = node.child_by_field_name("body")
            if body:
                for item in body.children:
                    if item.type == "method_definition":
                        m_name_node = item.child_by_field_name("name")
                        if m_name_node:
                            m_name = self._node_text(m_name_node, source_bytes)
                            m_params = self._js_params(item, source_bytes)
                            m_calls = self._extract_calls(item, source_bytes)
                            m_code = self._node_text(item, source_bytes)
                            methods[m_name] = self._make_func_dict(
                                m_name, m_params, m_calls,
                                item.start_point[0] + 1,
                                item.end_point[0] + 1, m_code)

            result["classes"][name] = {"inherits": inherits, "methods": methods}

        elif node.type == "export_statement":
            for child in node.children:
                self._walk_js_classes(child, source_bytes, result)


# ---------------------------------------------------------------------------
# TypeScript Parser (extends JavaScript)
# ---------------------------------------------------------------------------

class TypeScriptParser(JavaScriptParser):
    language = "typescript"
    ts_language = "typescript"


# ---------------------------------------------------------------------------
# Rust Parser
# ---------------------------------------------------------------------------

class RustParser(TreeSitterParser):
    language = "rust"
    ts_language = "rust"

    def _extract_imports(self, tree, source_bytes):
        imports = []
        for child in tree.root_node.children:
            if child.type == "use_declaration":
                arg = child.child_by_field_name("argument")
                if arg:
                    imports.append(self._node_text(arg, source_bytes))
        return imports

    def _extract_functions(self, tree, source_bytes, result):
        for child in tree.root_node.children:
            if child.type == "function_item":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                name = self._node_text(name_node, source_bytes)
                params = self._rust_params(child, source_bytes)
                calls = self._extract_calls(child, source_bytes)
                code = self._node_text(child, source_bytes)
                result["functions"][name] = self._make_func_dict(
                    name, params, calls,
                    child.start_point[0] + 1,
                    child.end_point[0] + 1, code)

    def _rust_params(self, func_node, source_bytes):
        params = []
        param_list = func_node.child_by_field_name("parameters")
        if param_list:
            for p in param_list.children:
                if p.type == "parameter":
                    pat = p.child_by_field_name("pattern")
                    if pat:
                        params.append(self._node_text(pat, source_bytes))
                elif p.type == "self_parameter":
                    params.append("self")
        return params

    def _extract_classes(self, tree, source_bytes, result):
        # First pass: struct definitions
        for child in tree.root_node.children:
            if child.type == "struct_item":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node, source_bytes)
                    if name not in result["classes"]:
                        result["classes"][name] = {
                            "inherits": [], "methods": {}}

        # Second pass: impl blocks -> methods
        for child in tree.root_node.children:
            if child.type == "impl_item":
                type_node = child.child_by_field_name("type")
                trait_node = child.child_by_field_name("trait")
                if not type_node:
                    continue
                struct_name = self._node_text(type_node, source_bytes)
                inherits = []
                if trait_node:
                    inherits.append(self._node_text(trait_node, source_bytes))

                if struct_name not in result["classes"]:
                    result["classes"][struct_name] = {
                        "inherits": inherits, "methods": {}}
                elif inherits:
                    result["classes"][struct_name]["inherits"].extend(inherits)

                body = child.child_by_field_name("body")
                if body:
                    for item in body.children:
                        if item.type == "function_item":
                            m_name_node = item.child_by_field_name("name")
                            if m_name_node:
                                m_name = self._node_text(
                                    m_name_node, source_bytes)
                                m_params = self._rust_params(
                                    item, source_bytes)
                                m_calls = self._extract_calls(
                                    item, source_bytes)
                                m_code = self._node_text(item, source_bytes)
                                result["classes"][struct_name]["methods"][m_name] = (
                                    self._make_func_dict(
                                        m_name, m_params, m_calls,
                                        item.start_point[0] + 1,
                                        item.end_point[0] + 1, m_code))


# ---------------------------------------------------------------------------
# Java Parser
# ---------------------------------------------------------------------------

class JavaParser(TreeSitterParser):
    language = "java"
    ts_language = "java"

    def _extract_imports(self, tree, source_bytes):
        imports = []
        for child in tree.root_node.children:
            if child.type == "import_declaration":
                for c in child.children:
                    if c.type == "scoped_identifier":
                        imports.append(self._node_text(c, source_bytes))
        return imports

    def _extract_classes(self, tree, source_bytes, result):
        for child in tree.root_node.children:
            if child.type == "class_declaration":
                self._parse_java_class(child, source_bytes, result)

    def _parse_java_class(self, node, source_bytes, result):
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        class_name = self._node_text(name_node, source_bytes)
        inherits = []
        superclass = node.child_by_field_name("superclass")
        if superclass:
            inherits.append(self._node_text(superclass, source_bytes))

        methods = {}
        body = node.child_by_field_name("body")
        if body:
            for item in body.children:
                if item.type in ("method_declaration",
                                 "constructor_declaration"):
                    m_name = item.child_by_field_name("name")
                    if m_name:
                        m_name_str = self._node_text(m_name, source_bytes)
                        params = self._java_params(item, source_bytes)
                        calls = self._extract_calls(item, source_bytes)
                        code = self._node_text(item, source_bytes)
                        methods[m_name_str] = self._make_func_dict(
                            m_name_str, params, calls,
                            item.start_point[0] + 1,
                            item.end_point[0] + 1, code)

        result["classes"][class_name] = {
            "inherits": inherits, "methods": methods}

    def _java_params(self, method_node, source_bytes):
        params = []
        param_list = method_node.child_by_field_name("parameters")
        if param_list:
            for p in param_list.children:
                if p.type == "formal_parameter":
                    name = p.child_by_field_name("name")
                    if name:
                        params.append(self._node_text(name, source_bytes))
        return params


# ---------------------------------------------------------------------------
# Solidity Parser (tree-sitter with fallback to regex)
# ---------------------------------------------------------------------------

class SolidityParser(BaseLanguageParser):
    """Regex-based Solidity parser (tree-sitter grammar not always available)."""
    language = "solidity"

    _CONTRACT_RE = re.compile(
        r'(?:contract|interface|library)\s+(\w+)'
        r'(?:\s+is\s+([\w,\s]+))?\s*\{', re.MULTILINE)
    _FUNC_RE = re.compile(
        r'function\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE)
    _IMPORT_RE = re.compile(
        r'import\s+[^;]*?["\']([^"\']+)["\']', re.MULTILINE)

    def parse_file(self, file_path: str) -> Optional[dict]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception:
            return None

        result = self._empty_result(file_path)
        lines = source.split("\n")

        # Imports
        for m in self._IMPORT_RE.finditer(source):
            result["imports"].append(m.group(1))

        # Contracts as classes
        for m in self._CONTRACT_RE.finditer(source):
            name = m.group(1)
            inherits = []
            if m.group(2):
                inherits = [x.strip() for x in m.group(2).split(",")]
            result["classes"][name] = {"inherits": inherits, "methods": {}}

        # Functions
        for m in self._FUNC_RE.finditer(source):
            fname = m.group(1)
            params_str = m.group(2).strip()
            args = []
            if params_str:
                for param in params_str.split(","):
                    parts = param.strip().split()
                    if len(parts) >= 2:
                        args.append(parts[-1])
                    elif len(parts) == 1:
                        args.append(parts[0])

            line_no = source[:m.start()].count("\n") + 1
            # Find approximate end (next function or end of file)
            next_func = self._FUNC_RE.search(source, m.end())
            end_pos = next_func.start() if next_func else len(source)
            end_line = source[:end_pos].count("\n") + 1
            code = source[m.start():end_pos].rstrip()

            # Assign to the nearest contract or as top-level
            assigned = False
            for cls_name, cls_data in result["classes"].items():
                cls_match = re.search(
                    rf'(?:contract|interface|library)\s+{re.escape(cls_name)}',
                    source)
                if cls_match and cls_match.start() < m.start():
                    cls_data["methods"][fname] = self._make_func_dict(
                        fname, args, [], line_no, end_line, code)
                    assigned = True
                    break
            if not assigned:
                result["functions"][fname] = self._make_func_dict(
                    fname, args, [], line_no, end_line, code)

        return result


# ---------------------------------------------------------------------------
# Assembly Parser (regex-based, NASM + GAS)
# ---------------------------------------------------------------------------

class AssemblyParser(BaseLanguageParser):
    language = "assembly"

    _LABEL_RE = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*):', re.MULTILINE)
    _EXTERN_RE = re.compile(r'^\s*extern\s+(\S+)', re.MULTILINE | re.IGNORECASE)
    _CALL_RE = re.compile(r'^\s*call\s+(\S+)', re.MULTILINE | re.IGNORECASE)
    _INCLUDE_RE = re.compile(r'^\s*%include\s+"([^"]+)"', re.MULTILINE)
    _GAS_EXTERN_RE = re.compile(r'^\s*\.extern\s+(\S+)', re.MULTILINE)
    _GAS_INCLUDE_RE = re.compile(r'^\s*\.include\s+"([^"]+)"', re.MULTILINE)

    def parse_file(self, file_path: str) -> Optional[dict]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception:
            return None

        lines = source.split("\n")
        result = self._empty_result(file_path)

        # Imports
        for m in self._EXTERN_RE.finditer(source):
            result["imports"].append(m.group(1))
        for m in self._GAS_EXTERN_RE.finditer(source):
            result["imports"].append(m.group(1))
        for m in self._INCLUDE_RE.finditer(source):
            result["imports"].append(m.group(1))
        for m in self._GAS_INCLUDE_RE.finditer(source):
            result["imports"].append(m.group(1))

        # Labels as functions
        labels = []
        for m in self._LABEL_RE.finditer(source):
            label_name = m.group(1)
            if label_name.startswith("."):
                continue
            line_no = source[:m.start()].count("\n") + 1
            labels.append((label_name, m.start(), line_no))

        for i, (label_name, start_pos, start_line) in enumerate(labels):
            if i + 1 < len(labels):
                end_pos = labels[i + 1][1]
                end_line = labels[i + 1][2] - 1
            else:
                end_pos = len(source)
                end_line = len(lines)

            code_block = source[start_pos:end_pos].rstrip()
            calls = [m.group(1)
                     for m in self._CALL_RE.finditer(code_block)]

            result["functions"][label_name] = self._make_func_dict(
                label_name, [], calls, start_line, end_line, code_block)

        # If no labels, store whole file
        if not result["functions"]:
            all_calls = [m.group(1) for m in self._CALL_RE.finditer(source)]
            result["functions"]["__file__"] = self._make_func_dict(
                "__file__", [], all_calls, 1, len(lines), source)

        return result


# ---------------------------------------------------------------------------
# Fallback Parser (unknown languages)
# ---------------------------------------------------------------------------

class FallbackParser(BaseLanguageParser):
    language = "unknown"

    def parse_file(self, file_path: str) -> Optional[dict]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception:
            return None

        lines = source.split("\n")
        result = self._empty_result(file_path)
        result["functions"]["__file__"] = self._make_func_dict(
            "__file__", [], [], 1, len(lines), source)
        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PARSER_REGISTRY: Dict[str, BaseLanguageParser] = {
    "c": CParser(),
    "cpp": CppParser(),
    "go": GoParser(),
    "javascript": JavaScriptParser(),
    "typescript": TypeScriptParser(),
    "rust": RustParser(),
    "java": JavaParser(),
    "solidity": SolidityParser(),
    "assembly": AssemblyParser(),
}

_fallback = FallbackParser()


def get_parser(language: str) -> BaseLanguageParser:
    """Get the parser for a language, falling back to FallbackParser."""
    return PARSER_REGISTRY.get(language, _fallback)
