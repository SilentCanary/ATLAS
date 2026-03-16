import ast
import importlib
import sys
from typing import Tuple


class CodeValidator:
    """Validates generated Python code before writing to disk."""

    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True, "Syntax OK"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def validate_imports(self, code: str) -> Tuple[bool, str]:
        """Check if imported modules are available."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, "Cannot check imports: syntax error"

        missing = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if not self._module_available(mod):
                        missing.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".")[0]
                    if not self._module_available(mod):
                        missing.append(node.module)

        if missing:
            return False, f"Missing modules: {', '.join(missing)}"
        return True, "All imports available"

    def _module_available(self, module_name: str) -> bool:
        """Check if a module can be found (without importing it)."""
        if module_name in sys.modules:
            return True
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ModuleNotFoundError, ValueError):
            return False

    def validate(self, code: str) -> Tuple[bool, list]:
        """Run all validations, return (pass, list_of_issues)."""
        issues = []

        ok, msg = self.validate_syntax(code)
        if not ok:
            issues.append(msg)
            return False, issues  # can't continue without valid syntax

        ok, msg = self.validate_imports(code)
        if not ok:
            issues.append(msg)

        return len(issues) == 0, issues
