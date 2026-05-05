import ast
import glob
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from typing import Tuple, Dict, List, Set


_EXT_LANGUAGE_MAP = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".sh": "shell",
    ".bat": "batch",
    ".cmd": "batch",
    ".ps1": "powershell",
}


class CodeValidator:
    """Validates generated code before writing to disk."""

    def __init__(
        self,
        repo_path: str = "",
        run_command: str = "",
        test_command: str = "",
        timeout_sec: int = 45,
    ):
        self.repo_path = os.path.abspath(repo_path or os.getcwd())
        self.run_command = run_command or os.getenv("ATLAS_RUN_CMD", "")
        self.test_command = test_command or os.getenv("ATLAS_TEST_CMD", "")
        self.timeout_sec = timeout_sec
        self.repo_signals = self._detect_repo_signals()
        self.package_scripts = self._load_package_scripts()

    def validate_syntax(self, code: str, file_label: str = "") -> Tuple[bool, str]:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True, "Syntax OK"
        except SyntaxError as e:
            prefix = f"{file_label}: " if file_label else ""
            return False, f"{prefix}Syntax error at line {e.lineno}: {e.msg}"

    def validate_imports(self, code: str, file_label: str = "") -> Tuple[bool, str]:
        """Check if imported modules are available."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            prefix = f"{file_label}: " if file_label else ""
            return False, f"{prefix}Cannot check imports: syntax error"

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
            prefix = f"{file_label}: " if file_label else ""
            return False, f"{prefix}Missing modules: {', '.join(missing)}"
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

    def _run_command(self, command: List[str], cwd: str) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout_sec,
            )
            if result.returncode != 0:
                return False, result.stdout.strip()
            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Command failed: {e}"

    def _detect_repo_signals(self) -> Dict[str, str]:
        signals = {}
        candidates = {
            "pyproject": "pyproject.toml",
            "requirements": "requirements.txt",
            "setup_py": "setup.py",
            "pipfile": "Pipfile",
            "poetry_lock": "poetry.lock",
            "package_json": "package.json",
            "tsconfig": "tsconfig.json",
            "go_mod": "go.mod",
            "cargo_toml": "Cargo.toml",
            "pom_xml": "pom.xml",
            "gradle": "build.gradle",
            "gradle_kts": "build.gradle.kts",
            "gradle_wrapper": "gradlew",
            "gradle_wrapper_bat": "gradlew.bat",
            "makefile": "Makefile",
            "cmake": "CMakeLists.txt",
        }
        for key, name in candidates.items():
            path = os.path.join(self.repo_path, name)
            if os.path.exists(path):
                signals[key] = path
        return signals

    def _load_package_scripts(self) -> Dict[str, str]:
        package_path = self.repo_signals.get("package_json")
        if not package_path:
            return {}
        try:
            with open(package_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("scripts", {}) or {}
        except Exception:
            return {}

    def _language_for_path(self, rel_path: str) -> str:
        base = os.path.basename(rel_path)
        if base.startswith("Dockerfile"):
            return "dockerfile"
        _, ext = os.path.splitext(base)
        ext = ext.lower()
        if ext in _EXT_LANGUAGE_MAP:
            return _EXT_LANGUAGE_MAP[ext]
        return "unknown"

    def _default_language_from_repo(self) -> str:
        if self.repo_signals.get("go_mod"):
            return "go"
        if self.repo_signals.get("cargo_toml"):
            return "rust"
        if self.repo_signals.get("pom_xml") or self.repo_signals.get("gradle") or self.repo_signals.get("gradle_kts"):
            return "java"
        if self.repo_signals.get("tsconfig"):
            return "typescript"
        if self.repo_signals.get("package_json"):
            return "javascript"
        return "unknown"

    def _collect_languages(self, file_map: Dict[str, str]) -> Set[str]:
        languages = set()
        for rel_path in file_map.keys():
            languages.add(self._language_for_path(rel_path))
        if self.repo_signals.get("pyproject") or self.repo_signals.get("requirements") or self.repo_signals.get("setup_py"):
            languages.add("python")
        if self.repo_signals.get("pipfile") or self.repo_signals.get("poetry_lock"):
            languages.add("python")
        if self.repo_signals.get("package_json"):
            languages.add("javascript")
        if self.repo_signals.get("tsconfig"):
            languages.add("typescript")
        if self.repo_signals.get("go_mod"):
            languages.add("go")
        if self.repo_signals.get("cargo_toml"):
            languages.add("rust")
        if self.repo_signals.get("pom_xml") or self.repo_signals.get("gradle") or self.repo_signals.get("gradle_kts"):
            languages.add("java")
        if self.repo_signals.get("cmake") or self.repo_signals.get("makefile"):
            languages.add("cpp")
        return languages

    def _get_language_commands(self, language: str) -> Tuple[List[List[str]], str]:
        commands: List[List[str]] = []
        reason = ""
        lang = (language or "").lower()

        if lang in {"javascript", "typescript"}:
            if not self.repo_signals.get("package_json"):
                return [], "package.json not found"
            if shutil.which("npm") is None:
                return [], "npm not available"

            if "build" in self.package_scripts:
                commands.append(["npm", "run", "build"])
            if "test" in self.package_scripts:
                commands.append(["npm", "test"])

            if not commands and lang == "typescript":
                if self.repo_signals.get("tsconfig"):
                    if shutil.which("npx"):
                        commands.append(["npx", "tsc", "-p", "tsconfig.json", "--noEmit"])
                    elif shutil.which("tsc"):
                        commands.append(["tsc", "-p", "tsconfig.json", "--noEmit"])

            if not commands:
                reason = "no npm build/test scripts found"
            return commands, reason

        if lang == "go":
            if not self.repo_signals.get("go_mod"):
                return [], "go.mod not found"
            if shutil.which("go") is None:
                return [], "go not available"
            return [["go", "test", "./..."]], ""

        if lang == "rust":
            if not self.repo_signals.get("cargo_toml"):
                return [], "Cargo.toml not found"
            if shutil.which("cargo") is None:
                return [], "cargo not available"
            return [["cargo", "test"]], ""

        if lang == "java":
            if self.repo_signals.get("pom_xml") and shutil.which("mvn"):
                return [["mvn", "-q", "test"]], ""
            if os.path.exists(os.path.join(self.repo_path, "gradlew")):
                return [["./gradlew", "test"]], ""
            if os.path.exists(os.path.join(self.repo_path, "gradlew.bat")):
                return [["gradlew.bat", "test"]], ""
            if (self.repo_signals.get("gradle") or self.repo_signals.get("gradle_kts")) and shutil.which("gradle"):
                return [["gradle", "test"]], ""
            return [], "no Maven/Gradle command available"

        if lang in {"c", "cpp"}:
            if self.repo_signals.get("makefile") and shutil.which("make"):
                return [["make"]], ""
            if self.repo_signals.get("cmake") and shutil.which("cmake"):
                return [["cmake", "--build", "."]], ""
            return [], "no build system detected"

        if lang == "shell":
            if shutil.which("sh"):
                return [["sh", "-n"]], ""
            return [], "sh not available"

        return [], "no validation command available"

    def _detect_test_command(self) -> List[str]:
        if self.test_command:
            return shlex.split(self.test_command)

        test_patterns = ["**/test_*.py", "**/*_test.py"]
        has_tests = any(
            glob.glob(os.path.join(self.repo_path, pattern), recursive=True)
            for pattern in test_patterns
        )
        if not has_tests:
            return []

        if shutil.which("pytest"):
            return [sys.executable, "-m", "pytest", "-q"]

        return [sys.executable, "-m", "unittest", "discover", "-v"]

    def _detect_run_command(self) -> List[str]:
        if self.run_command:
            return shlex.split(self.run_command)
        return []

    def _apply_bundle(self, file_map: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
        backups: Dict[str, str] = {}
        created: List[str] = []
        for rel_path, code in file_map.items():
            full_path = os.path.normpath(os.path.join(self.repo_path, rel_path))
            if not full_path.startswith(os.path.normpath(self.repo_path) + os.sep):
                raise ValueError(f"Unsafe path outside repo: {rel_path}")
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    backups[full_path] = f.read()
            else:
                created.append(full_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(code)
        return backups, created

    def _restore_bundle(self, backups: Dict[str, str], created: List[str]) -> None:
        for path in created:
            if os.path.exists(path):
                os.remove(path)
        for path, content in backups.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    def validate_bundle(self, file_map: Dict[str, str], language: str = "python") -> Tuple[bool, list]:
        """Validate a bundle by compiling, running, and testing in repo context."""
        errors: List[str] = []
        warnings: List[str] = []

        if not file_map:
            return False, ["No files generated to validate"]

        backups: Dict[str, str] = {}
        created: List[str] = []
        try:
            backups, created = self._apply_bundle(file_map)

            languages = self._collect_languages(file_map)

            if "python" in languages:
                ran_any = False
                py_files = [p for p in file_map.keys() if self._language_for_path(p) == "python"]
                if py_files:
                    ran_any = True
                    for rel_path in py_files:
                        code = file_map[rel_path]
                        ok, msg = self.validate_syntax(code, file_label=rel_path)
                        if not ok:
                            errors.append(msg)
                            return False, errors

                    for rel_path in py_files:
                        code = file_map[rel_path]
                        ok, msg = self.validate_imports(code, file_label=rel_path)
                        if not ok:
                            errors.append(msg)

                    abs_py = [os.path.join(self.repo_path, p) for p in py_files]
                    compile_cmd = [sys.executable, "-m", "py_compile"] + abs_py
                    ok, out = self._run_command(compile_cmd, cwd=self.repo_path)
                    if not ok:
                        errors.append(f"Python compile failed: {out}")

                test_cmd = self._detect_test_command()
                if test_cmd:
                    ran_any = True
                    ok, out = self._run_command(test_cmd, cwd=self.repo_path)
                    if not ok:
                        errors.append(f"Python tests failed: {out}")

                if not ran_any:
                    warnings.append("python: unable to validate (no checks available). Please validate manually.")

            for lang in sorted(l for l in languages if l != "python"):
                commands, reason = self._get_language_commands(lang)
                if not commands:
                    warnings.append(f"{lang}: unable to validate ({reason}). Please validate manually.")
                    continue

                for cmd in commands:
                    if lang == "shell" and cmd[:2] == ["sh", "-n"]:
                        for rel_path in file_map.keys():
                            if self._language_for_path(rel_path) == "shell":
                                abs_path = os.path.join(self.repo_path, rel_path)
                                ok, out = self._run_command(cmd + [abs_path], cwd=self.repo_path)
                                if not ok:
                                    errors.append(f"Shell check failed ({rel_path}): {out}")
                    else:
                        ok, out = self._run_command(cmd, cwd=self.repo_path)
                        if not ok:
                            errors.append(f"{lang} validation failed: {out}")

            run_cmd = self._detect_run_command()
            if run_cmd:
                ok, out = self._run_command(run_cmd, cwd=self.repo_path)
                if not ok:
                    errors.append(f"Run failed: {out}")

            if errors:
                return False, errors
            return True, warnings
        finally:
            self._restore_bundle(backups, created)

    def validate(self, code: str, language: str = "python") -> Tuple[bool, list]:
        """Run basic syntax/import validations for single-file code."""
        if language and language.lower() != "python":
            return True, []
        issues = []

        ok, msg = self.validate_syntax(code)
        if not ok:
            issues.append(msg)
            return False, issues

        ok, msg = self.validate_imports(code)
        if not ok:
            issues.append(msg)

        return len(issues) == 0, issues
