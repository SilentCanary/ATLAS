#!/usr/bin/env python3
"""
Multi-language parser tests for ATLAS.
Verifies all language parsers produce correct output format
and work on real repo files (BlitzOS, VIGILUM).
"""

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Required keys that graph_builder.py accesses directly (not via .get())
REQUIRED_FUNC_KEYS = {"args", "calls", "start_line", "end_line", "code"}


def check_output_contract(result, label):
    """Verify a parsed result matches the ATLAS output contract."""
    issues = []

    if not isinstance(result, dict):
        return [f"{label}: result is not a dict"]

    for key in ("file_path", "language", "imports", "functions", "classes"):
        if key not in result:
            issues.append(f"{label}: missing top-level key '{key}'")

    if not issues:
        for fname, fdata in result.get("functions", {}).items():
            missing = REQUIRED_FUNC_KEYS - set(fdata.keys())
            if missing:
                issues.append(f"{label}: function '{fname}' missing keys: {missing}")

        for cname, cdata in result.get("classes", {}).items():
            if "inherits" not in cdata:
                issues.append(f"{label}: class '{cname}' missing 'inherits'")
            if "methods" not in cdata:
                issues.append(f"{label}: class '{cname}' missing 'methods'")
            for mname, mdata in cdata.get("methods", {}).items():
                missing = REQUIRED_FUNC_KEYS - set(mdata.keys())
                if missing:
                    issues.append(f"{label}: method '{cname}.{mname}' missing keys: {missing}")

    return issues


def test_c_parser():
    """Test C parser on BlitzOS keyboard driver."""
    print("\n" + "=" * 60)
    print("C PARSER TEST")
    print("=" * 60)

    from parser.language_parsers import CParser
    parser = CParser()

    test_file = "repos/BlitzOS/drivers/keyboard.c"
    if not os.path.exists(test_file):
        print(f"[SKIP] {test_file} not found")
        return True

    result = parser.parse_file(test_file)
    if result is None:
        print("[FAIL] CParser returned None")
        return False

    issues = check_output_contract(result, "keyboard.c")
    if issues:
        for i in issues:
            print(f"[FAIL] {i}")
        return False

    print(f"  Language: {result['language']}")
    print(f"  Imports: {result['imports'][:5]}")
    print(f"  Functions: {list(result['functions'].keys())[:5]}")
    print(f"  Classes/Structs: {list(result['classes'].keys())[:5]}")

    if result["imports"]:
        print("[PASS] C parser found imports (#include)")
    else:
        print("[WARN] No imports found (may be expected)")

    if result["functions"]:
        print(f"[PASS] C parser found {len(result['functions'])} functions")
    else:
        print("[WARN] No functions found in keyboard.c")

    print("[PASS] C parser output contract valid")
    return True


def test_assembly_parser():
    """Test Assembly parser on BlitzOS boot.asm."""
    print("\n" + "=" * 60)
    print("ASSEMBLY PARSER TEST")
    print("=" * 60)

    from parser.language_parsers import AssemblyParser
    parser = AssemblyParser()

    test_file = "repos/BlitzOS/kernel/arch/x86_64/boot/boot.asm"
    if not os.path.exists(test_file):
        print(f"[SKIP] {test_file} not found")
        return True

    result = parser.parse_file(test_file)
    if result is None:
        print("[FAIL] AssemblyParser returned None")
        return False

    issues = check_output_contract(result, "boot.asm")
    if issues:
        for i in issues:
            print(f"[FAIL] {i}")
        return False

    print(f"  Language: {result['language']}")
    print(f"  Imports (extern): {result['imports'][:5]}")
    print(f"  Labels as functions: {list(result['functions'].keys())[:8]}")

    if result["functions"]:
        print(f"[PASS] Assembly parser found {len(result['functions'])} labels/functions")
    else:
        print("[FAIL] No labels found in boot.asm")
        return False

    # Check for extern imports
    if result["imports"]:
        print(f"[PASS] Assembly parser found {len(result['imports'])} extern/include imports")
    else:
        print("[WARN] No extern imports found")

    # Check for call detection
    all_calls = []
    for fdata in result["functions"].values():
        all_calls.extend(fdata["calls"])
    if all_calls:
        print(f"[PASS] Assembly parser found {len(all_calls)} call instructions")
    else:
        print("[WARN] No call instructions found")

    print("[PASS] Assembly parser output contract valid")
    return True


def test_go_parser():
    """Test Go parser on VIGILUM backend."""
    print("\n" + "=" * 60)
    print("GO PARSER TEST")
    print("=" * 60)

    from parser.language_parsers import GoParser
    parser = GoParser()

    test_file = "repos/VIGILUM/backend/cmd/api/main.go"
    if not os.path.exists(test_file):
        print(f"[SKIP] {test_file} not found")
        return True

    result = parser.parse_file(test_file)
    if result is None:
        print("[FAIL] GoParser returned None")
        return False

    issues = check_output_contract(result, "main.go")
    if issues:
        for i in issues:
            print(f"[FAIL] {i}")
        return False

    print(f"  Language: {result['language']}")
    print(f"  Imports: {result['imports'][:5]}")
    print(f"  Functions: {list(result['functions'].keys())[:5]}")
    print(f"  Structs/Types: {list(result['classes'].keys())[:5]}")

    if "main" in result["functions"]:
        print("[PASS] Go parser found main() function")
    else:
        print("[WARN] main() not found (may be in different file)")

    if result["imports"]:
        print(f"[PASS] Go parser found {len(result['imports'])} imports")
    else:
        print("[WARN] No imports found")

    print("[PASS] Go parser output contract valid")
    return True


def test_solidity_parser():
    """Test Solidity parser on VIGILUM contracts."""
    print("\n" + "=" * 60)
    print("SOLIDITY PARSER TEST")
    print("=" * 60)

    from parser.language_parsers import SolidityParser
    parser = SolidityParser()

    # Find a good Solidity file
    test_file = None
    sol_dir = "repos/VIGILUM/contracts"
    if os.path.exists(sol_dir):
        for root, _, files in os.walk(sol_dir):
            for f in files:
                if f.endswith(".sol") and "test" not in f.lower():
                    candidate = os.path.join(root, f)
                    if os.path.getsize(candidate) > 200:
                        test_file = candidate
                        break
            if test_file:
                break

    if not test_file:
        print("[SKIP] No suitable .sol file found")
        return True

    result = parser.parse_file(test_file)
    if result is None:
        print(f"[FAIL] SolidityParser returned None for {test_file}")
        return False

    issues = check_output_contract(result, os.path.basename(test_file))
    if issues:
        for i in issues:
            print(f"[FAIL] {i}")
        return False

    print(f"  File: {os.path.basename(test_file)}")
    print(f"  Language: {result['language']}")
    print(f"  Imports: {result['imports'][:3]}")
    print(f"  Contracts: {list(result['classes'].keys())[:5]}")

    if result["classes"]:
        print(f"[PASS] Solidity parser found {len(result['classes'])} contracts")
    else:
        print("[WARN] No contracts found")

    print("[PASS] Solidity parser output contract valid")
    return True


def test_typescript_parser():
    """Test TypeScript parser on VIGILUM SDK."""
    print("\n" + "=" * 60)
    print("TYPESCRIPT PARSER TEST")
    print("=" * 60)

    from parser.language_parsers import TypeScriptParser
    parser = TypeScriptParser()

    test_file = "repos/VIGILUM/sdk/ts-sdk/src/analyzer.ts"
    if not os.path.exists(test_file):
        print(f"[SKIP] {test_file} not found")
        return True

    result = parser.parse_file(test_file)
    if result is None:
        print("[FAIL] TypeScriptParser returned None")
        return False

    issues = check_output_contract(result, "analyzer.ts")
    if issues:
        for i in issues:
            print(f"[FAIL] {i}")
        return False

    print(f"  Language: {result['language']}")
    print(f"  Imports: {result['imports'][:5]}")
    print(f"  Functions: {list(result['functions'].keys())[:5]}")
    print(f"  Classes: {list(result['classes'].keys())[:5]}")

    total_entities = len(result["functions"]) + len(result["classes"])
    if total_entities > 0:
        print(f"[PASS] TypeScript parser found {total_entities} entities")
    else:
        print("[WARN] No functions/classes found")

    print("[PASS] TypeScript parser output contract valid")
    return True


def test_rust_parser():
    """Test Rust parser on VIGILUM WASM SDK."""
    print("\n" + "=" * 60)
    print("RUST PARSER TEST")
    print("=" * 60)

    from parser.language_parsers import RustParser
    parser = RustParser()

    test_file = "repos/VIGILUM/sdk/src/wasm/src/lib.rs"
    if not os.path.exists(test_file):
        print(f"[SKIP] {test_file} not found")
        return True

    result = parser.parse_file(test_file)
    if result is None:
        print("[FAIL] RustParser returned None")
        return False

    issues = check_output_contract(result, "lib.rs")
    if issues:
        for i in issues:
            print(f"[FAIL] {i}")
        return False

    print(f"  Language: {result['language']}")
    print(f"  Imports (use): {result['imports'][:5]}")
    print(f"  Functions: {list(result['functions'].keys())[:5]}")
    print(f"  Structs: {list(result['classes'].keys())[:5]}")

    total_entities = len(result["functions"]) + len(result["classes"])
    if total_entities > 0:
        print(f"[PASS] Rust parser found {total_entities} entities")
    else:
        print("[WARN] No functions/structs found")

    print("[PASS] Rust parser output contract valid")
    return True


def test_fallback_parser():
    """Test fallback parser on unknown file type."""
    print("\n" + "=" * 60)
    print("FALLBACK PARSER TEST")
    print("=" * 60)

    from parser.language_parsers import FallbackParser
    parser = FallbackParser()

    # Use any text file as test
    test_file = "README.md"
    if not os.path.exists(test_file):
        test_file = "test_multilang_parser.py"

    result = parser.parse_file(test_file)
    if result is None:
        print("[FAIL] FallbackParser returned None")
        return False

    issues = check_output_contract(result, "fallback")
    if issues:
        for i in issues:
            print(f"[FAIL] {i}")
        return False

    if "__file__" in result["functions"]:
        print("[PASS] Fallback parser stores whole file as __file__ function")
    else:
        print("[FAIL] Fallback parser missing __file__ function")
        return False

    print("[PASS] Fallback parser output contract valid")
    return True


def test_repo_parser_blitzos():
    """Test full repo parser on BlitzOS (C + ASM)."""
    print("\n" + "=" * 60)
    print("REPO PARSER: BlitzOS (C/ASM)")
    print("=" * 60)

    from parser.repo_parser import CodeParser
    parser = CodeParser()

    repo_path = "repos/BlitzOS"
    if not os.path.exists(repo_path):
        print(f"[SKIP] {repo_path} not found")
        return True

    parsed = parser.parse_repository(repo_path)
    print(f"  Total files parsed: {len(parsed)}")

    # Count by language
    lang_counts = {}
    for data in parsed.values():
        lang = data.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"  By language: {lang_counts}")

    # Verify all outputs satisfy contract
    all_ok = True
    for fpath, data in parsed.items():
        issues = check_output_contract(data, os.path.basename(fpath))
        if issues:
            for i in issues:
                print(f"[FAIL] {i}")
            all_ok = False

    if len(parsed) > 5:
        print(f"[PASS] BlitzOS: parsed {len(parsed)} files across {len(lang_counts)} languages")
    else:
        print(f"[WARN] Only parsed {len(parsed)} files")

    if all_ok:
        print("[PASS] All BlitzOS outputs satisfy contract")
    else:
        print("[FAIL] Some BlitzOS outputs have contract violations")

    return all_ok


def test_repo_parser_vigilum():
    """Test full repo parser on VIGILUM (Go/Solidity/TS/Rust)."""
    print("\n" + "=" * 60)
    print("REPO PARSER: VIGILUM (Go/Sol/TS/Rust)")
    print("=" * 60)

    from parser.repo_parser import CodeParser
    parser = CodeParser()

    repo_path = "repos/VIGILUM"
    if not os.path.exists(repo_path):
        print(f"[SKIP] {repo_path} not found")
        return True

    parsed = parser.parse_repository(repo_path)
    print(f"  Total files parsed: {len(parsed)}")

    lang_counts = {}
    for data in parsed.values():
        lang = data.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"  By language: {lang_counts}")

    all_ok = True
    for fpath, data in parsed.items():
        issues = check_output_contract(data, os.path.basename(fpath))
        if issues:
            for i in issues:
                print(f"[FAIL] {i}")
            all_ok = False

    if len(parsed) > 20:
        print(f"[PASS] VIGILUM: parsed {len(parsed)} files across {len(lang_counts)} languages")
    else:
        print(f"[WARN] Only parsed {len(parsed)} files")

    if all_ok:
        print("[PASS] All VIGILUM outputs satisfy contract")
    else:
        print("[FAIL] Some VIGILUM outputs have contract violations")

    return all_ok


def test_graph_builder_multilang():
    """Test that graph_builder works on multi-language parsed output."""
    print("\n" + "=" * 60)
    print("GRAPH BUILDER + MULTI-LANG")
    print("=" * 60)

    from parser.repo_parser import CodeParser
    from graph.graph_builder import CodeGraphBuilder

    repo_path = "repos/BlitzOS"
    if not os.path.exists(repo_path):
        print(f"[SKIP] {repo_path} not found")
        return True

    parser = CodeParser()
    parsed = parser.parse_repository(repo_path)

    if not parsed:
        print("[SKIP] No parsed data")
        return True

    try:
        builder = CodeGraphBuilder()
        graph = builder.build_graph(parsed)
        print(f"  Graph nodes: {graph.number_of_nodes()}")
        print(f"  Graph edges: {graph.number_of_edges()}")

        if graph.number_of_nodes() > 0:
            print("[PASS] Graph builder works with multi-language input (no KeyError)")
        else:
            print("[WARN] Graph built but has no nodes")

        return True
    except KeyError as e:
        print(f"[FAIL] KeyError in graph_builder: {e}")
        print("  This means a parser is missing required keys in its output")
        return False
    except Exception as e:
        print(f"[FAIL] Graph builder error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("MULTI-LANGUAGE PARSER TESTS")
    print("=" * 60)

    results = []

    # Individual parser tests
    results.append(("C Parser", test_c_parser()))
    results.append(("Assembly Parser", test_assembly_parser()))
    results.append(("Go Parser", test_go_parser()))
    results.append(("Solidity Parser", test_solidity_parser()))
    results.append(("TypeScript Parser", test_typescript_parser()))
    results.append(("Rust Parser", test_rust_parser()))
    results.append(("Fallback Parser", test_fallback_parser()))

    # Full repo tests
    results.append(("BlitzOS Repo", test_repo_parser_blitzos()))
    results.append(("VIGILUM Repo", test_repo_parser_vigilum()))

    # Graph integration
    results.append(("Graph Builder", test_graph_builder_multilang()))

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All multi-language parser tests passed!")
        return 0
    else:
        print("\n[ERROR] Some tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
