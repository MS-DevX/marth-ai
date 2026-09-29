"""Report definitions that appear more than once in a module.

Duplicated definitions are silent in Python: the last one wins, so the
earlier copy is dead code that still reads as though it were live. That
is exactly what happened to safety.py, where the tests were passing
against a copy nobody was looking at.

Run from the repo root:  python3 check_duplicates.py
"""

import ast
import sys
from pathlib import Path


def top_level_names(tree: ast.Module) -> dict[str, int]:
    """Count how often each top-level name is defined.

    Imports are counted by the name they actually bind. `import
    urllib.error` and `import urllib.request` both bind `urllib`, but
    that is ordinary Python rather than a duplicated definition, so
    submodule imports are skipped entirely.
    """
    counts: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            counts[node.name] = counts.get(node.name, 0) + 1
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    counts[target.id] = counts.get(target.id, 0) + 1
    return counts


def main() -> int:
    """Check every module under agent/ and report duplicates."""
    problems = 0
    for path in sorted(Path("agent").glob("*.py")):
        tree = ast.parse(path.read_text())
        docstring_nodes = sum(
            1
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        dupes = {n: c for n, c in top_level_names(tree).items() if c > 1}
        extras = ""
        if docstring_nodes > 1:
            extras = f"  ({docstring_nodes} module docstrings!)"
        if dupes or extras:
            problems += 1
            print(f"{path}:{extras}")
            for name, count in sorted(dupes.items()):
                print(f"    {name} defined {count} times")
    if problems == 0:
        print("No duplicate definitions in agent/.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
