"""Mutation check: break each safety property, confirm a test catches it.

Run from the repo root. Restores every file it touches.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()

MUTATIONS = [
    # (file, description, old substring, new substring)
    (
        "agent/safety.py",
        "blocklist: uppercase HOME (the bug the lowercase input hid)",
        r'|~|\$home|\*',
        r'|~|\$HOME|\*',
    ),
    (
        "agent/safety.py",
        "blocklist: drop the trailing [/*]* that catches `/*` and `~/`",
        r'[/*]*(?:\s|$)',
        r'(?:\s|$)',
    ),
    (
        "agent/safety.py",
        "blocklist: drop the long-form rm flag support",
        r'(?:\s+-{1,2}[a-z-]+)*',
        r'',
    ),
    (
        "agent/safety.py",
        "blocklist: allow sudo",
        r'r"\bsudo\b",' + "\n",
        '',
    ),
    (
        "agent/safety.py",
        "sandbox: allow any path",
        "if resolved != base and not resolved.is_relative_to(base):",
        "if False:",
    ),
    (
        "agent/safety.py",
        "confirmation: accept anything (yes-by-default)",
        'return answer in {"y", "yes"}',
        'return answer != ""',
    ),
    (
        "agent/safety.py",
        "confirmation: auto-approve when there is no terminal",
        "if not sys.stdin.isatty():",
        "if False:",
    ),
    (
        "agent/safety.py",
        "secret files: treat nothing as secret",
        "    name = path.name.lower()",
        "    return False\n    name = path.name.lower()",
    ),
    (
        "agent/tools.py",
        "grep: return nothing, hiding matches",
        "if needle in haystack:",
        "if False:",
    ),
    (
        "agent/tools.py",
        "edit_file: allow multiple matches",
        "if matches > 1:",
        "if False:",
    ),
    (
        "agent/tools.py",
        "run_command: ignore the timeout",
        "timeout=config.COMMAND_TIMEOUT_SECONDS,",
        "timeout=None,",
    ),
]


def run_tests() -> tuple[bool, str]:
    """Run the suite in a copy of the repo. Returns (passed, output)."""
    result = subprocess.run(
        ["./.venv/bin/python", "-m", "pytest", "-q", "--no-header", "-x"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout


def main() -> int:
    """Apply each mutation in turn and report which tests notice."""
    survived = []
    for filename, description, old, new in MUTATIONS:
        path = ROOT / filename
        original = path.read_text()
        if original.count(old) < 1:
            print(f"SKIP  (anchor not found) {description}")
            continue
        shutil.copy(path, path.with_suffix(".bak"))
        try:
            path.write_text(original.replace(old, new, 1))
            passed, output = run_tests()
            if passed:
                survived.append(description)
                print(f"SURVIVED  {description}")
            else:
                first = [
                    line for line in output.splitlines() if line.startswith("FAILED")
                ]
                print(f"caught    {description}  -> {first[:1]}")
        finally:
            shutil.move(str(path.with_suffix(".bak")), str(path))

    print()
    if survived:
        print(f"{len(survived)} mutation(s) survived - tests are too weak:")
        for item in survived:
            print(f"  - {item}")
        return 1
    print("Every mutation was caught.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
