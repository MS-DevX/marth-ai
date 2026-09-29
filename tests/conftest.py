"""Fixtures shared across the test suite.

`project` lives here rather than in one test module because the sandbox,
tool, and write-tool tests all need a workspace root that is not the real
project directory. Pointing the agent at the actual repository in a test
is how you end up debugging a deleted file.
"""

from pathlib import Path

import pytest

from agent import config


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a temp directory that looks like a small project.

    The sandbox root is redirected to it so the tools operate on this
    fixture instead of the real workspace. Tools deliberately take only a
    `path` argument (the model supplies it), so the root has to come from
    config rather than from a parameter.

    It also contains the noise a real repository has — a `.git` directory
    and a `node_modules` — so the skipping rules are exercised by every
    test that lists or searches, not only by the ones that mean to.
    """
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "main.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# Title\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret=1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "left-pad.js").write_text("//\n")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    return tmp_path
