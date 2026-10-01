"""Tests for the install manifest.

A pyproject.toml is the one file in this project that nothing imports and
every user depends on. If it is wrong the failure is not a traceback, it
is a pip error or an agent that installs and then sandboxes itself
inside a venv. So it gets tested like code: the entry point is called,
not just read, and the two places that list dependencies are checked
against each other so they cannot drift apart quietly.
"""

import tomllib
from pathlib import Path

from agent import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "pyproject.toml"


def load_manifest() -> dict:
    """Return the parsed pyproject.toml.

    tomllib is standard library from 3.11, which is this project's floor,
    so the test suite needs no extra dependency to check its own packaging.
    """
    return tomllib.loads(MANIFEST.read_text())


def requirement_lines() -> list[str]:
    """Return the requirements in requirements.txt, blank lines dropped."""
    text = (REPO_ROOT / "requirements.txt").read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


# --- the two dependency lists must agree -----------------------------------


def test_requirements_txt_matches_the_manifest() -> None:
    """requirements.txt is what the README tells a developer to run;
    pyproject is what pip installs. If they disagree, one of those two
    instructions is a lie and the difference is only found at runtime."""
    project = load_manifest()["project"]
    declared = sorted(project["dependencies"] + project["optional-dependencies"]["dev"])
    assert sorted(requirement_lines()) == declared


def test_pytest_is_not_a_runtime_dependency() -> None:
    """Nobody needs a test runner to run the agent, and a 5MB transitive
    tree nobody asked for is a good reason not to install something."""
    project = load_manifest()["project"]
    assert not any(dep.startswith("pytest") for dep in project["dependencies"])
    assert any(dep.startswith("pytest") for dep in project["optional-dependencies"]["dev"])


# --- the entry point -------------------------------------------------------


def test_the_console_script_target_exists_and_is_callable() -> None:
    """`marth = "agent.main:main"` is a string, so nothing checks it
    resolves until a user installs the package and the shim fails. Called
    here instead, with no arguments, so it parses an empty argv and
    reports there being no task."""
    scripts = load_manifest()["project"]["scripts"]
    assert scripts == {"marth": "agent.main:main"}

    module_name, _, attribute = scripts["marth"].partition(":")
    module = __import__(module_name, fromlist=[attribute])
    entry = getattr(module, attribute)
    assert callable(entry)


def test_the_entry_point_returns_an_exit_code(project: Path) -> None:
    """A console script hands the return value to sys.exit, so anything
    other than an int makes `marth` exit 0 no matter what happened.

    --history is used because it is the one path that returns a code
    without a model call, a terminal, or a network. Asking for no task at
    all would sit waiting on input() instead, which is the interactive
    case and not what is being checked here.
    """
    from agent.main import main

    assert main(["--history"]) == 0


# --- what gets shipped -----------------------------------------------------


def test_only_the_agent_package_is_installed() -> None:
    """Auto-discovery would sweep up tests/ and the dev scripts at the
    repo root and install them as top-level importable modules, putting a
    `tests` package on every user's import path."""
    packages = load_manifest()["tool"]["setuptools"]["packages"]
    assert packages == ["agent"]


def test_the_version_is_read_from_the_package() -> None:
    """A hardcoded version in two places drifts the first time a release
    is cut, and the drift is invisible until someone installs a stale
    wheel."""
    project = load_manifest()["project"]
    assert "version" in project["dynamic"]
    assert project["dynamic"] == ["version"]
    dynamic = load_manifest()["tool"]["setuptools"]["dynamic"]
    assert dynamic["version"] == {"attr": "agent.__version__"}


def test_the_version_looks_like_a_version() -> None:
    assert __version__.count(".") >= 1
    assert all(part.isdigit() for part in __version__.split(".")[:2])


# --- the floor -------------------------------------------------------------


def test_requires_python_matches_the_projects_own_floor() -> None:
    """The code uses tomllib and PEP 604 unions, both of which are 3.11.
    Claiming a lower floor would produce an install that cannot run."""
    assert load_manifest()["project"]["requires-python"] == ">=3.11"


# --- the install instructions have to be true -------------------------------


def test_the_readme_install_command_names_this_repository() -> None:
    """The one line most people will ever type. If the org, the repo or
    the URL is wrong, everyone who follows the README gets a 404, and
    nothing in this suite would otherwise notice."""
    readme = (REPO_ROOT / "README.md").read_text()
    assert "pipx install git+https://github.com/MS-DevX/marth-ai.git" in readme


def test_the_readme_documents_the_setup_step() -> None:
    """Installing the package is half of it. Without the model there is
    nothing to run, and the failure looks like a broken agent."""
    readme = (REPO_ROOT / "README.md").read_text()
    assert "marth --setup" in readme


def usage_block() -> str:
    """Return the README's Usage section, up to the next `## ` heading.

    Bounded by the next heading rather than by a named one, so renaming or
    reordering the sections after Usage cannot make these tests read the
    wrong part of the file.

    Returns:
        The Usage section as markdown.

    Raises:
        ValueError: if the README has no Usage section at all, which would
            itself be a documentation failure worth failing loudly for.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    start = readme.index("## Usage")
    end = readme.find("\n## ", start + 1)
    if end == -1:
        end = len(readme)
    return readme[start:end]


def test_every_documented_marth_flag_exists() -> None:
    """A renamed flag would leave the README telling people to type
    something that errors out.

    Read out of the Usage block rather than the whole file, so prose
    that merely mentions a flag in passing does not pin it for ever.
    """
    import re

    from agent.main import build_parser

    usage = usage_block()

    documented = set(re.findall(r"\bmarth (--[a-z][a-z-]*)", usage))
    assert documented, "no marth flags found in the Usage section"

    known = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    missing = documented - known
    assert not missing, f"README documents flags that do not exist: {missing}"


def test_the_usage_block_covers_every_action_flag() -> None:
    """The other direction: a flag nothing in the README mentions is one
    nobody will know about."""
    import re

    from agent.main import build_parser

    usage = usage_block()

    documented = set(re.findall(r"\bmarth (--[a-z][a-z-]*)", usage))
    undocumented = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
        if option not in documented
    }
    # --help is argparse's own and is never written as `marth --help`.
    assert undocumented <= {"--help", "-h"}, undocumented


def test_the_manifest_ships_a_readme() -> None:
    """pip renders it on the install page, and PyPI rejects the upload
    without it."""
    project = load_manifest()["project"]
    assert project["readme"] == "README.md"
    assert (REPO_ROOT / project["readme"]).is_file()
