"""
Daily project release script.

Picks the next unpublished project from `library/`, copies it into
`published/`, and records it in `published/PUBLISHED_LOG.md` and
`published/.state.json`.

Designed to be run once per day by a GitHub Actions workflow, but it also
works locally. It is idempotent-friendly: if every project has already been
published it exits cleanly without making changes.

Exit codes:
  0  a project was published, OR nothing left to publish (no error)
"""

import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "library"
PUBLISHED = ROOT / "published"
STATE_FILE = PUBLISHED / ".state.json"
LOG_FILE = PUBLISHED / "PUBLISHED_LOG.md"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"published": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def available_projects():
    """All project folders in the library, in stable alphabetical order."""
    if not LIBRARY.exists():
        return []
    return sorted(p.name for p in LIBRARY.iterdir() if p.is_dir())


def read_description(project_dir: Path) -> str:
    """Grab the first non-empty, non-heading line of the project README."""
    readme = project_dir / "README.md"
    if not readme.exists():
        return ""
    for line in readme.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def append_log(project_name: str, description: str):
    today = date.today().isoformat()
    row = f"| {today} | `{project_name}` | {description} |\n"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(row)


def main():
    PUBLISHED.mkdir(exist_ok=True)
    state = load_state()
    published = set(state.get("published", []))

    projects = available_projects()
    if not projects:
        print("No projects found in library/. Nothing to do.")
        return

    remaining = [p for p in projects if p not in published]
    if not remaining:
        print(f"All {len(projects)} projects already published. Nothing to do.")
        return

    next_project = remaining[0]
    src = LIBRARY / next_project
    dst = PUBLISHED / next_project

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    description = read_description(src)
    append_log(next_project, description)

    state.setdefault("published", []).append(next_project)
    save_state(state)

    print(f"Published: {next_project}")
    print(f"Remaining after this run: {len(remaining) - 1}")


if __name__ == "__main__":
    main()
