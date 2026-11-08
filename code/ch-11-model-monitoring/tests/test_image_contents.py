"""Check the image carries every module the code inside it imports.

The Dockerfile copies a named list of files rather than the whole of src, so a
new module does not join the image by being written. Nothing local notices,
because PYTHONPATH=src finds it on a laptop either way; the Processing job is
where it goes missing, and a lazy import inside a branch hides it until that
branch runs in the cloud.

That is how src/bias.py and src/capture.py both came to be absent from an image
that ran fine locally.
"""

import re
from pathlib import Path

CHAPTER = Path(__file__).resolve().parent.parent
SRC = CHAPTER / "src"


def modules_in_image() -> set[str]:
    """Read the module names the Dockerfile copies out of src."""
    return set(re.findall(r"src/([a-z_]+)\.py", (CHAPTER / "Dockerfile").read_text()))


def imports_between_modules() -> dict[str, set[str]]:
    """Map each src module to the other src modules it imports."""
    local = {p.stem for p in SRC.glob("*.py")}
    out = {}
    for path in SRC.glob("*.py"):
        found = re.findall(
            r"^\s*(?:import|from)\s+([a-z_][a-z0-9_]*)", path.read_text(), re.M
        )
        out[path.stem] = {name for name in found if name in local}
    return out


def test_image_carries_what_its_modules_import():
    """Every module imported by a module in the image is in the image too."""
    in_image = modules_in_image()
    graph = imports_between_modules()

    missing = {
        module: sorted(needs - in_image)
        for module, needs in graph.items()
        if module in in_image and needs - in_image
    }
    assert not missing, (
        "the Dockerfile copies these modules but not what they import: "
        + "; ".join(f"{m} needs {', '.join(n)}" for m, n in sorted(missing.items()))
    )


def test_every_module_the_dockerfile_names_exists():
    """The Dockerfile does not name a module that has since been renamed away."""
    gone = sorted(m for m in modules_in_image() if not (SRC / f"{m}.py").exists())
    assert not gone, f"the Dockerfile copies files that no longer exist: {gone}"
