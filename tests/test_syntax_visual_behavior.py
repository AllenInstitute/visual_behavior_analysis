import os
import ast
import pytest


VISUAL_BEHAVIOR_DIR = os.path.join(
    os.path.realpath(os.path.dirname(os.path.dirname(__file__))),
    "visual_behavior"
)


def _check_syntax(path):
    if os.path.isdir(path):
        for fname in os.listdir(path):
            _check_syntax(os.path.join(path, fname))
    elif os.path.isfile(path):
        with open(path, "rb") as sstream:
            ast.parse(sstream.read(), filename=path)
    else:
        raise ValueError("unexpected fallthrough: {}".format(path))


@pytest.mark.parametrize("path", [
    os.path.join(VISUAL_BEHAVIOR_DIR, "cohorts"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "inscopix"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "metrics"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "__init__.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "analyze.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "core.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "data.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "devices.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "io.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "masks.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "nwb.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "plotting.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "summarize.py"),
    os.path.join(VISUAL_BEHAVIOR_DIR, "utilities.py"),
])
def test_syntax(path):
    _check_syntax(path)
