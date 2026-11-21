import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
T1DSIM_AI_SRC = REPO_ROOT / "T1DSim_AI" / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(T1DSIM_AI_SRC) not in sys.path:
    sys.path.insert(0, str(T1DSIM_AI_SRC))
