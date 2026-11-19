import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = yaml.safe_load(open(ROOT / "configs" / "ablations.yaml"))

print(type(CONFIGS))
print(CONFIGS)
