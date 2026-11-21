import yaml
from pathlib import Path
import alexnet_cifar10.runner as runner

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = yaml.safe_load(open(ROOT / "configs" / "ablations.yaml"))
RESULTS_DIR = ROOT/"results"

for ablation, config in CONFIGS.items():
    for seed in [1, 2, 3]:
        runner.run(ablation, seed, RESULTS_DIR, aug=config["aug"],
                   no_reg=config["no_reg"], act=config["act"],   dropout=config["dropout"])
