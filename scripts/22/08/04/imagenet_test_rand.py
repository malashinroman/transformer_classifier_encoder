import os
import sys

sys.path.append(".")

from local_config import WEAK_CLASSIFIERS
from script_manager.func.script_boilerplate import do_everything
from script_manager.func.script_parse_args import get_script_args

args = get_script_args()

# script to be used
main_script = os.path.join("main.py")

# weights and biases project name
wandb_project_name = "clebert"

# keys
appendix_keys = ["tag"]
extra_folder_keys = []

# default parameteres
test_parameters = {
    "num_workers": 0,
    "epochs": 0,
}

default_parameters = {
    "batch_size": 64,
    "classifiers_indexes": "[0,1,2,3,4,5,6,7,8,9]",
    "dataset": "imagenet",
    "device": "cuda:0",
    "epochs": 0,
    "fixed_zero_exp_num": 1,
    "model": "FILLMASK_RAND",
    "num_workers": 8,
    "skip_training": 1,
    "use_static_files": 0,
    "random_seed": 0,
    "weak_classifier_folder": os.path.join(
        WEAK_CLASSIFIERS,
        "cifar100_single_resent/2020-12-02T15-21-48_700332_weight_decay_0_0001_linear_search_False/tb",
    ),
}

configs = []
checkpoints = [
    # "notable_checkpoints/imagenet_pretrained/best_net_ep1.pkl",
    "notable_checkpoints/imagenet_pretrained/best_net_ep5.pkl",
]

for i, checkpoint in enumerate(checkpoints):
    configs.append(
        [
            {
                "tag": f"imagenet_test_{i}",
                "load_checkpoint": checkpoint,
            },
            None,
        ]
    )

print(f"total_number_of_scripts = {len(configs)}")
if __name__ == "__main__":
    do_everything(
        default_parameters=default_parameters,
        configs=configs,
        extra_folder_keys=extra_folder_keys,
        appendix_keys=appendix_keys,
        main_script=main_script,
        test_parameters=test_parameters,
        wandb_project_name=wandb_project_name,
        script_file=__file__,
    )
