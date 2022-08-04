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
    "fixed_zero_exp_num": 1,
    "batch_size": 64,
    "device": "cuda:0",
    "num_workers": 8,
}

configs = []

for optimizer in ["AdamW"]:
    for fixed_zero_exp_num in [1]:
        for lr in [5e-5]:
            configs.append(
                [
                    {
                        "epochs": 0,
                        "fixed_zero_exp_num": fixed_zero_exp_num,
                        "tag": f"imagenet_test",
                        "model": "FILLMASK",
                        "weak_classifier_folder": os.path.join(
                            WEAK_CLASSIFIERS,
                            "cifar100_single_resent/2020-12-02T15-21-48_700332_weight_decay_0_0001_linear_search_False/tb",
                        ),
                        "classifiers_indexes": "[0,1,2,3,4,5,6,7,8,9]",
                        "use_static_files": 0,
                        "dataset": "imagenet",
                        "load_checkpoint": "notable_checkpoints/imagenet_pretrained/best_net.pkl",
                        "skip_training": 1,
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
