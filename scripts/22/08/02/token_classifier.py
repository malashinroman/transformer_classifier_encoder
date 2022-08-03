import os
import sys

sys.path.append(".")

from script_manager.func.script_boilerplate import do_everything
from script_manager.func.script_parse_args import get_script_args

args = get_script_args()

# script to be used
main_script = os.path.join("main.py")

# weights and biases project name
wandb_project_name = "SIMPLE_FC"

# keys
appendix_keys = ["tag"]
extra_folder_keys = []

# default parameteres
test_parameters = {
    "num_workers": 0,
    "epochs": 1,
}

default_parameters = {
    # "zeroout_prob": 0.15,
    "fixed_zero_exp_num": 1,
    "batch_size": 64,
    "device": "cuda:0",
    "epochs": 200,
    "num_workers": 8,
}

# configs to be exectuted
configs = []
# for fixed_zero_exp_num in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
for fixed_zero_exp_num in [1]:
    configs.append(
        [
            {
                "fixed_zero_exp_num": fixed_zero_exp_num,
                "tag": f"fixed_zero_exp_num_{fixed_zero_exp_num}",
                "model": "FILLMASK",
                "loss": "AE_MSE_LOSS",
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
