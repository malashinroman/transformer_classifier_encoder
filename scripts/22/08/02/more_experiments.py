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
for fixed_zero_exp_num in [2, 3, 4, 5, 6, 7, 8, 9]:
    configs.append(
        [
            {
                "fixed_zero_exp_num": fixed_zero_exp_num,
                "tag": f"fixed_zero_exp_num_{fixed_zero_exp_num}",
                "model": "SIMPLE_FC",
                "loss": "AE_MSE_LOSS",
            },
            None,
        ]
    )

# for lr in [1e-3, 1e-4, 1e-5]:
#     for batch_size in [128, 256]:
#         for model in ["SIMPLE_FC", "FILLMASK"]:
#             configs.append(
#                 [
#                     {
#                         "tag": "lr_{}_batch_{}_no_pl_{}".format(lr, batch_size, model),
#                         "lr": lr,
#                         "batch_size": batch_size,
#                         "model": "SIMPLE_FC",
#                         "loss": "AE_MSE_LOSS",
#                     },
#                     None,
#                 ]
#             )
#             # configs.append([{"tag": "SIMPLE_FC", "model": "SIMPLE_FC", "lr": 1e3}, None])

print(f"total_number_of_scripts = {len(configs)}")
# RUN everything
# !normally you don't have to change anything here
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
