import os
import sys

sys.path.append(".")

from script_manager.func.script_parse_args import get_script_args

args = get_script_args()

from script_manager.func.script_boilerplate import do_everything

# script to be used
main_script = os.path.join("pylit_main.py")

# weights and biases project name
wandb_project_name = "clebert"

# keys
appendix_keys = ["tag"]
extra_folder_keys = []

# default parameteres
test_parameters = {"num_workers": 0}
default_parameters = {
    "zeroout_prob": 0.15,
    "batch_size": 64,
    "device": "cuda:0",
    "epochs": 100,
    "num_workers": 2,
}

# configs to be exectuted
configs = []

configs.append([{"tag": "TOKEN_CLASSIFIER", "model": "TOKEN_CLASSIFIER"}, None])


configs.append([{"tag": "FILLMASK", "model": "FILLMASK"}, None])

configs.append(
    [{"tag": "FILLMASK_baseline", "model": "FILLMASK", "zeroout_prob": 0.6}, None]
)
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
