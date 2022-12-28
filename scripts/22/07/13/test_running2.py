import os
import sys

sys.path.append(".")

from script_manager.func.script_boilerplate import do_everything
from script_manager.func.script_parse_args import get_script_args

args = get_script_args()

# script to be used
main_script = os.path.join("pylit_main.py")

# weights and biases project name
wandb_project_name = "test_some_functionality"

# keys
appendix_keys = ["tag"]
extra_folder_keys = []

# default parameteres
test_parameters = {}
default_parameters = {}

# configs to be exectuted
configs = []
config1 = {"tag": "num_1_", "model": "TokenClassifier"}

configs.append([config1, None])

# configs.append([config2,
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
