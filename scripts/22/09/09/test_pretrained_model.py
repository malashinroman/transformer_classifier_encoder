import os
import sys

sys.path.append(".")

from local_config import TRANSFORMER_CLASSIFIER_ENCODER_PATH, WEAK_CLASSIFIERS
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
    "epochs": 5,
    "lr_scheduler": "MultiStepLR",
    "lr_milestones": "[2,4]",
    "num_workers": 0,
    "train_set_size": 1000,
    "test_set_size": 1000,
}

default_parameters = {
    "batch_size": 64,
    "device": "cuda:0",
    "epochs": 200,
    "fixed_zero_exp_num": 1,
    "num_workers": 8,
    "lr_scheduler": "MultiStepLR",
    "lr_milestones": "[100,150]",
}

load_checkpoint_path = os.path.join(
    TRANSFORMER_CLASSIFIER_ENCODER_PATH,
    "notable_checkpoints/imagenet_pretrained_zeroprob_200_ep/best_net.pkl",
)
configs = []
for fixed_zero_exp_num in range(1, 10):
    configs.append(
        [
            {
                "classifiers_indexes": "[0,1,2,3,4,5,6,7,8,9]",
                "dataset": "imagenet",
                "fixed_zero_exp_num": fixed_zero_exp_num,
                "loss": "CE_LOSS",
                "lr": 5e-5,
                "model": "FILLMASK_RAND",
                "optimizer": "AdamW",
                "random_seed": 0,
                "tag": f"test_pretrained_zero_experts_{fixed_zero_exp_num}",
                "use_static_files": 0,
                "zeroout_prob": 0.0,
                "use_pretrained_bert": 0,
                "skip_training": 1,
                "epochs": 1,
                "load_checkpoint": load_checkpoint_path,
                "weak_classifier_folder": os.path.join(
                    WEAK_CLASSIFIERS,
                    "cifar100_single_resent/2020-12-02T15-21-48_700332_weight_decay_0_0001_linear_search_False/tb",
                ),
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
