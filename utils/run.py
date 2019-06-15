#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

"""
Manh: This function generates commands for different split run. The command will be 
store in *.sh file in /gen_scripts folder. Then it will executes each command sequentially. 
"""

from __future__ import print_function
from __future__ import division
from six.moves import range

import os
import sys
import datetime
import json
import subprocess
"""
Me-Example run: 
# Train proposed model and ablation models
    python utils/run.py scripts/5fold.json run <gpu id>
# Train proposed model only
    python utils/run.py scripts/5fold_proposed_only.json run <gpu id>
"""

def add_str(x, y, end=False):
    if end:
        return x + y
    else:
        return x + y + " "


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python run.py <json_file> run/test <gpu_id>")
        exit(1)

    input_name = sys.argv[1]                # Example: scripts/5fold.json
    decision = sys.argv[2]                  # Example: run 
    gpu_id = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    # Read configuration from json file.
    with open(input_name, "r") as f:
        data = json.load(f)

    # Create gen_scripts folder if not exists
    # This folder will consists of "real" run command in 
    # *.sh files.
    if not os.path.exists("gen_scripts"):
        os.makedirs("gen_scripts")

    date = datetime.datetime.now()
    experiment_id = os.path.splitext(os.path.basename(input_name))[0] + date.strftime("_%y%m%d_%H%M%S")
    base_str = "python -u "
    base_str = add_str(base_str, data["script_name"])       # python -u train_cv.py
    # python -u train_cv.py --root_dir -experiments/5fold_proposed_only_190105_151546
    base_str = add_str(base_str, "--root_dir {}".format(os.path.join("experiments", experiment_id))) 

    # Add flags in fixed_args
    for key, value in data["fixed_args"].items():
        base_str = add_str(base_str, "--{}".format(key))
        if type(value) == list:
            base_str = add_str(base_str, " ".join(map(str, value)))
        else:
            base_str = add_str(base_str, str(value))

    # currently dont see it in given json file
    if "combination_args" in data:
        commands = []
        for value_dict in data["combination_args"]:
            comb_str = base_str
            for key, value in value_dict.items():
                comb_str = add_str(comb_str, "--{}".format(key))
                if type(value) == list:
                    comb_str = add_str(comb_str, " ".join(map(str, value)))
                else:
                    comb_str = add_str(comb_str, str(value))
            commands.append(comb_str)
    else:
        commands = [base_str]

    # Add command for different model types
    # example: --model cnn_ego_pose_scale
    if "dynamic_args" in data:
        for key, value_list in data["dynamic_args"].items():
            commands = [add_str(cmd, "--{} {}".format(key, " ".join(map(str, v)) if type(v) == list else v)) for v in value_list for cmd in commands]

    if decision == "runtest" and "test_args" in data:
        for key, value in data["test_args"].items():
            commands = [add_str(cmd, "--{} {}".format(key, " ".join(map(str, value)) if type(value) == list else value)) for cmd in commands]
    
    # Add command for different splits.
    # example: --nb_splits 5 --eval_split 4 --gpu 1 
    if "cv" in data and not decision == "runtest":
        nb_splits = data["cv"]
        commands = [add_str(cmd, "--nb_splits {} --eval_split {}".format(nb_splits, sp)) for cmd in commands for sp in range(nb_splits)]

    commands = [add_str(cmd, "--gpu {}".format(gpu_id)) for cmd in commands]

    # Print all commands going to run
    for cmd in commands:
        print(cmd)

    if decision == "run" or decision == "runtest":

        # Store commands to gen_scirpts folder.
        script_path = os.path.join("gen_scripts", "{}.sh".format(experiment_id))
        print("Scripts written to {}".format(script_path))
        with open(script_path, "w") as f:
            f.write("#! /bin/sh\n")
            f.write("cd {}\n".format(os.getcwd()))
            for idx, cmd in enumerate(commands):
                f.write(cmd+"\n")
        cmd = "chmod +x {}".format(script_path)
        print(cmd)
        subprocess.call(cmd.split(" "))     # execute chmod +x gen_scripts/5fold_proposed_only_190107_210551.sh

        # Running all the generated commands  
        cmd = "sh {}".format(script_path)
        print(cmd)
        subprocess.call(cmd.split(" "))
    else:
        print(len(commands))
        print("Test finished")
