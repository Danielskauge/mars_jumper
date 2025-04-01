
import subprocess
import shlex

# Define the base command
BASE_CMD = "python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train.py --task mars-jumper-manager-based --distributed --headless --wandb"

# List of additional parameters for each run
PARAMS = [
    "--run_name final_baseline",
    "--run_name 2x_num_envs --num_envs 256 --minibatch_size 2048",
    "--run_name 2x_num_envs_2x_num_minibatch --num_envs 256 --num_minibatches 8"
]

# Loop over each set of parameters and run the command
for param in PARAMS:
    cmd = f"{BASE_CMD} {param}"
    print(f"Running: {cmd}")
    subprocess.run(shlex.split(cmd))