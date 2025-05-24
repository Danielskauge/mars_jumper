# Mars Jumper Project

This document provides instructions for setting up and running the Mars Jumper simulation environment.

## Running the Container

### Build Base Image (If not already done)
1. Navigate to the `isaaclab/docker` directory within your Isaac Lab installation.
2. Execute:
   ```bash
   python container.py start base
   ```

### Build and Run Mars Jumper Container
1. Navigate to the `mars_jumper/docker` directory within this project.
2. Build the image:
   ```bash
   docker compose --env-file .env.base --file docker-compose.yaml build mars-jumper
   ```
3. Start the container in detached mode:
   ```bash
   docker compose --env-file .env.base --file docker-compose.yaml up -d mars-jumper-daniel
   ```
   *Note: This command uses the service name `mars-jumper-daniel`. Adjust if your service name differs. Running this will overwrite any existing container with the same service name and mount the local `mars_jumper` directory into the container.*

### Interact with the Container
- Access the container's shell:
  ```bash
  docker exec -it mars-jumper-daniel bash
  ```
  *(Replace `mars-jumper-daniel` if using a different service name)*
- Run TensorBoard (from within the container shell):
  ```bash
  tensorboard --logdir logs/rl_games/mars_jumper --bind_all
  ```
  *(Access TensorBoard via `http://<your-ip>:6006` in your browser)*

### Stop and Clean Up
- Stop the running container:
  ```bash
  docker compose --env-file .env.base --file docker-compose.yaml down
  ```
  *Alternatively, stop a specific container:*
  ```bash
  docker stop mars-jumper-daniel
  ```
- Remove the stopped container:
  ```bash
  docker rm mars-jumper-daniel
  ```
- Remove the Docker image:
  ```bash
  docker rmi mars-jumper
  ```

## Training

All training commands should be run from within the container's shell (`/workspace/mars_jumper`).

### Basic Training
```bash
python train.py --task full-jump --headless
```

### Training with Weights & Biases Logging
```bash
python train.py --task full-jump --headless --wandb --project takeoff_cmd --run <your_run_name>
```

### Hyperparameter Sweeping

Create a sweep:
```bash
python -m wandb sweep sweep.yaml --project <project_name>
```
Run a sweep agent:
```bash
python -m wandb agent danielskauge/mars_jumper-mars_jumper/<sweep_id>
```

### Training with Video Recording
```bash
python train.py --task mars-jumper-manager-based --headless --enable_cameras --video --video_length 500 --video_interval 10000
```
- `--enable_cameras`: Required for video recording in headless mode.
- `--video`: Enable video recording.
- `--video_length`: Duration of each recorded video (in simulation steps). Default: 500.
- `--video_interval`: Frequency of video recording (in simulation steps). Default: 10000.
- *Note: Enabling recording significantly slows down training.*
- Recorded videos are saved in: `logs/rl_games/mars_jumper/<run_name>/videos/train`

### Distributed Training
Ensure sufficient shared memory is allocated to the container (see `shm_size` in `docker-compose.yaml`, default is 1GB).

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train.py --task mars-jumper-manager-based --distributed --headless
```
- `--distributed`: Enable distributed training.
- `--nproc_per_node`: Number of processes (usually matches the number of GPUs available/intended).

### Distributed Training with W&B and Video
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train.py --task mars-jumper-manager-based --distributed --headless --enable_cameras --video --video_length 500 --video_interval 10000 --wandb --project_name takeoff --run_name <your_run_name>
```

### Selecting a Specific GPU
From within the container shell:
```bash
export CUDA_VISIBLE_DEVICES=<gpu_id> # e.g., export CUDA_VISIBLE_DEVICES=0
```
Check available GPUs: `nvidia-smi`

## Playing (Evaluating Trained Models)

All play commands should be run from within the container's shell (`/workspace/mars_jumper`).

### Play with Latest Checkpoint
This uses the checkpoint from the most recent training run found in `logs/rl_games/mars_jumper/`.
```bash
python play.py --task mars-jumper-manager-based --use_last_checkpoint --num_envs 1
```

### Play with Specific Checkpoint
```bash
python play.py --task mars-jumper-manager-based --checkpoint <path/to/your/checkpoint.pth> --num_envs 1
```
- Example path: `logs/rl_games/mars_jumper/<run_name>/nn/<checkpoint_name>.pth`

### Play with Video Recording (Headless)
```bash
python play.py --task mars-jumper-manager-based --use_last_checkpoint --headless --enable_cameras --video --video_length 200 --num_envs 1
```
- `--video`: Record a video of the playthrough.
- `--video_length`: Duration of the video (in simulation steps).
- The script will exit after recording one video of the specified length.
- Recorded videos are saved in: `logs/rl_games/mars_jumper/<run_name>/videos/play` (or a similar directory if a specific checkpoint is used).

## Utility Scripts

### Convert URDF to USD
The script `scripts/convert_urdf.py` converts URDF files to the USD format required by Isaac Sim. Run it from within the container.
```bash
python scripts/convert_urdf.py --headless <input_urdf_path> <output_usd_path>
```
Example:
```bash
python scripts/convert_urdf.py --headless /workspace/mars_jumper/submodules/cad/simplified_robot/robot_urdf_try1/urdf/robot_urdf_try1.urdf /workspace/mars_jumper/USD_files/example_usd/example_USD.usd
```
**Important:**
1.  `--headless` or `--livestream` is required when running remotely or without a graphical display.
2.  The conversion generates the `.usd` file along with auxiliary files/folders (`configuration/`, `.asset_info`, `config.yaml`). These **must** remain in the same directory as the `.usd` file for it to load correctly in simulations.

## Process Management (Inside Container)

- List running Python processes related to training:
  ```bash
  ps aux | grep train.py
  ```
- Kill a specific process by its ID (PID):
  ```bash
  kill <pid>
  ```
- Kill all processes containing "train":
  ```bash
  pkill -f train
  ```
- View system resource usage:
  ```bash
  htop
  ```
- If a process won't terminate, find its parent process and kill that:
  ```bash
  ps -ef --forest # Find the parent PID
  kill <parent_pid>
  ```