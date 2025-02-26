# Running the Container

## Build Base Image
<<<<<<< HEAD
1. Go to `isaaclab/docker`.
2. Execute: `python container.py start base`.

## Build and Run Mars Jumper
1. Go to `mars_jumper/docker`.
2. Build the image: `docker compose --env-file .env.base --file docker-compose.yaml build mars-jumper`.
3. Start the container:
   - Foreground: `docker compose --env-file .env.base --file docker-compose.yaml up <container_name>`
   - Detached: `docker compose --env-file .env.base --file docker-compose.yaml up -d <container_name>`

Note: Running a container will overwrite the existing container with the same service name, and mount the mars jumper directory from your own user directory.

## Interact with the Container
- Access shell: `docker exec -it mars-jumper bash`
- Run TensorBoard: `tensorboard --logdir logs/rl_games/mars_jumper --host 0.0.0.0`

## Stop and Clean Up
- Stop container: `docker compose --env-file .env.base --file docker-compose.yaml down` or `docker stop mars-jumper`
- Remove container: `docker rm mars-jumper`
- Remove image: `docker rmi mars-jumper`

# Training
- run the training script: `python train.py --task mars-jumper-manager-based`
    - `--livestream 2`: view livestream in webrtc
    - `--headless`: run training without rendering
- run training with recording: `python train.py --task mars-jumper-manager-based --video --headless --enable_cameras`
    - `--video_length`: length of each recorded video (in steps)
    - `--video_interval`: interval between each video recording (in steps)
    - Make sure to also add the `--enable_cameras` argument when running headless. Note that enabling recording is equivalent to enabling rendering during training, which will slow down both startup and runtime performance.

The recorded videos will be saved in the same directory as the training checkpoints, under logs/<rl_workflow>/<task>/<run>/videos/train

# Evaluation

# Other functionality

## Select GPU
While in container, run `nvidia-smi` to see available GPUs.
Run `export CUDA_VISIBLE_DEVICES=<gpu_id>` to select a specific GPU. GPU id is 0, 1, 2.

## Kill processes
- `ps aux | grep train.py`
- `kill <pid>`


## How to convert URDF to USD

In this repository is a script `convert_urdf.py` taken from the isaaclab repository. For its input arguments, see the file itself. For an example of use, see below. 
```     
root@1bbd0e0a6993:/workspace/mars_jumper/scripts# python convert_urdf.py --headless "/workspace/mars_jumper/submodules/cad/simplified_robot/robot_urdf_try1/urdf/robot_urdf_try1.urdf" /workspace/mars_jumper/USD_files/example_usd/example_USD.usd
```
Some critical aspects of its use: 
1. If working remote, --headless or --livestream must be used, despite this being a script without visuals.  
2. The extra files generated together with the `.usd` file itself are critical. 
* a folder: `configuration`
* a file: `.asset_has`
* a file: `config.yaml`

To successfully load your `.usd` file in the webrtc GUI, your `.usd` file must remain in the same directory as these extra files. This is new, 6 months ago isaaclab had a script of the same name that generated a standalone usd file without the bloat. 





