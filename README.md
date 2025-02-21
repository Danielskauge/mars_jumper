# Running the Container

## Build Base Image
1. Go to `isaaclab/docker`.
2. Execute: `python container.py start base`.

## Build and Run Mars Jumper
1. Go to `mars_jumper/docker`.
2. Build the image: `docker compose --env-file .env.base --file docker-compose.yaml build mars-jumper`.
3. Start the container:
   - Foreground: `docker compose --env-file .env.base --file docker-compose.yaml up`
   - Detached: `docker compose --env-file .env.base --file docker-compose.yaml up -d`

## Interact with the Container
- Access shell: `docker exec -it mars-jumper bash`
- Run TensorBoard: `tensorboard --logdir logs/rl_games/mars_jumper --host 0.0.0.0`

## Stop and Clean Up
- Stop container: `docker compose --env-file .env.base --file docker-compose.yaml down` or `docker stop mars-jumper`
- Remove container: `docker rm mars-jumper`
- Remove image: `docker rmi mars-jumper`

# Training
- run the training script: `python train.py --task mars-jumper-manager-based`
- optional arguments:
    - `--livestream 2`: view livestream in webrtc
    - `--video`: enables video recording during training
    - `--video_length`: length of each recorded video (in steps)
    - `--video_interval`: interval between each video recording (in steps)
    - Make sure to also add the `--enable_cameras` argument when running headless. Note that enabling recording is equivalent to enabling rendering during training, which will slow down both startup and runtime performance.
- run training with recording:
    - `python train.py --task mars-jumper-manager-based --video --video_length 1000 --video_interval 1000`

The recorded videos will be saved in the same directory as the training checkpoints, under IsaacLab/logs/<rl_workflow>/<task>/<run>/videos/train

# Evaluation






