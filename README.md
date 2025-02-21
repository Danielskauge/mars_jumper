# How to Run the Container

## Build Base Image
First build the Isaac Lab base image if you haven't already:
1. Navigate to `isaaclab/docker`
2. Run: `python container.py start base`

Then build the Mars Jumper image:
1. Navigate to `mars_jumper/docker`
2. Run: `docker compose --env-file .env.base --file docker-compose.yaml build mars-jumper`

Then run the Mars Jumper container, first navigate to mars_jumper/docker and run:
```
docker compose --env-file .env.base --file docker-compose.yaml up
```
or run in detached mode:
```
docker compose --env-file .env.base --file docker-compose.yaml up -d
```
run commands inside the running container:
```
docker exec -it mars-jumper bash
```
run tensorboard:
```
tensorboard --logdir logs/rl_games/mars_jumper --host 0.0.0.0
```

stop the container:
```
docker compose --env-file .env.base --file docker-compose.yaml down
```
or 
```
docker stop mars-jumper
```
remove the container:
```
docker rm mars-jumper
```
remove the image:
```
docker rmi mars-jumper
```

# Other functionality

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

