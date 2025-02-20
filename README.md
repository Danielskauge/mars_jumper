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