#!/bin/bash
# Script for launching docker container

# Check for at least ?? required arguments
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <unique_id> <network> <gpu_id=all>"
  exit 1
fi

# INPUT ARGUMENT VARIABLES
unique_id="$1"
network="$2"
gpu_arg=$3

# OTHER VARIABLES
MARS_JUMPER_CODE_SRC_PATH="$HOME/workspace/mars_jumper"
MARS_JUMPER_CODE_DST_PATH="/workspace/mars_jumper"

case $gpu_arg in
    0)
        gpu_arg='"device=0"'
        container_name="mars-jumper-${unique_id}-${network}-gpu0"
        ;;
    1)
        gpu_arg='"device=1"'
        container_name="mars-jumper-${unique_id}-${network}-gpu1"
        ;;
    2)
        gpu_arg='"device=2"'
        container_name="mars-jumper-${unique_id}-${network}-gpu2"
        ;;
    *)
        gpu_arg=all
        container_name="mars-jumper-${unique_id}-${network}-gpuall"
        ;;
esac


echo "Launching container: $container_name"
#echo $MARS_JUMPER_CODE_SRC_PATH

docker run --name "$container_name" --entrypoint bash -it --gpus "$gpu_arg" -e "ACCEPT_EULA=Y" --rm --network=host \
        -e "PRIVACY_CONSENT=Y" \
        -e "OMNI_KIT_ALLOW_ROOT=1" \
        -v "${MARS_JUMPER_CODE_SRC_PATH}:${MARS_JUMPER_CODE_DST_PATH}:rw"\
    isaac-lab-base


