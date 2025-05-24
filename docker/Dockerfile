ARG ISAACLAB_BASE_IMAGE_ARG

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_MARS_JUMPER_PATH_ARG
ENV DOCKER_MARS_JUMPER_PATH=${DOCKER_MARS_JUMPER_PATH_ARG}

USER root

# # Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
# COPY ../ ${DOCKER_MARS_JUMPER_PATH} #commented out because we are using bind mount instead

# # Install whatever you need as additional dependencies.
RUN bash -i -c "source ${HOME}/.bashrc"

RUN git config --global --add safe.directory /workspace/mars_jumper


# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace