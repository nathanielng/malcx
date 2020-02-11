#!/bin/bash

# ----- Kill the docker container -----
CONTAINER_ID=`docker ps | grep intellegensdocker | awk '{print $1}'`
docker kill ${CONTAINER_ID}

# ----- Delete the lock file -----
rm -v ${ALCX_RUN_FOLDER}/lock
