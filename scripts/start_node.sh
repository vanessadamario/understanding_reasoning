#!/bin/bash

# config_runtime.sh COPYRIGHT Fujitsu Limited 2021 and FUJITSU LABORATORIES LTD. 2021
# Authors: Atsushi Kajita (kajita@fixstars.com), G R Ramdas Pillai (ramdas@fixstars.com)



. configure_runtime.sh

REDIS_PORT=${HEAD_PORT}

function error_msg() {
    echo "Usage:"
    echo "bash start_node.sh -n [head/worker] -a [head address if starting worker](optional)"
    echo 
    echo "Example: bash start_head.sh -n worker -a <head address>"
    echo
    exit  
}



while getopts n:a: option 
do 
    case "${option}" 
    in 
    n) NODE=${OPTARG};;
    a) HEAD_ADDR=${OPTARG};;
    *) echo "Insufficient arguements"
    esac 
done 

if [ -z "$NODE" ]; then
    error_msg
fi

if [ ${NODE} = "head" ]; then
    echo "head"

	ray start --head --port=${REDIS_PORT}  
	if [ $? -eq 1 ]; then
		echo "ray start failed! Retrying after ray stop."
		ray stop
		ray start --head --port=${REDIS_PORT}
		exit $?
	else
		exit 0
	fi
elif [ $NODE == "worker" ]; then

	if [ -z "$HEAD_ADDR" ]; then
		error_msg
	fi

	ray start --address=''${HEAD_ADDR}':'${REDIS_PORT}'' --redis-password='5241590000000000'
	if [ $? -eq 1 ]; then
		echo "ray start failed! Retrying after ray stop."
		ray stop
		ray start --address=''${HEAD_ADDR}':'${REDIS_PORT}'' --redis-password='5241590000000000'
		exit $?
    else
        exit 0
    fi
fi

