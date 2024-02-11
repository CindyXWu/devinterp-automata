#!/bin/bash

# Will run untl the end of time, unless killed by finding PID 
# Do this with ps aux | grep delete_contents.sh

# Specify the directory path
dir1="$HOME/.cache/wandb/artifacts/obj"
dir2="$HOME/.local/share/wandb/artifacts"

while true
do
    if [ -d "$dir1" ]; then
        rm -rf "$dir1"/*
    else
        echo "Directory $dir1 does not exist."
    fi
    if [ -d "$dir2" ]; then
        rm -rf "$dir2"/*
    else
        echo "Directory $dir2 does not exist."
    fi
    sleep 500
done