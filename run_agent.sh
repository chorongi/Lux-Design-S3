#!/bin/bash
agent_dir=$1
today=$(date +'%m-%d-%Y')
luxai-s3 "$agent_dir"/main.py "$agent_dir"/main.py --output="$agent_dir"/replays/replay_"$today".html
