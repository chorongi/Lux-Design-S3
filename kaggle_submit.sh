#!/bin/bash
agent_dir=$1
description=$2
echo "Submitting Agent: $agent_dir"
cd $agent_dir
tar --exclude=__pycache__ -czvf submission.tar.gz *
kaggle competitions submit -c lux-ai-season-3 -f submission.tar.gz -m "$description"
cd ~/projects/Kaggle/Lux-Design-S3
