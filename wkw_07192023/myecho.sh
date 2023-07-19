#!/bin/bash

echo "global rank id = $SLURM_PROCID, local rank id = $SLURM_LOCALID, size = $SLURM_NPROCS, addr = $SLURM_LAUNCH_NODE_IPADDR"
