#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A RECUP
#PBS -l filesystems=home:grand:eagle

export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export IBV_FORK_SAFE=1

num_node=2
num_epoch=5

echo "Start doing ML!"

CMD_ML="python /home/twang3/myWork/myTest/pytorch/test_ddp/test_01/main.py \
		--batch_size=8 \
		--device=gpu \
		--epoch=${num_epoch} \
		--lr=0.0001 \
		--shuffle=local \
		--num_workers=1
		"
echo $CMD_ML

SHARED_FILE=/home/twang3/myWork/myTest/pytorch/test_ddp/test_01/sharedfile
if [[ -f "${SHARED_FILE}" ]];
then
	rm ${SHARED_FILE}
fi

NNODES_ML=${num_node}
NRANKS_PER_NODE_ML=4
NDEPTH=8
NTOTRANKS_ML=$(( NNODES_ML * NRANKS_PER_NODE_ML ))
echo "ML: NUM_OF_NODES= ${NNODES_ML} TOTAL_NUM_RANKS= ${NTOTRANKS_ML} RANKS_PER_NODE= ${NRANKS_PER_NODE_ML}"

#mpiexec -n ${NTOTRANKS_ML} --ppn ${NRANKS_PER_NODE_ML} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=8 ./set_affinity_gpu_polaris.sh $CMD_ML
mpiexec -n ${NTOTRANKS_ML} --ppn ${NRANKS_PER_NODE_ML} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=8 $CMD_ML

echo "ML is finished!"
