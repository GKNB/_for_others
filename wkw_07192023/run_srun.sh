#!/bin/bash
#SBATCH -A m2845_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load pytorch/2.0.1

cd /global/homes/t/tianle/_temp/_for_others/wkw_07192023

export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export IBV_FORK_SAFE=1
export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=32

echo "Start doing ML!"

num_epoch=5
CMD_ML="python /global/homes/t/tianle/_temp/_for_others/wkw_07192023/main.py \
		--batch_size=8 \
		--device=gpu \
		--epoch=${num_epoch} \
		--lr=0.0001 \
		--shuffle=local \
		--num_workers=1
		"
echo $CMD_ML

#SHARED_FILE=/global/homes/t/tianle/_temp/_for_others/wkw_07192023/sharedfile
SHARED_FILE=/pscratch/sd/t/tianle/sharedfile
if [[ -f "${SHARED_FILE}" ]];
then
	rm ${SHARED_FILE}
fi

srun $CMD_ML

echo "ML is finished!"
