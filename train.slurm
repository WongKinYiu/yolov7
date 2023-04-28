#!/bin/sh
#SBATCH --partition=GPUQ # CHANGE THIS TO CPUQ IF YOU DON'T NEED GPUs!!
#SBATCH --account=ie-idi # READ THIS TO FIND YOUR ACCOUNT: https://www.hpc.ntnu.no/idun/getting-started-on-idun/accounting/
#SBATCH --mem=64GB # Total memory you need. Most DL jobs should do fine with 64GB.
#SBATCH --nodes=1 # Number of nodes. Leave this to 1 as long as you're not running multi-node/multi-gpu training
#SBATCH --cpus-per-task=4 # Number of CPU cores required. Most V100 nodes has 4 CPU cores per GPU, so to maximize usage of the nodes, leave it to 4 (unless you know that your specific node has more than 4 cores/GPU).
#SBATCH --job-name=train_rdd2022 # Set this to whatever you want
#SBATCH --output=/cluster/home/haakohu/out.slurm # The output file. NOTE! The output directory (/cluster/home/haakohu/ in this case)  has to exist before you submit the job!
#SBATCH --time=8:00:00 # Maximum running time of your job.
#SBATCH --export=ALL # Export all environment variables.
#SBATCH --gres=gpu:V100:1 # GPU type and number of GPUs.
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total of GPUS: $CUDA_VISIBLE_DEVICES"
nvidia-smi
nvidia-smi nvlink -s
nvidia-smi topo -m
module purge
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Anaconda3/2020.07 
# >>> conda initialize >>> You might not need this, but in my case I need it for some reason....
__conda_setup="$('/cluster/apps/eb/software/Anaconda3/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
   if [ -f "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh" ]; then
      . "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
   else
      export PATH="/cluster/apps/eb/software/Anaconda3/2020.07/bin:$PATH"
   fi
fi
unset __conda_setup
#conda activate cv # NOTE! Change the environment variable
cd /cluster/home/kaspems/yolov7 # CD to the folder containing your files on the server
python train.py --workers 8 --device 1 --batch-size 32 --data data/rdd2022.yaml --img 2016 1024 --cfg cfg/training/yolov7-rdd2022.yaml --weights 'yolov7_training.pt' --name rdd2022 --hyp data/hyp.rdd2022.yaml | tee ./log.txt