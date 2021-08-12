#!/bin/bash
#SBATCH -J test                               # 作业名为 test
#SBATCH -o test.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=3                   # 单节点启动的进程数为 2
#SBATCH --cpus-per-task=3                     # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:titan_xp:1
#SBATCH -t 100:00:00                            # 任务运行的最长时间为 100 小时

source ~/.bashrc

# 设置运行环境
conda activate scan
cd /users10/zyzhang/multimodel/LC-ACSA
# 输入要执行的命令，例如 ./hello 或 python test.py 等
python train_ACD.py --dataset $1  --epoch 10 
