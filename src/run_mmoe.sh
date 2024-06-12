#!/bin/bash

# 定义参数
gpus=(0 1 2)
labels=("0 1 3")
#  "0 1 3" "0 1 3 4" "0 1 3 4 5" "0 1 3 4 5 6"
horizons=(0 1 2 3 4)
period=(0 1 2 3 4 5)
repeat=1  # 每个任务的重复运行次数
max_parallel_tasks=2  # 每个 GPU 上并行的最大任务数量

# 函数来生成任务命令
generate_commands() {
  for label in "${labels[@]}"; do
    for horizon in "${horizons[@]}"; do
      for period in "${period[@]}"; do
        for ((r=0; r<$repeat; r++)); do
          echo "python train_mmoe.py -n 100 -f 3 -b 100 -ho $horizon -p $period --labels $label"
        done
      done
    done
  done
}

# 检查指定 GPU 上是否有正在运行的 train_mmoe.py 进程
check_gpu_usage() {
  local gpu=$1
  local count=$(ps -aux | grep "train_mmoe.py" | grep -v grep | grep " -g $gpu" | wc -l)
  echo $count
}

# 提交任务到指定 GPU
submit_task() {
  local cmd=$1
  local gpu=$2
  echo "Submitting task: $cmd on GPU $gpu"
  $cmd -g $gpu &
  sleep 2  # 等待2秒以确保任务启动
}

# 生成所有命令
commands=()
while IFS= read -r cmd; do
  commands+=("$cmd")
done < <(generate_commands)

# 任务调度逻辑
for cmd in "${commands[@]}"; do
  submitted=false
  while [ "$submitted" = false ]; do
    for gpu in "${gpus[@]}"; do
      gpu_usage=$(check_gpu_usage $gpu)
      if [ "$gpu_usage" -lt $max_parallel_tasks ]; then
        submit_task "$cmd" "$gpu"
        submitted=true
        break
      fi
    done
    if [ "$submitted" = false ]; then
      sleep 5
    fi
  done
done

# 等待所有任务完成
wait
