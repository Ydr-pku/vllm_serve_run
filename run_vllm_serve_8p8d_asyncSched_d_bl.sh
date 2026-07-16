#!/bin/bash
TIMESTAMP=$(TZ='Asia/Shanghai' date +"%Y%m%d_%H%M")
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/home/y00906461/20260228_vllm_ascend/20260713_pr_vllm_ascend/vllm/
export PYTHONPATH=$PYTHONPATH:/home/y00906461/20260228_vllm_ascend/20260713_pr_vllm_ascend/vllm-ascend/

export VLLM_TORCH_PROFILER_DIR="./profiling/${TIMESTAMP}"
# export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
# export VLLM_TORCH_PROFILER_WITH_STACK=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ASCEND_ENABLE_ACLGRAPH=1
export HCCL_IF_IP=61.28.30.29
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export VLLM_LOG_STATS_INTERVAL=5.0
export TASK_QUEUE_ENABLE=1
export MC_LOG_LEVEL=ERROR
export GLOG_minloglevel=1
export VLLM_LOGGING_LEVEL=WARNING

vllm serve /home/y00906461/models/Qwen3-30B-A3B-Instruct-2507  \
  --host 61.28.30.29 \
  --port 13701 \
  --async-scheduling \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --data-parallel-address 61.28.30.29 \
  --data-parallel-rpc-port 12777 \
  --max-num-seqs 8\
  --block-size 128 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name qwen3_30B \
  --max-model-len 110000  \
  --max-num-batched-tokens 32768  \
  --trust-remote-code \
  --gpu-memory-utilization 0.8  \
  --additional-config \
  '{"enable_cpu_binding":true,
  "NONBSP_ENABLE": 0,
  "NONBSP_START_STEP": 0,
  "NONBSP_END_STEP": -1,
  "NONBSP_BUBBLE_THRESHOLD": 5.0,
  "NONBSP_LONG_REQ_BLOCK_THRESHOLD": 700,
  "NONBSP_DYNAMIC_MAX_STEP": 256
  }' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "max_cudagraph_capture_size": 20, "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 20]}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "30100",
  "engine_id": "1",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 8,
                    "tp_size": 1
             }
      }
  }'