#!/bin/bash

# 获取命令行传入的第一个参数作为 note
NOTE=$1
# 获取传入的第二个参数作为运行轮数，如果未传则默认执行 1 轮
NUM_ROUNDS=${2:-1}
PROGRESS_FILE=${BENCHMARK_PROGRESS_FILE:-}

if ! [[ "$NUM_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 运行轮数必须是正整数，当前值: $NUM_ROUNDS" >&2
    exit 2
fi

emit_progress() {
    [ -n "$PROGRESS_FILE" ] || return
    printf '%s|%s|%s|%s|%s|%s\n' \
        "$1" "$NOTE" "$2" "$NUM_ROUNDS" "${3:-0}" "${4:-0}" \
        >> "$PROGRESS_FILE"
}

# 1. 环境准备
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/home/y00906461/20260228_vllm_ascend/20260713_pr_vllm_ascend/vllm/

# 2. 代理免疫：确保请求不经过 HIS Proxy
export no_proxy=$no_proxy,61.28.30.29
export NO_PROXY=$NO_PROXY,61.28.30.29

LOG_DIR="./log"
mkdir -p "${LOG_DIR}"

echo "🚀 计划连续执行 ${NUM_ROUNDS} 轮压测..."

# 外层循环控制多轮执行
for (( i=1; i<=NUM_ROUNDS; i++ )); do
    ROUND_START_EPOCH=$(date +%s)
    emit_progress "ROUND_START" "$i" "$ROUND_START_EPOCH"

    # 3. 获取当前时间戳（精确到秒，放在循环内部确保每轮时间不同）
    TIMESTAMP=$(TZ='Asia/Shanghai' date +"%Y%m%d_%H%M")

    # 4. 定义【压测客户端】专属的日志文件名（加入 round 轮次标识）
    if [ -z "$NOTE" ]; then
        BENCH_LOG="${LOG_DIR}/${TIMESTAMP}_vllm_bench_result_round${i}.log"
    else
        BENCH_LOG="${LOG_DIR}/${TIMESTAMP}_vllm_bench_result_${NOTE}_round${i}.log"
    fi

    echo "----------------------------------------------------"
    echo "${TIMESTAMP} ▶️ 开始第 ${i}/${NUM_ROUNDS} 轮压测"

    # 5. 启动压测
    vllm bench serve \
        --backend openai-chat \
        --host 61.28.30.29 \
        --port 8080 \
        --endpoint /v1/chat/completions \
        --dataset-name custom \
        --dataset-path ./mixed_prompts_lognormal.jsonl \
        --custom-output-len -1 \
        --max-concurrency 128 \
        --num-prompts 1000 \
        --model qwen3_30B \
        --tokenizer /home/y00906461/models/Qwen3-30B-A3B-Instruct-2507 \
        --save-result \
        --save-detailed \
        --temperature 0.0 \
        --metric-percentiles "50,90,99" \
        --request-rate 400
    ROUND_STATUS=$?
    ROUND_END_EPOCH=$(date +%s)
    ROUND_DURATION=$((ROUND_END_EPOCH - ROUND_START_EPOCH))
    emit_progress "ROUND_END" "$i" "$ROUND_DURATION" "$ROUND_STATUS"
    chmod -R 777 /home/y00906461/20260228_vllm_ascend

    if [ "$ROUND_STATUS" -ne 0 ]; then
        echo "❌ 第 ${i}/${NUM_ROUNDS} 轮压测失败，退出码: ${ROUND_STATUS}" >&2
        exit "$ROUND_STATUS"
    fi

    # 6. （可选）如果有多轮，在每轮之间给服务端一点缓冲时间释放连接和资源
    if [ "$i" -lt "$NUM_ROUNDS" ]; then
        echo "⏳ 第 ${i} 轮执行完毕，等待 5 秒后开始下一轮..."
        sleep 5
    fi
done
echo "${TIMESTAMP} ✅ 所有 ${NUM_ROUNDS} 轮压测已全部执行完毕！"
