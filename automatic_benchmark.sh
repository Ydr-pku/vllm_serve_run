#!/bin/bash

set -m
pkill -9 VLLM && pkill -9 python && pkill -9 vllm
# ================= 配置区 =================
PROXY_SCRIPT="./run_proxy_server.sh"
PREFILLER_SCRIPT="./run_vllm_serve_8p8d_asyncSched_p.sh"
DECODER_SCRIPT_BL="./run_vllm_serve_8p8d_asyncSched_d_bl.sh"
DECODER_SCRIPT_LB="./run_vllm_serve_8p8d_asyncSched_d_lb.sh"
DECODER_SCRIPT_DYNAM="./run_vllm_serve_8p8d_asyncSched_d_dynam.sh"
BENCHMARK_SCRIPT="./run_test_proxy_benchmark_manualPrompts_multiRound.sh"

LOG_DIR="./log"
mkdir -p "$LOG_DIR"
RESULT_BL="${LOG_DIR}/$(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M')_dataset10_bl.txt"
RESULT_LB="${LOG_DIR}/$(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M')_dataset10_lb.txt"
RESULT_DYNAM="${LOG_DIR}/$(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M')_dataset10_dynam.txt"

READY_STR_PREFILLER="Application startup complete" 
READY_STR_DECODER="Application startup complete"
# ==========================================

echo "🚀 $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 开始全自动化流水线测试 (BL 模式 -> LB 模式 -> DYNAM 模式)..."
echo "📂 日志目录: $LOG_DIR"

# 1. 动态清理机制
cleanup() {
    echo -e "\n🧹 正在触发全局清理，终止所有组件进程组..."
    # 使用 -9 和 -PID 强杀整个进程组，确保常驻的 Python Worker 彻底死掉
    [ -n "$PROXY_PID" ] && kill -9 -"$PROXY_PID" 2>/dev/null
    [ -n "$PREFILLER_PID" ] && kill -9 -"$PREFILLER_PID" 2>/dev/null
    [ -n "$DECODER_PID" ] && kill -9 -"$DECODER_PID" 2>/dev/null
    echo "✅ 清理完成。"
}
trap cleanup EXIT

# 2. 公共日志轮询等待函数
wait_for_ready() {
    local log_file=$1
    local ready_str=$2
    local module_name=$3
    local pid=$4
    
    echo "⏳ 等待 $module_name 就绪 (监控关键字: '$ready_str')..."
    while true; do
        if grep -q "$ready_str" "$log_file" 2>/dev/null; then
            echo "✅ $module_name 已就绪!"
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "❌ 错误: $module_name 进程意外退出，请检查日志 $log_file"
            exit 1
        fi
        sleep 2
    done
}

# ========================================================
# 基础服务初始化：启动 Proxy 和 Prefiller
# ========================================================
echo "========================================================"
echo "🎯 Step 1: 启动基础常驻服务 (Proxy & Prefiller)"
echo "========================================================"

echo "▶️ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 启动 Proxy Server..."
bash "$PROXY_SCRIPT" > "${LOG_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!

echo "▶️ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 启动 Prefiller..."
bash "$PREFILLER_SCRIPT" > "${LOG_DIR}/prefiller.log" 2>&1 &
PREFILLER_PID=$!

# ========================================================
# 阶段一：测试 BL Decoder
# ========================================================
echo -e "\n========================================================"
echo "🎯 Step 2: 开始 BL 模式测试"
echo "========================================================"
sleep 10
echo "▶️ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 启动 Decoder (BL)..."
bash "$DECODER_SCRIPT_BL" > "${LOG_DIR}/decoder_bl.log" 2>&1 &
DECODER_PID=$!

# 等待 Prefiller 就绪
wait_for_ready "${LOG_DIR}/prefiller.log" "$READY_STR_PREFILLER" "Prefiller" $PREFILLER_PID
# 等待 BL Decoder 就绪
wait_for_ready "${LOG_DIR}/decoder_bl.log" "$READY_STR_DECODER" "Decoder_BL" $DECODER_PID
sleep 3

echo "🚀 $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 运行 Benchmark (BL)..."

bash "$BENCHMARK_SCRIPT" "bl" "10" > "$RESULT_BL" 2>&1
echo "📊 $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') BL 测试完成，结果已保存至: $RESULT_BL"

# ========================================================
# 核心切换点：安全卸载 BL Decoder
# ========================================================
echo -e "\n========================================================"
echo "🎯 Step 3: 卸载 BL Decoder 并释放硬件资源"
echo "========================================================"
echo "⏳ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 正在终止 Decoder (BL) 进程 [PID: $DECODER_PID]..."

kill -9 -"$DECODER_PID" 2>/dev/null
# 等待该进程完全退出，确保硬件上下文正确释放
wait "$DECODER_PID" 2>/dev/null 
unset DECODER_PID

# 强烈建议此处 sleep 几秒，给底层驱动/运行时留出清理显存/NPU内存的时间
echo "⏳ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 等待底层硬件资源完全释放 (5s)..."
sleep 5
echo "✅ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 资源释放完毕。"

# ========================================================
# 阶段二：测试 LB Decoder
# ========================================================
echo -e "\n========================================================"
echo "🎯 Step 4: 开始 LB 模式测试"
echo "========================================================"

echo "▶️ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 启动 Decoder (LB)..."
bash "$DECODER_SCRIPT_LB" > "${LOG_DIR}/decoder_lb.log" 2>&1 &
DECODER_PID=$!

# 等待 LB Decoder 就绪
wait_for_ready "${LOG_DIR}/decoder_lb.log" "$READY_STR_DECODER" "Decoder_LB" $DECODER_PID
sleep 3

echo "🚀 $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 运行 Benchmark (LB)..."
bash "$BENCHMARK_SCRIPT" "lb" "10" > "$RESULT_LB" 2>&1
echo "📊 $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') LB 测试完成，结果已保存至: $RESULT_LB"

# ========================================================
# 核心切换点：安全卸载 LB Decoder
# ========================================================
echo -e "\n========================================================"
echo "🎯 Step 5: 卸载 LB Decoder 并释放硬件资源"
echo "========================================================"
echo "⏳ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 正在终止 Decoder (LB) 进程 [PID: $DECODER_PID]..."

kill -9 -"$DECODER_PID" 2>/dev/null
wait "$DECODER_PID" 2>/dev/null
unset DECODER_PID

echo "⏳ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 等待底层硬件资源完全释放 (5s)..."
sleep 5
echo "✅ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 资源释放完毕。"

# ========================================================
# 阶段三：测试 DYNAM Decoder (NONBSP_ENABLE=2)
# ========================================================
echo -e "\n========================================================"
echo "🎯 Step 6: 开始 DYNAM 模式测试"
echo "========================================================"

echo "▶️ $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 启动 Decoder (DYNAM)..."
bash "$DECODER_SCRIPT_DYNAM" > "${LOG_DIR}/decoder_dynam.log" 2>&1 &
DECODER_PID=$!

# 等待 DYNAM Decoder 就绪
wait_for_ready "${LOG_DIR}/decoder_dynam.log" "$READY_STR_DECODER" "Decoder_DYNAM" $DECODER_PID
sleep 3

echo "🚀 $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 运行 Benchmark (DYNAM)..."
bash "$BENCHMARK_SCRIPT" "dynam" "10" > "$RESULT_DYNAM" 2>&1
echo "📊 $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') DYNAM 测试完成，结果已保存至: $RESULT_DYNAM"

echo -e "\n🎉 $(TZ='Asia/Shanghai' date +'%Y%m%d_%H%M%S') 所有自动化测试流程执行完毕！"
# 退出脚本时，trap 会自动清理常驻的 Proxy 和 Prefiller 进程
