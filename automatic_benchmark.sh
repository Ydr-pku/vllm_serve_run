#!/usr/bin/env bash

set -m

# ================= 配置区 =================
PROXY_SCRIPT="${PROXY_SCRIPT:-./run_proxy_server.sh}"
PREFILLER_SCRIPT="${PREFILLER_SCRIPT:-./run_vllm_serve_8p8d_asyncSched_p.sh}"
DECODER_SCRIPT_BL="${DECODER_SCRIPT_BL:-./run_vllm_serve_8p8d_asyncSched_d_bl.sh}"
DECODER_SCRIPT_LB="${DECODER_SCRIPT_LB:-./run_vllm_serve_8p8d_asyncSched_d_lb.sh}"
DECODER_SCRIPT_DYNAM="${DECODER_SCRIPT_DYNAM:-./run_vllm_serve_8p8d_asyncSched_d_dynam.sh}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-./run_test_proxy_benchmark_manualPrompts_multiRound.sh}"

NUM_ROUNDS="${NUM_ROUNDS:-10}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
TEST_MODES="${TEST_MODES:-bl,lb,dynam}"
DATASET_PATH="${DATASET_PATH:-./mixed_prompts_lognormal.jsonl}"
BENCHMARK_TZ="${BENCHMARK_TZ:-UTC-8}"
INTERNAL_NO_PROXY_HOSTS="${INTERNAL_NO_PROXY_HOSTS:-61.28.30.29,127.0.0.1,localhost}"
INITIAL_STARTUP_DELAY="${INITIAL_STARTUP_DELAY:-10}"
DECODER_DRAIN_DELAY="${DECODER_DRAIN_DELAY:-30}"
RESOURCE_RELEASE_DELAY="${RESOURCE_RELEASE_DELAY:-5}"
READY_SETTLE_DELAY="${READY_SETTLE_DELAY:-3}"
SKIP_INITIAL_PROCESS_CLEANUP="${SKIP_INITIAL_PROCESS_CLEANUP:-0}"

LOG_DIR="${LOG_DIR:-./log}"
READY_STR_PROXY="${READY_STR_PROXY:-Application startup complete}"
READY_STR_PREFILLER="${READY_STR_PREFILLER:-Application startup complete}"
READY_STR_DECODER="${READY_STR_DECODER:-Application startup complete}"
export TZ="$BENCHMARK_TZ"
export no_proxy="${no_proxy:+${no_proxy},}${INTERNAL_NO_PROXY_HOSTS}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${INTERNAL_NO_PROXY_HOSTS}"
# ==========================================

usage() {
    cat <<'EOF'
用法:
  ./automatic_benchmark.sh [轮数] [模式列表] [数据集路径] [每轮请求数]
  ./automatic_benchmark.sh --rounds 10 --num-prompts 1000 --modes bl,lb,dynam --dataset-path Dataset-20260717-0930.jsonl

选项:
  -n, --rounds N       每种模式执行的 benchmark 轮数，默认 10
  -p, --num-prompts N  每轮 benchmark 的 request 数，默认 1000
  -m, --modes LIST     要测试的模式，逗号分隔；支持 bl、lb、dynam
  -d, --dataset-path P 数据集 JSONL 路径
      --drain-seconds N Benchmark 完成后停止 Decoder 前的 drain 秒数，默认 30
  -h, --help           显示帮助

示例:
  ./automatic_benchmark.sh 5
  ./automatic_benchmark.sh 10 bl,lb Dataset-20260717-0930.jsonl 2000
  ./automatic_benchmark.sh --rounds 3 --num-prompts 2000 --modes bl,dynam --dataset-path Dataset-20260717-0930.jsonl

也可以通过环境变量 NUM_ROUNDS、NUM_PROMPTS、TEST_MODES、DATASET_PATH、BENCHMARK_TZ、
INTERNAL_NO_PROXY_HOSTS 和 DECODER_DRAIN_DELAY 配置。
EOF
}

POSITIONAL_INDEX=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        -n|--rounds)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            NUM_ROUNDS=$2
            shift 2
            ;;
        -p|--num-prompts)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            NUM_PROMPTS=$2
            shift 2
            ;;
        -m|--modes)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            TEST_MODES=$2
            shift 2
            ;;
        -d|--dataset-path)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            DATASET_PATH=$2
            shift 2
            ;;
        --drain-seconds)
            [ "$#" -ge 2 ] || {
                echo "❌ $1 缺少参数" >&2
                exit 2
            }
            DECODER_DRAIN_DELAY=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "❌ 未知选项: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [ "$POSITIONAL_INDEX" -eq 0 ]; then
                NUM_ROUNDS=$1
            elif [ "$POSITIONAL_INDEX" -eq 1 ]; then
                TEST_MODES=$1
            elif [ "$POSITIONAL_INDEX" -eq 2 ]; then
                DATASET_PATH=$1
            elif [ "$POSITIONAL_INDEX" -eq 3 ]; then
                NUM_PROMPTS=$1
            else
                echo "❌ 多余参数: $1" >&2
                usage >&2
                exit 2
            fi
            POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
            shift
            ;;
    esac
done

if ! [[ "$NUM_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 轮数必须是正整数，当前值: $NUM_ROUNDS" >&2
    exit 2
fi

if ! [[ "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ 每轮 request 数必须是正整数，当前值: $NUM_PROMPTS" >&2
    exit 2
fi

if ! [[ "$DECODER_DRAIN_DELAY" =~ ^[0-9]+$ ]]; then
    echo "❌ Decoder drain 秒数必须是非负整数，当前值: $DECODER_DRAIN_DELAY" >&2
    exit 2
fi

IFS=',' read -r -a RAW_MODES <<< "$TEST_MODES"
MODES=()
for raw_mode in "${RAW_MODES[@]}"; do
    mode=$(printf '%s' "$raw_mode" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
    case "$mode" in
        bl|lb|dynam)
            duplicate=0
            for existing_mode in "${MODES[@]}"; do
                [ "$existing_mode" = "$mode" ] && duplicate=1
            done
            [ "$duplicate" -eq 1 ] || MODES+=("$mode")
            ;;
        "")
            ;;
        *)
            echo "❌ 不支持的模式: ${raw_mode}；仅支持 bl、lb、dynam" >&2
            exit 2
            ;;
    esac
done

if [ "${#MODES[@]}" -eq 0 ]; then
    echo "❌ 至少需要选择一种测试模式" >&2
    exit 2
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ 数据集文件不存在: $DATASET_PATH" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M')
DATASET_FILENAME="${DATASET_PATH##*/}"
DATASET_NAME="${DATASET_FILENAME%.*}"
SCRIPT_START_EPOCH=$(date +%s)

PROGRESS_BL=0
PROGRESS_LB=0
PROGRESS_DYNAM=0
LAST_DURATION_BL=0
LAST_DURATION_LB=0
LAST_DURATION_DYNAM=0
ROUND_START_BL=0
ROUND_START_LB=0
ROUND_START_DYNAM=0
STATUS_BL="等待"
STATUS_LB="等待"
STATUS_DYNAM="等待"

PROXY_PID=""
PREFILLER_PID=""
DECODER_PID=""
BENCHMARK_PID=""
DASHBOARD_ENABLED=0
DASHBOARD_STOPPED=0

mode_label() {
    case "$1" in
        bl) printf 'BL' ;;
        lb) printf 'LB' ;;
        dynam) printf 'DYNAM' ;;
    esac
}

modes_display() {
    local display=""
    local mode
    for mode in "${MODES[@]}"; do
        if [ -n "$display" ]; then
            display="${display} -> "
        fi
        display="${display}$(mode_label "$mode")"
    done
    printf '%s' "$display"
}

mode_decoder_script() {
    case "$1" in
        bl) printf '%s' "$DECODER_SCRIPT_BL" ;;
        lb) printf '%s' "$DECODER_SCRIPT_LB" ;;
        dynam) printf '%s' "$DECODER_SCRIPT_DYNAM" ;;
    esac
}

mode_result_file() {
    printf '%s/%s_%s_rounds%s_%s.txt' \
        "$LOG_DIR" "$RUN_TIMESTAMP" "$DATASET_NAME" "$NUM_ROUNDS" "$1"
}

get_mode_progress() {
    case "$1" in
        bl) printf '%s' "$PROGRESS_BL" ;;
        lb) printf '%s' "$PROGRESS_LB" ;;
        dynam) printf '%s' "$PROGRESS_DYNAM" ;;
    esac
}

get_mode_duration() {
    case "$1" in
        bl) printf '%s' "$LAST_DURATION_BL" ;;
        lb) printf '%s' "$LAST_DURATION_LB" ;;
        dynam) printf '%s' "$LAST_DURATION_DYNAM" ;;
    esac
}

get_mode_status() {
    case "$1" in
        bl) printf '%s' "$STATUS_BL" ;;
        lb) printf '%s' "$STATUS_LB" ;;
        dynam) printf '%s' "$STATUS_DYNAM" ;;
    esac
}

get_mode_round_start() {
    case "$1" in
        bl) printf '%s' "$ROUND_START_BL" ;;
        lb) printf '%s' "$ROUND_START_LB" ;;
        dynam) printf '%s' "$ROUND_START_DYNAM" ;;
    esac
}

set_mode_progress() {
    case "$1" in
        bl) PROGRESS_BL=$2 ;;
        lb) PROGRESS_LB=$2 ;;
        dynam) PROGRESS_DYNAM=$2 ;;
    esac
}

set_mode_duration() {
    case "$1" in
        bl) LAST_DURATION_BL=$2 ;;
        lb) LAST_DURATION_LB=$2 ;;
        dynam) LAST_DURATION_DYNAM=$2 ;;
    esac
}

set_mode_status() {
    case "$1" in
        bl) STATUS_BL=$2 ;;
        lb) STATUS_LB=$2 ;;
        dynam) STATUS_DYNAM=$2 ;;
    esac
}

set_mode_round_start() {
    case "$1" in
        bl) ROUND_START_BL=$2 ;;
        lb) ROUND_START_LB=$2 ;;
        dynam) ROUND_START_DYNAM=$2 ;;
    esac
}

format_duration() {
    local total_seconds=${1:-0}
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    if [ "$hours" -gt 0 ]; then
        printf '%dh%02dm%02ds' "$hours" "$minutes" "$seconds"
    elif [ "$minutes" -gt 0 ]; then
        printf '%dm%02ds' "$minutes" "$seconds"
    else
        printf '%ds' "$seconds"
    fi
}

format_epoch() {
    local epoch=$1
    if date -d "@$epoch" +'%Y-%m-%d %H:%M:%S' >/dev/null 2>&1; then
        date -d "@$epoch" +'%Y-%m-%d %H:%M:%S'
    else
        date -r "$epoch" +'%Y-%m-%d %H:%M:%S'
    fi
}

make_progress_bar() {
    local completed=$1
    local total=$2
    local width=28
    local filled=$((completed * width / total))
    local empty=$((width - filled))
    local bar=""
    local i

    for ((i = 0; i < filled; i++)); do
        bar="${bar}#"
    done
    for ((i = 0; i < empty; i++)); do
        bar="${bar}-"
    done
    printf '%s' "$bar"
}

completed_round_count() {
    local total=0
    local mode
    for mode in "${MODES[@]}"; do
        total=$((total + $(get_mode_progress "$mode")))
    done
    printf '%s' "$total"
}

eta_text() {
    local completed
    local total_rounds
    local now
    local elapsed
    local average_seconds
    local remaining
    local eta_epoch

    completed=$(completed_round_count)
    total_rounds=$((${#MODES[@]} * NUM_ROUNDS))
    now=$(date +%s)

    if [ "$completed" -eq 0 ]; then
        printf '预计完成时间: 等待首轮测试完成后估算'
        return
    fi

    elapsed=$((now - SCRIPT_START_EPOCH))
    average_seconds=$((elapsed / completed))
    remaining=$((total_rounds - completed))
    eta_epoch=$((now + average_seconds * remaining))
    printf '预计完成时间: %s' "$(format_epoch "$eta_epoch")"
}

dashboard_line() {
    printf '\033[2K%s\n' "$1"
}

render_dashboard() {
    [ "$DASHBOARD_ENABLED" -eq 1 ] || return

    local mode
    local completed
    local duration
    local duration_text
    local round_start
    local current_duration_text
    local status
    local modes_text
    modes_text=$(modes_display)

    printf '\0337\033[1;1H'
    dashboard_line "🚀 全自动化流水线测试：${modes_text}"
    dashboard_line "📄 当前数据集: ${DATASET_PATH}"
    dashboard_line "📂 每种模式 ${NUM_ROUNDS} 轮 | 每轮 ${NUM_PROMPTS} requests | 日志目录: ${LOG_DIR} | 启动时间: $(format_epoch "$SCRIPT_START_EPOCH")"

    for mode in "${MODES[@]}"; do
        completed=$(get_mode_progress "$mode")
        duration=$(get_mode_duration "$mode")
        round_start=$(get_mode_round_start "$mode")
        status=$(get_mode_status "$mode")
        if [ "$duration" -gt 0 ]; then
            duration_text="$(format_duration "$duration")"
        else
            duration_text="--"
        fi
        if [ "$round_start" -gt 0 ]; then
            current_duration_text="$(format_duration $(($(date +%s) - round_start)))"
        else
            current_duration_text="--"
        fi
        dashboard_line "$(printf '%-5s' "$(mode_label "$mode")") [$(make_progress_bar "$completed" "$NUM_ROUNDS")] [${completed}/${NUM_ROUNDS}] 上轮耗时: ${duration_text} | 本轮已运行: ${current_duration_text} | ${status}"
    done

    dashboard_line "⏱️  $(eta_text)"
    dashboard_line ""
    dashboard_line ""
    dashboard_line "============================================================"
    printf '\0338'
}

init_dashboard() {
    local required_lines
    required_lines=$((${#MODES[@]} + 7))

    if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ] && command -v tput >/dev/null 2>&1; then
        TERM_ROWS=$(tput lines 2>/dev/null || printf '24')
        if [ "$TERM_ROWS" -gt $((required_lines + 4)) ]; then
            DASHBOARD_ENABLED=1
            DASHBOARD_SCROLL_START=$((required_lines + 1))
            printf '\033[2J\033[H\033[?25l'
            printf '\033[%d;%dr' "$DASHBOARD_SCROLL_START" "$TERM_ROWS"
            render_dashboard
            printf '\033[%d;1H' "$DASHBOARD_SCROLL_START"
            return
        fi
    fi

    echo "🚀 全自动化流水线测试：$(modes_display)"
    echo "📄 当前数据集: ${DATASET_PATH}"
    echo "📂 每种模式 ${NUM_ROUNDS} 轮 | 每轮 ${NUM_PROMPTS} requests | 日志目录: ${LOG_DIR} | 启动时间: $(format_epoch "$SCRIPT_START_EPOCH")"
}

stop_dashboard() {
    [ "$DASHBOARD_STOPPED" -eq 0 ] || return
    DASHBOARD_STOPPED=1

    if [ "$DASHBOARD_ENABLED" -eq 1 ]; then
        render_dashboard
        printf '\033[r\033[%d;1H\033[?25h' "$TERM_ROWS"
    fi
}

log_event() {
    printf '%s %s\n' "$(date +'%Y%m%d_%H%M%S')" "$*"
}

cleanup() {
    local exit_code=$1
    trap - EXIT

    if [ "$exit_code" -eq 0 ]; then
        log_event "🧹 测试结束，正在释放所有组件资源..."
    else
        log_event "🧹 流程异常退出（code=${exit_code}），正在释放所有组件资源..."
    fi

    [ -n "$BENCHMARK_PID" ] && kill -9 -"$BENCHMARK_PID" 2>/dev/null
    [ -n "$DECODER_PID" ] && kill -9 -"$DECODER_PID" 2>/dev/null
    [ -n "$PROXY_PID" ] && kill -9 -"$PROXY_PID" 2>/dev/null
    [ -n "$PREFILLER_PID" ] && kill -9 -"$PREFILLER_PID" 2>/dev/null

    log_event "✅ 全局清理完成。"
    stop_dashboard
    exit "$exit_code"
}

trap 'cleanup $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_ready() {
    local log_file=$1
    local ready_str=$2
    local module_name=$3
    local pid=$4

    log_event "⏳ 等待 ${module_name} 就绪（关键字: '${ready_str}'）..."
    while true; do
        if grep -q "$ready_str" "$log_file" 2>/dev/null; then
            log_event "✅ ${module_name} 已就绪。"
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log_event "❌ ${module_name} 进程意外退出，请检查日志: ${log_file}"
            return 1
        fi
        render_dashboard
        sleep 2
    done
}

process_progress_file() {
    local progress_file=$1
    local target_mode=$2
    local current_line=0
    local event
    local event_mode
    local round
    local total
    local duration
    local round_status

    [ -f "$progress_file" ] || return

    while IFS='|' read -r event event_mode round total duration round_status; do
        current_line=$((current_line + 1))
        [ "$current_line" -le "$PROCESSED_PROGRESS_LINES" ] && continue
        [ "$event_mode" = "$target_mode" ] || continue

        case "$event" in
            ROUND_START)
                set_mode_round_start "$target_mode" "$duration"
                set_mode_status "$target_mode" "第 ${round}/${total} 轮运行中"
                ;;
            ROUND_END)
                set_mode_duration "$target_mode" "$duration"
                set_mode_round_start "$target_mode" 0
                if [ "$round_status" -eq 0 ]; then
                    set_mode_progress "$target_mode" "$round"
                    set_mode_status "$target_mode" "第 ${round}/${total} 轮完成"
                    if [ "$DASHBOARD_ENABLED" -eq 0 ]; then
                        log_event "📈 $(mode_label "$target_mode") [${round}/${total}] 上轮耗时: $(format_duration "$duration") | 本轮已运行: -- | $(eta_text)"
                    fi
                else
                    set_mode_status "$target_mode" "第 ${round}/${total} 轮失败"
                fi
                ;;
        esac
    done < "$progress_file"

    PROCESSED_PROGRESS_LINES=$current_line
}

run_benchmark_mode() {
    local mode=$1
    local result_file
    local progress_file
    local benchmark_status

    result_file=$(mode_result_file "$mode")
    progress_file="${TMPDIR:-/tmp}/automatic_benchmark_${RUN_TIMESTAMP}_${mode}_$$.progress"
    : > "$progress_file"
    PROCESSED_PROGRESS_LINES=0

    set_mode_status "$mode" "Benchmark 启动中"
    render_dashboard
    log_event "🚀 开始运行 $(mode_label "$mode") Benchmark，共 ${NUM_ROUNDS} 轮，每轮 ${NUM_PROMPTS} 个请求。"

    DATASET_PATH="$DATASET_PATH" \
        MODE="$mode" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        BENCHMARK_PROGRESS_FILE="$progress_file" \
        bash "$BENCHMARK_SCRIPT" \
            --mode "$mode" \
            --rounds "$NUM_ROUNDS" \
            --num-prompts "$NUM_PROMPTS" \
            --dataset-path "$DATASET_PATH" \
        > "$result_file" 2>&1 &
    BENCHMARK_PID=$!

    while kill -0 "$BENCHMARK_PID" 2>/dev/null; do
        process_progress_file "$progress_file" "$mode"
        render_dashboard
        sleep 1
    done

    wait "$BENCHMARK_PID"
    benchmark_status=$?
    BENCHMARK_PID=""
    process_progress_file "$progress_file" "$mode"
    rm -f "$progress_file"

    if [ "$benchmark_status" -ne 0 ]; then
        set_mode_status "$mode" "Benchmark 失败"
        render_dashboard
        log_event "❌ $(mode_label "$mode") Benchmark 失败（code=${benchmark_status}），请检查: ${result_file}"
        return "$benchmark_status"
    fi

    set_mode_status "$mode" "测试完成"
    render_dashboard
    log_event "📊 $(mode_label "$mode") 测试完成，结果已保存至: ${result_file}"
}

release_decoder() {
    local mode=$1

    if [ "$DECODER_DRAIN_DELAY" -gt 0 ]; then
        set_mode_status "$mode" "Drain 等待 ${DECODER_DRAIN_DELAY}s"
        render_dashboard
        log_event "⏳ $(mode_label "$mode") Benchmark 已结束，保留 Decoder ${DECODER_DRAIN_DELAY}s，等待在途 KV 传输和完成通知收尾..."
        sleep "$DECODER_DRAIN_DELAY"
    fi

    set_mode_status "$mode" "释放资源"
    render_dashboard
    log_event "⏳ 正在终止 Decoder ($(mode_label "$mode")) 进程 [PID: ${DECODER_PID}]..."
    kill -9 -"$DECODER_PID" 2>/dev/null
    wait "$DECODER_PID" 2>/dev/null
    DECODER_PID=""
    set_mode_status "$mode" "已完成"
    render_dashboard
    log_event "✅ Decoder ($(mode_label "$mode")) 资源已释放。"
}

init_dashboard

if [ "$SKIP_INITIAL_PROCESS_CLEANUP" -eq 1 ]; then
    log_event "🧪 已跳过启动前残留进程清理。"
else
    log_event "🧹 清理可能残留的 VLLM、python 和 vllm 进程..."
    pkill -9 VLLM 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    pkill -9 vllm 2>/dev/null || true
fi

log_event "▶️ 启动 Proxy Server..."
bash "$PROXY_SCRIPT" > "${LOG_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_ready "${LOG_DIR}/proxy.log" "$READY_STR_PROXY" "Proxy" "$PROXY_PID" || exit 1

log_event "▶️ 启动 Prefiller..."
bash "$PREFILLER_SCRIPT" > "${LOG_DIR}/prefiller.log" 2>&1 &
PREFILLER_PID=$!

first_mode=1
mode_index=0
for mode in "${MODES[@]}"; do
    mode_index=$((mode_index + 1))
    decoder_script=$(mode_decoder_script "$mode")
    decoder_log="${LOG_DIR}/decoder_${mode}.log"

    set_mode_status "$mode" "拉起服务"
    render_dashboard

    if [ "$first_mode" -eq 1 ] && [ "$INITIAL_STARTUP_DELAY" -gt 0 ]; then
        log_event "⏳ 基础服务启动缓冲 ${INITIAL_STARTUP_DELAY}s..."
        sleep "$INITIAL_STARTUP_DELAY"
    fi

    log_event "▶️ 启动 Decoder ($(mode_label "$mode"))..."
    bash "$decoder_script" > "$decoder_log" 2>&1 &
    DECODER_PID=$!

    if [ "$first_mode" -eq 1 ]; then
        wait_for_ready "${LOG_DIR}/prefiller.log" "$READY_STR_PREFILLER" "Prefiller" "$PREFILLER_PID" || exit 1
        first_mode=0
    fi
    wait_for_ready "$decoder_log" "$READY_STR_DECODER" "Decoder_$(mode_label "$mode")" "$DECODER_PID" || exit 1
    if [ "$READY_SETTLE_DELAY" -gt 0 ]; then
        sleep "$READY_SETTLE_DELAY"
    fi

    run_benchmark_mode "$mode" || exit $?
    release_decoder "$mode"

    if [ "$mode_index" -lt "${#MODES[@]}" ] && [ "$RESOURCE_RELEASE_DELAY" -gt 0 ]; then
        log_event "⏳ 等待底层硬件资源完全释放（${RESOURCE_RELEASE_DELAY}s）..."
        sleep "$RESOURCE_RELEASE_DELAY"
    fi
done

log_event "🎉 所有自动化测试流程执行完毕！"
