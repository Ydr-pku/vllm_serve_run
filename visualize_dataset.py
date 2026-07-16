import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize prompt and output-token length distributions."
    )
    parser.add_argument("dataset", type=Path, help="Input JSONL dataset path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization"),
        help="Directory for the PNG chart and JSON summary.",
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "Hugging Face tokenizer name or local model path. When provided, "
            "prompt lengths are measured in model tokens instead of characters."
        ),
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Measure raw prompt tokens without applying the tokenizer chat template.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Histogram bin count. Defaults to a value based on dataset size.",
    )
    return parser.parse_args()


def load_dataset(dataset_path):
    prompts = []
    output_lengths = []

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt = row["prompt"]
                output_tokens = int(row["output_tokens"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid dataset row at line {line_number}: {error}"
                ) from error

            if not isinstance(prompt, str):
                raise ValueError(
                    f"Invalid prompt at line {line_number}: expected a string."
                )
            if output_tokens < 1:
                raise ValueError(
                    f"Invalid output_tokens at line {line_number}: "
                    f"expected a positive integer, got {output_tokens}."
                )

            prompts.append(prompt)
            output_lengths.append(output_tokens)

    if not prompts:
        raise ValueError(f"No valid samples found in {dataset_path}.")

    return prompts, np.asarray(output_lengths, dtype=np.int64)


def load_tokenizer(tokenizer_path, trust_remote_code):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Token-based prompt lengths require transformers. Install it with "
            "`python -m pip install transformers`."
        ) from error

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )


def get_prompt_lengths(
    prompts,
    tokenizer_path=None,
    apply_chat_template=True,
    trust_remote_code=False,
):
    if tokenizer_path is None:
        return (
            np.asarray([len(prompt) for prompt in prompts], dtype=np.int64),
            "characters",
        )

    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code)
    prompt_lengths = []

    for prompt in prompts:
        text = prompt
        if apply_chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
        prompt_lengths.append(len(token_ids))

    return np.asarray(prompt_lengths, dtype=np.int64), "tokens"


def calculate_statistics(values):
    percentiles = np.percentile(values, [50, 90, 95, 99])
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "mean": float(values.mean()),
        "std_dev": float(values.std()),
        "p50": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "max": int(values.max()),
    }


def default_bin_count(sample_count):
    return max(10, min(60, int(math.sqrt(sample_count))))


def add_distribution_plot(axis, values, title, x_label, bins, color):
    axis.hist(
        values,
        bins=bins,
        color=color,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )
    mean = values.mean()
    median = np.median(values)
    axis.axvline(mean, color="#d62728", linewidth=1.5, label=f"mean={mean:.1f}")
    axis.axvline(
        median,
        color="#2ca02c",
        linewidth=1.5,
        linestyle="--",
        label=f"p50={median:.1f}",
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Requests")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()


def create_visualization(
    prompt_lengths,
    output_lengths,
    prompt_length_unit,
    output_path,
    bins,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install it with "
            "`python -m pip install matplotlib`."
        ) from error

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    add_distribution_plot(
        axes[0, 0],
        prompt_lengths,
        "Prompt length distribution",
        f"Prompt length ({prompt_length_unit})",
        bins,
        "#4c78a8",
    )
    add_distribution_plot(
        axes[0, 1],
        output_lengths,
        "Output length distribution",
        "Output tokens",
        bins,
        "#f58518",
    )

    if prompt_lengths.size >= 200:
        joint_plot = axes[1, 0].hexbin(
            prompt_lengths,
            output_lengths,
            gridsize=35,
            mincnt=1,
            cmap="viridis",
        )
        figure.colorbar(joint_plot, ax=axes[1, 0], label="Requests")
    else:
        axes[1, 0].scatter(
            prompt_lengths,
            output_lengths,
            alpha=0.65,
            s=24,
            color="#54a24b",
        )
    axes[1, 0].set_title("Prompt vs. output length")
    axes[1, 0].set_xlabel(f"Prompt length ({prompt_length_unit})")
    axes[1, 0].set_ylabel("Output tokens")
    axes[1, 0].grid(alpha=0.2)

    sorted_outputs = np.sort(output_lengths)
    theoretical_quantiles = np.sort(
        np.random.default_rng(0).normal(
            loc=output_lengths.mean(),
            scale=output_lengths.std(),
            size=sorted_outputs.size,
        )
    )
    axes[1, 1].scatter(
        theoretical_quantiles,
        sorted_outputs,
        alpha=0.55,
        s=16,
        color="#b279a2",
    )
    lower = min(theoretical_quantiles.min(), sorted_outputs.min())
    upper = max(theoretical_quantiles.max(), sorted_outputs.max())
    axes[1, 1].plot(
        [lower, upper],
        [lower, upper],
        color="#d62728",
        linewidth=1.3,
        linestyle="--",
        label="ideal normal",
    )
    axes[1, 1].set_title("Output-length normal Q-Q plot")
    axes[1, 1].set_xlabel("Theoretical normal quantiles")
    axes[1, 1].set_ylabel("Observed output tokens")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    prompts, output_lengths = load_dataset(args.dataset)
    prompt_lengths, prompt_length_unit = get_prompt_lengths(
        prompts,
        tokenizer_path=args.tokenizer,
        apply_chat_template=not args.skip_chat_template,
        trust_remote_code=args.trust_remote_code,
    )

    bins = args.bins or default_bin_count(len(prompts))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset.stem
    chart_path = args.output_dir / f"{dataset_name}_distribution.png"
    summary_path = args.output_dir / f"{dataset_name}_summary.json"

    create_visualization(
        prompt_lengths,
        output_lengths,
        prompt_length_unit,
        chart_path,
        bins,
    )

    correlation = (
        float(np.corrcoef(prompt_lengths, output_lengths)[0, 1])
        if prompt_lengths.size > 1
        else None
    )
    summary = {
        "dataset": str(args.dataset),
        "prompt_length_unit": prompt_length_unit,
        "prompt_length": calculate_statistics(prompt_lengths),
        "output_tokens": calculate_statistics(output_lengths),
        "prompt_output_correlation": correlation,
    }
    if prompt_length_unit == "tokens":
        summary["total_sequence_length"] = calculate_statistics(
            prompt_lengths + output_lengths
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Chart: {chart_path}")
    print(f"Summary: {summary_path}")
    print(
        f"Samples={len(prompts)}, "
        f"prompt_mean={prompt_lengths.mean():.2f} {prompt_length_unit}, "
        f"output_mean={output_lengths.mean():.2f} tokens"
    )


if __name__ == "__main__":
    main()
