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
    parser.add_argument(
        "--metadata",
        type=Path,
        help=(
            "Generation metadata JSON path. By default, the script looks for "
            "<dataset_stem>_metadata.json next to the dataset."
        ),
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


def calculate_float_statistics(values):
    percentiles = np.percentile(values, [50, 90, 95, 99])
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std_dev": float(values.std()),
        "p50": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "max": float(values.max()),
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


def load_generation_metadata(dataset_path, metadata_path=None):
    if metadata_path is None:
        metadata_path = dataset_path.with_name(
            f"{dataset_path.stem}_metadata.json"
        )
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def format_generation_annotation(generation_metadata):
    if not generation_metadata:
        return None

    prompt = generation_metadata.get("prompt_distribution", {})
    output = generation_metadata.get("output_distribution", {})

    if prompt.get("type") == "lognormal_ratio":
        prompt_formula = (
            "Prompt ratio: R ~ LogNormal("
            f"mu={prompt['mu']}, sigma={prompt['sigma']})"
        )
    elif prompt.get("type") == "normal_ratio":
        prompt_formula = (
            "Prompt ratio: R ~ Normal("
            f"mu={prompt['mean']}, sigma={prompt['std_dev']})"
        )
    else:
        prompt_formula = prompt.get("formula", "Prompt distribution: unknown")

    prompt_details = (
        f"R = clip(R, 0, 1); forced R=1: "
        f"{prompt.get('num_long_prompts', 0)}/{generation_metadata.get('total_count')}"
    )
    if output.get("type") == "conditional_truncated_normal":
        output_formula = (
            "Output mean: mu_i="
            f"{output.get('mean_min')}+"
            f"({output.get('mean_max')}-{output.get('mean_min')})"
            f"*(L_i/L_max)^{output.get('curve_power')}"
        )
        output_details = (
            "O_i ~ TruncNormal("
            f"mu_i, sigma={output.get('std_dev')}, "
            f"range=[{output.get('min')}, {output.get('max')}])"
        )
    else:
        output_formula = (
            "Output: O ~ TruncNormal("
            f"mu={output.get('mean')}, sigma={output.get('std_dev')}, "
            f"range=[{output.get('min')}, {output.get('max')}])"
        )
        output_details = None
    seed_details = (
        f"Seeds: prompt={prompt.get('seed')}, output={output.get('seed')}"
    )
    annotation_lines = [prompt_formula, prompt_details, output_formula]
    if output_details:
        annotation_lines.append(output_details)
    annotation_lines.append(seed_details)
    return "\n".join(annotation_lines)


def calculate_conditional_output_means(prompt_lengths, generation_metadata):
    if not generation_metadata:
        return None

    output = generation_metadata.get("output_distribution", {})
    if output.get("type") != "conditional_truncated_normal":
        return None

    max_prompt_length = prompt_lengths.max()
    if max_prompt_length <= 0:
        normalized_lengths = np.zeros_like(prompt_lengths, dtype=np.float64)
    else:
        normalized_lengths = prompt_lengths / max_prompt_length

    return output["mean_min"] + (
        output["mean_max"] - output["mean_min"]
    ) * np.power(normalized_lengths, output["curve_power"])


def add_qq_plot(axis, observed_values, title, x_label, y_label):
    sorted_observed = np.sort(observed_values)
    theoretical_quantiles = np.sort(
        np.random.default_rng(0).normal(
            loc=observed_values.mean(),
            scale=observed_values.std(),
            size=sorted_observed.size,
        )
    )
    axis.scatter(
        theoretical_quantiles,
        sorted_observed,
        alpha=0.55,
        s=16,
        color="#b279a2",
    )
    lower = min(theoretical_quantiles.min(), sorted_observed.min())
    upper = max(theoretical_quantiles.max(), sorted_observed.max())
    axis.plot(
        [lower, upper],
        [lower, upper],
        color="#d62728",
        linewidth=1.3,
        linestyle="--",
        label="ideal normal",
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.2)
    axis.legend()


def create_visualization(
    prompt_lengths,
    output_lengths,
    prompt_length_unit,
    output_path,
    bins,
    generation_metadata=None,
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
    annotation = format_generation_annotation(generation_metadata)
    if annotation:
        axes[0, 0].text(
            0.02,
            0.97,
            annotation,
            transform=axes[0, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": "#777777",
                "alpha": 0.9,
            },
        )
    add_distribution_plot(
        axes[0, 1],
        output_lengths,
        "Output length distribution",
        "Output tokens",
        bins,
        "#f58518",
    )

    conditional_means = calculate_conditional_output_means(
        prompt_lengths,
        generation_metadata,
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
    if conditional_means is not None:
        curve_order = np.argsort(prompt_lengths)
        axes[1, 0].plot(
            prompt_lengths[curve_order],
            conditional_means[curve_order],
            color="#d62728",
            linewidth=2,
            label="conditional mean mu(L)",
        )
        axes[1, 0].legend()

    if conditional_means is not None:
        output_std_dev = generation_metadata["output_distribution"]["std_dev"]
        standardized_residuals = (
            output_lengths - conditional_means
        ) / output_std_dev
        add_qq_plot(
            axes[1, 1],
            standardized_residuals,
            "Conditional residual normal Q-Q plot",
            "Theoretical standard-normal quantiles",
            "Observed standardized residuals",
        )
    else:
        add_qq_plot(
            axes[1, 1],
            output_lengths,
            "Output-length normal Q-Q plot",
            "Theoretical normal quantiles",
            "Observed output tokens",
        )

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def visualize_dataset(
    dataset_path,
    output_dir=Path("visualization"),
    tokenizer_path=None,
    apply_chat_template=True,
    trust_remote_code=False,
    bins=None,
    generation_metadata=None,
):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    prompts, output_lengths = load_dataset(dataset_path)
    prompt_lengths, prompt_length_unit = get_prompt_lengths(
        prompts,
        tokenizer_path=tokenizer_path,
        apply_chat_template=apply_chat_template,
        trust_remote_code=trust_remote_code,
    )

    bins = bins or default_bin_count(len(prompts))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_path.stem
    chart_path = output_dir / f"{dataset_name}_distribution.png"
    summary_path = output_dir / f"{dataset_name}_summary.json"

    create_visualization(
        prompt_lengths,
        output_lengths,
        prompt_length_unit,
        chart_path,
        bins,
        generation_metadata=generation_metadata,
    )

    correlation = (
        float(np.corrcoef(prompt_lengths, output_lengths)[0, 1])
        if prompt_lengths.size > 1
        else None
    )
    summary = {
        "dataset": str(dataset_path),
        "prompt_length_unit": prompt_length_unit,
        "prompt_length": calculate_statistics(prompt_lengths),
        "output_tokens": calculate_statistics(output_lengths),
        "prompt_output_correlation": correlation,
    }
    if generation_metadata:
        summary["generation"] = generation_metadata
        conditional_means = calculate_conditional_output_means(
            prompt_lengths,
            generation_metadata,
        )
        if conditional_means is not None:
            output_std_dev = generation_metadata["output_distribution"][
                "std_dev"
            ]
            standardized_residuals = (
                output_lengths - conditional_means
            ) / output_std_dev
            summary["conditional_output_mean"] = calculate_float_statistics(
                conditional_means
            )
            summary["standardized_output_residual"] = (
                calculate_float_statistics(standardized_residuals)
            )
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
    return chart_path, summary_path


def main():
    args = parse_args()
    generation_metadata = load_generation_metadata(
        args.dataset,
        metadata_path=args.metadata,
    )
    visualize_dataset(
        args.dataset,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        apply_chat_template=not args.skip_chat_template,
        trust_remote_code=args.trust_remote_code,
        bins=args.bins,
        generation_metadata=generation_metadata,
    )


if __name__ == "__main__":
    main()
