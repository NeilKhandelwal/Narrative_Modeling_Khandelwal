"""
Dataset balancing CLI for politext.

Provides commands for creating balanced datasets for sentiment bias
research and analysis.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from politext.config import load_config

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--seed",
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool, seed: int) -> None:
    """Politext dataset balancing tool.

    Create balanced datasets for sentiment bias research.
    """
    ctx.ensure_object(dict)

    ctx.obj["config"] = load_config(config_path=config) if config else load_config()
    ctx.obj["seed"] = seed

    random.seed(seed)

    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--balance-field",
    "-b",
    required=True,
    help="Field to balance on (e.g., 'keyword_categories', 'political_score')",
)
@click.option(
    "--samples-per-class",
    "-n",
    type=int,
    help="Number of samples per class (default: min class size)",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["undersample", "oversample"]),
    default="undersample",
    help="Balancing strategy (default: undersample)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "parquet", "csv"]),
    default="json",
    help="Output format (default: json)",
)
@click.pass_context
def by_class(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    balance_field: str,
    samples_per_class: int | None,
    strategy: str,
    format: str,
) -> None:
    """Balance dataset by class labels.

    Creates a balanced dataset where each class has equal representation.

    Example:
        politext-balance by-class data.json -b keyword_categories -n 100
    """
    config = ctx.obj["config"]
    seed = ctx.obj["seed"]

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.processed_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.processed_data_path / f"balanced_{timestamp}.{format}")

    click.echo(f"Loading data from: {input_file}")
    click.echo(f"Balancing on field: {balance_field}")

    try:
        # Load data
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
            items = df.to_dict("records")
        else:
            with open(input_file) as f:
                data = json.load(f)
            items = data.get("items", data) if isinstance(data, dict) else data

        if not items:
            click.echo("Error: No items found in input file", err=True)
            sys.exit(1)

        click.echo(f"Found {len(items)} items")

        # Group items by class
        class_buckets: dict[str, list] = defaultdict(list)

        for item in items:
            value = item.get(balance_field)

            # Handle list fields (e.g., keyword_categories)
            if isinstance(value, list):
                for v in value:
                    class_buckets[str(v)].append(item)
            elif value is not None:
                class_buckets[str(value)].append(item)

        if not class_buckets:
            click.echo(f"Error: No items have field '{balance_field}'", err=True)
            sys.exit(1)

        # Show class distribution
        click.echo(f"\nOriginal class distribution:")
        for cls, bucket_items in sorted(class_buckets.items(), key=lambda x: -len(x[1])):
            click.echo(f"  {cls}: {len(bucket_items)}")

        # Determine samples per class
        min_class_size = min(len(b) for b in class_buckets.values())
        max_class_size = max(len(b) for b in class_buckets.values())

        if samples_per_class is None:
            samples_per_class = min_class_size
        elif strategy == "undersample" and samples_per_class > min_class_size:
            click.echo(
                f"Warning: samples_per_class ({samples_per_class}) > min class size ({min_class_size})",
                err=True,
            )
            click.echo("  Some classes will have fewer samples", err=True)

        click.echo(f"\nTarget samples per class: {samples_per_class}")

        # Balance dataset
        balanced_items = []

        for cls, bucket_items in class_buckets.items():
            if strategy == "undersample":
                # Random undersample
                n_samples = min(samples_per_class, len(bucket_items))
                sampled = random.sample(bucket_items, n_samples)
            else:  # oversample
                # Random oversample with replacement
                if len(bucket_items) >= samples_per_class:
                    sampled = random.sample(bucket_items, samples_per_class)
                else:
                    sampled = random.choices(bucket_items, k=samples_per_class)

            balanced_items.extend(sampled)

        # Shuffle final dataset
        random.shuffle(balanced_items)

        click.echo(f"\nBalanced dataset: {len(balanced_items)} items")

        # Save output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "source_file": input_file,
                            "balance_field": balance_field,
                            "samples_per_class": samples_per_class,
                            "strategy": strategy,
                            "seed": seed,
                            "total_items": len(balanced_items),
                            "timestamp": datetime.now().isoformat(),
                        },
                        "items": balanced_items,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        elif format == "parquet":
            df = pd.DataFrame(balanced_items)
            df.to_parquet(output_path, compression="snappy")
        elif format == "csv":
            df = pd.DataFrame(balanced_items)
            df.to_csv(output_path, index=False)

        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Balancing failed")
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--left-keywords",
    "-l",
    required=True,
    multiple=True,
    help="Keywords indicating left-leaning content (can specify multiple)",
)
@click.option(
    "--right-keywords",
    "-r",
    required=True,
    multiple=True,
    help="Keywords indicating right-leaning content (can specify multiple)",
)
@click.option(
    "--samples-per-side",
    "-n",
    type=int,
    help="Number of samples per political side",
)
@click.option(
    "--include-neutral",
    is_flag=True,
    help="Include neutral (neither left nor right) samples",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "parquet", "csv"]),
    default="json",
    help="Output format (default: json)",
)
@click.pass_context
def political(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    left_keywords: tuple[str, ...],
    right_keywords: tuple[str, ...],
    samples_per_side: int | None,
    include_neutral: bool,
    format: str,
) -> None:
    """Create politically balanced dataset.

    Balances content between left-leaning and right-leaning based on keywords.

    Example:
        politext-balance political data.json \\
            -l "Democrat" -l "liberal" -l "Biden" \\
            -r "Republican" -r "conservative" -r "Trump" \\
            -n 500
    """
    config = ctx.obj["config"]
    seed = ctx.obj["seed"]

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.processed_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.processed_data_path / f"political_balanced_{timestamp}.{format}")

    click.echo(f"Loading data from: {input_file}")
    click.echo(f"Left keywords: {left_keywords}")
    click.echo(f"Right keywords: {right_keywords}")

    try:
        # Load data
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
            items = df.to_dict("records")
        else:
            with open(input_file) as f:
                data = json.load(f)
            items = data.get("items", data) if isinstance(data, dict) else data

        if not items:
            click.echo("Error: No items found in input file", err=True)
            sys.exit(1)

        click.echo(f"Found {len(items)} items")

        # Classify items
        left_items = []
        right_items = []
        neutral_items = []

        left_keywords_lower = [k.lower() for k in left_keywords]
        right_keywords_lower = [k.lower() for k in right_keywords]

        for item in items:
            text = item.get("text", "").lower()

            # Check for keyword matches in text and keyword_matches field
            item_keywords = []
            if "keyword_matches" in item:
                item_keywords = [m.get("keyword", "").lower() for m in item.get("keyword_matches", [])]

            has_left = any(k in text or k in item_keywords for k in left_keywords_lower)
            has_right = any(k in text or k in item_keywords for k in right_keywords_lower)

            if has_left and not has_right:
                left_items.append({**item, "political_label": "left"})
            elif has_right and not has_left:
                right_items.append({**item, "political_label": "right"})
            elif not has_left and not has_right:
                neutral_items.append({**item, "political_label": "neutral"})
            # Items with both left and right are ambiguous, skip them

        click.echo(f"\nClassification results:")
        click.echo(f"  Left-leaning: {len(left_items)}")
        click.echo(f"  Right-leaning: {len(right_items)}")
        click.echo(f"  Neutral: {len(neutral_items)}")

        # Determine sample size
        if samples_per_side is None:
            samples_per_side = min(len(left_items), len(right_items))

        if samples_per_side > len(left_items) or samples_per_side > len(right_items):
            click.echo(
                f"Warning: Not enough samples. Using {min(len(left_items), len(right_items))} per side.",
                err=True,
            )
            samples_per_side = min(len(left_items), len(right_items))

        # Sample
        balanced_left = random.sample(left_items, samples_per_side)
        balanced_right = random.sample(right_items, samples_per_side)

        balanced_items = balanced_left + balanced_right

        if include_neutral and neutral_items:
            n_neutral = min(len(neutral_items), samples_per_side)
            balanced_neutral = random.sample(neutral_items, n_neutral)
            balanced_items.extend(balanced_neutral)

        random.shuffle(balanced_items)

        click.echo(f"\nBalanced dataset: {len(balanced_items)} items")
        click.echo(f"  Left: {samples_per_side}")
        click.echo(f"  Right: {samples_per_side}")
        if include_neutral:
            click.echo(f"  Neutral: {min(len(neutral_items), samples_per_side)}")

        # Save output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "source_file": input_file,
                            "left_keywords": list(left_keywords),
                            "right_keywords": list(right_keywords),
                            "samples_per_side": samples_per_side,
                            "include_neutral": include_neutral,
                            "seed": seed,
                            "total_items": len(balanced_items),
                            "timestamp": datetime.now().isoformat(),
                        },
                        "items": balanced_items,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        elif format == "parquet":
            df = pd.DataFrame(balanced_items)
            df.to_parquet(output_path, compression="snappy")
        elif format == "csv":
            df = pd.DataFrame(balanced_items)
            df.to_csv(output_path, index=False)

        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Political balancing failed")
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for splits",
)
@click.option(
    "--train-ratio",
    default=0.8,
    help="Training set ratio (default: 0.8)",
)
@click.option(
    "--val-ratio",
    default=0.1,
    help="Validation set ratio (default: 0.1)",
)
@click.option(
    "--test-ratio",
    default=0.1,
    help="Test set ratio (default: 0.1)",
)
@click.option(
    "--stratify-field",
    type=str,
    help="Field to stratify splits on",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "parquet", "csv"]),
    default="json",
    help="Output format (default: json)",
)
@click.pass_context
def split(
    ctx: click.Context,
    input_file: str,
    output_dir: str | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify_field: str | None,
    format: str,
) -> None:
    """Split dataset into train/val/test sets.

    Creates reproducible train/validation/test splits with optional stratification.

    Example:
        politext-balance split data.json --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
    """
    config = ctx.obj["config"]
    seed = ctx.obj["seed"]

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        click.echo(f"Error: Ratios must sum to 1.0 (got {total_ratio})", err=True)
        sys.exit(1)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(config.storage.processed_data_path / f"splits_{timestamp}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading data from: {input_file}")
    click.echo(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

    try:
        # Load data
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
            items = df.to_dict("records")
        else:
            with open(input_file) as f:
                data = json.load(f)
            items = data.get("items", data) if isinstance(data, dict) else data

        if not items:
            click.echo("Error: No items found in input file", err=True)
            sys.exit(1)

        click.echo(f"Found {len(items)} items")

        if stratify_field:
            # Stratified split
            click.echo(f"Stratifying on: {stratify_field}")

            # Group by stratify field
            buckets: dict[str, list] = defaultdict(list)
            for item in items:
                value = item.get(stratify_field)
                if isinstance(value, list) and value:
                    value = value[0]  # Use first value if list
                buckets[str(value) if value else "unknown"].append(item)

            train_items = []
            val_items = []
            test_items = []

            for cls, bucket_items in buckets.items():
                random.shuffle(bucket_items)
                n = len(bucket_items)

                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)

                train_items.extend(bucket_items[:n_train])
                val_items.extend(bucket_items[n_train : n_train + n_val])
                test_items.extend(bucket_items[n_train + n_val :])

        else:
            # Random split
            items_shuffled = items.copy()
            random.shuffle(items_shuffled)

            n = len(items_shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_items = items_shuffled[:n_train]
            val_items = items_shuffled[n_train : n_train + n_val]
            test_items = items_shuffled[n_train + n_val :]

        # Shuffle each split
        random.shuffle(train_items)
        random.shuffle(val_items)
        random.shuffle(test_items)

        click.echo(f"\nSplit sizes:")
        click.echo(f"  Train: {len(train_items)}")
        click.echo(f"  Val: {len(val_items)}")
        click.echo(f"  Test: {len(test_items)}")

        # Save splits
        splits = [
            ("train", train_items),
            ("val", val_items),
            ("test", test_items),
        ]

        for split_name, split_items in splits:
            split_file = output_path / f"{split_name}.{format}"

            if format == "json":
                with open(split_file, "w") as f:
                    json.dump(
                        {
                            "metadata": {
                                "source_file": input_file,
                                "split": split_name,
                                "stratify_field": stratify_field,
                                "seed": seed,
                                "total_items": len(split_items),
                            },
                            "items": split_items,
                        },
                        f,
                        indent=2,
                        default=str,
                    )
            elif format == "parquet":
                df = pd.DataFrame(split_items)
                df.to_parquet(split_file, compression="snappy")
            elif format == "csv":
                df = pd.DataFrame(split_items)
                df.to_csv(split_file, index=False)

            click.echo(f"  Saved: {split_file}")

        click.echo(f"\nAll splits saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Split failed")
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--n-samples",
    "-n",
    required=True,
    type=int,
    help="Number of samples to select",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "parquet", "csv"]),
    default="json",
    help="Output format (default: json)",
)
@click.pass_context
def sample(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    n_samples: int,
    format: str,
) -> None:
    """Randomly sample items from dataset.

    Example:
        politext-balance sample data.json -n 1000 -o sample.json
    """
    config = ctx.obj["config"]
    seed = ctx.obj["seed"]

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.processed_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.processed_data_path / f"sample_{timestamp}.{format}")

    click.echo(f"Loading data from: {input_file}")

    try:
        # Load data
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
            items = df.to_dict("records")
        else:
            with open(input_file) as f:
                data = json.load(f)
            items = data.get("items", data) if isinstance(data, dict) else data

        if not items:
            click.echo("Error: No items found in input file", err=True)
            sys.exit(1)

        click.echo(f"Found {len(items)} items")

        if n_samples > len(items):
            click.echo(
                f"Warning: Requested {n_samples} samples but only {len(items)} available",
                err=True,
            )
            n_samples = len(items)

        sampled_items = random.sample(items, n_samples)

        click.echo(f"Sampled {len(sampled_items)} items")

        # Save output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "source_file": input_file,
                            "n_samples": n_samples,
                            "seed": seed,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "items": sampled_items,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        elif format == "parquet":
            df = pd.DataFrame(sampled_items)
            df.to_parquet(output_path, compression="snappy")
        elif format == "csv":
            df = pd.DataFrame(sampled_items)
            df.to_csv(output_path, index=False)

        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Sampling failed")
        sys.exit(1)


def main() -> None:
    """Main entry point for the balance CLI."""
    cli()


if __name__ == "__main__":
    main()
