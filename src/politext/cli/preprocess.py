"""
Data preprocessing CLI for politext.

Provides commands for cleaning, filtering, and enriching collected
political text data.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from politext.config import load_config
from politext.preprocessing import TextCleaner
from politext.detection import KeywordMatcher, PoliticalClassifier

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
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """Politext preprocessing tool.

    Clean, filter, and enrich collected political text data.
    """
    ctx.ensure_object(dict)

    ctx.obj["config"] = load_config(config_path=config) if config else load_config()

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
    "--remove-urls/--keep-urls",
    default=True,
    help="Remove URLs from text (default: remove)",
)
@click.option(
    "--remove-mentions/--keep-mentions",
    default=False,
    help="Remove @mentions (default: keep)",
)
@click.option(
    "--normalize-hashtags/--keep-hashtags",
    default=True,
    help="Normalize hashtags (default: normalize)",
)
@click.option(
    "--language",
    "-l",
    default="en",
    help="Filter by language (default: en, use 'all' to skip filtering)",
)
@click.option(
    "--min-length",
    default=10,
    help="Minimum text length after cleaning (default: 10)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "parquet", "csv"]),
    default="json",
    help="Output format (default: json)",
)
@click.pass_context
def clean(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    remove_urls: bool,
    remove_mentions: bool,
    normalize_hashtags: bool,
    language: str,
    min_length: int,
    format: str,
) -> None:
    """Clean and normalize collected text data.

    Takes a JSON file with collected tweets/text and outputs cleaned data.

    Example:
        politext-preprocess clean data/raw/tweets.json -o data/processed/cleaned.json
    """
    config = ctx.obj["config"]

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.processed_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.processed_data_path / f"cleaned_{timestamp}.{format}")

    click.echo(f"Loading data from: {input_file}")

    try:
        with open(input_file) as f:
            data = json.load(f)

        items = data.get("items", data) if isinstance(data, dict) else data

        if not items:
            click.echo("Error: No items found in input file", err=True)
            sys.exit(1)

        click.echo(f"Found {len(items)} items to process")

        cleaner = TextCleaner(
            remove_urls=remove_urls,
            remove_mentions=remove_mentions,
            normalize_hashtags=normalize_hashtags,
            detect_language=language != "all",
            min_length=min_length,
        )

        cleaned_items = []
        skipped_count = 0

        with tqdm(items, desc="Cleaning") as pbar:
            for item in pbar:
                text = item.get("text", "")
                if not text:
                    skipped_count += 1
                    continue

                result = cleaner.clean(text)

                # Language filter
                if language != "all" and result.language != language:
                    skipped_count += 1
                    continue

                # Length filter
                if len(result.cleaned) < min_length:
                    skipped_count += 1
                    continue

                cleaned_item = {
                    **item,
                    "text_original": text,
                    "text": result.cleaned,
                    "language": result.language,
                    "urls_removed": result.removed_urls,
                    "mentions_removed": result.removed_mentions,
                    "hashtags_normalized": result.normalized_hashtags,
                    "emoji_count": result.emoji_count,
                    "original_length": len(text),
                    "cleaned_length": len(result.cleaned),
                }
                cleaned_items.append(cleaned_item)

        click.echo(f"\nCleaned {len(cleaned_items)} items ({skipped_count} skipped)")

        # Save output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "source_file": input_file,
                            "items_processed": len(items),
                            "items_kept": len(cleaned_items),
                            "language_filter": language,
                            "min_length": min_length,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "items": cleaned_items,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        elif format == "parquet":
            df = pd.DataFrame(cleaned_items)
            df.to_parquet(output_path, compression="snappy")
        elif format == "csv":
            df = pd.DataFrame(cleaned_items)
            df.to_csv(output_path, index=False)

        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Cleaning failed")
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
    "--keywords-dir",
    "-k",
    type=click.Path(exists=True),
    help="Directory containing keyword JSON files",
)
@click.option(
    "--min-score",
    default=0.3,
    help="Minimum political relevance score (0-1, default: 0.3)",
)
@click.option(
    "--include-entities/--no-entities",
    default=True,
    help="Include named entity recognition (default: yes)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "parquet"]),
    default="json",
    help="Output format (default: json)",
)
@click.pass_context
def detect(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    keywords_dir: str | None,
    min_score: float,
    include_entities: bool,
    format: str,
) -> None:
    """Detect political content in preprocessed text.

    Runs keyword matching and optionally entity recognition to
    identify and score political relevance.

    Example:
        politext-preprocess detect data/processed/cleaned.json -o data/annotated/detected.json
    """
    config = ctx.obj["config"]

    if keywords_dir is None:
        keywords_dir = str(config.detection.keywords_path)

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.annotated_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.annotated_data_path / f"detected_{timestamp}.{format}")

    click.echo(f"Loading data from: {input_file}")
    click.echo(f"Using keywords from: {keywords_dir}")

    try:
        # Load input data
        with open(input_file) as f:
            data = json.load(f)

        items = data.get("items", data) if isinstance(data, dict) else data

        if not items:
            click.echo("Error: No items found in input file", err=True)
            sys.exit(1)

        click.echo(f"Found {len(items)} items to process")

        # Initialize keyword matcher
        keywords_path = Path(keywords_dir)
        matcher = KeywordMatcher()

        # Load keyword files
        keyword_files = list(keywords_path.glob("*.json"))
        for kw_file in keyword_files:
            matcher.load_keywords(kw_file)
            click.echo(f"  Loaded keywords from: {kw_file.name}")

        # Initialize entity recognizer if requested
        entity_recognizer = None
        if include_entities:
            try:
                from politext.detection import EntityRecognizer

                entity_recognizer = EntityRecognizer(
                    model_name=config.preprocessing.spacy_model
                )
                click.echo(f"  Entity recognition enabled (model: {config.preprocessing.spacy_model})")
            except Exception as e:
                click.echo(f"  Warning: Entity recognition disabled - {e}", err=True)
                include_entities = False

        detected_items = []
        filtered_count = 0

        with tqdm(items, desc="Detecting") as pbar:
            for item in pbar:
                text = item.get("text", "")
                if not text:
                    continue

                # Keyword matching
                matches = matcher.find_matches(text)
                political_score = matcher.calculate_political_score(text)

                # Filter by score
                if political_score < min_score:
                    filtered_count += 1
                    continue

                detected_item = {
                    **item,
                    "political_score": political_score,
                    "keyword_matches": [
                        {
                            "keyword": m.keyword,
                            "category": m.category,
                            "subcategory": m.subcategory,
                            "weight": m.weight,
                        }
                        for m in matches
                    ],
                    "keyword_categories": list(set(m.category for m in matches)),
                }

                # Entity recognition
                if include_entities and entity_recognizer:
                    entities = entity_recognizer.extract_entities(text)
                    detected_item["entities"] = [
                        {
                            "text": e.text,
                            "label": e.label,
                            "political_type": e.political_type,
                        }
                        for e in entities
                    ]

                detected_items.append(detected_item)

        click.echo(f"\nDetected political content in {len(detected_items)} items")
        click.echo(f"Filtered out {filtered_count} items (score < {min_score})")

        # Save output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "source_file": input_file,
                            "keywords_dir": keywords_dir,
                            "min_score": min_score,
                            "include_entities": include_entities,
                            "items_processed": len(items),
                            "items_kept": len(detected_items),
                            "timestamp": datetime.now().isoformat(),
                        },
                        "items": detected_items,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        elif format == "parquet":
            # Flatten nested structures for parquet
            flat_items = []
            for item in detected_items:
                flat_item = {k: v for k, v in item.items() if not isinstance(v, (list, dict))}
                flat_item["keyword_matches_json"] = json.dumps(item.get("keyword_matches", []))
                flat_item["keyword_categories_json"] = json.dumps(item.get("keyword_categories", []))
                if "entities" in item:
                    flat_item["entities_json"] = json.dumps(item.get("entities", []))
                flat_items.append(flat_item)

            df = pd.DataFrame(flat_items)
            df.to_parquet(output_path, compression="snappy")

        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Detection failed")
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
    "--hash-salt",
    type=str,
    help="Salt for hashing (overrides config)",
)
@click.option(
    "--remove-pii/--keep-pii",
    default=True,
    help="Detect and remove PII (default: remove)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "parquet"]),
    default="json",
    help="Output format (default: json)",
)
@click.pass_context
def anonymize(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    hash_salt: str | None,
    remove_pii: bool,
    format: str,
) -> None:
    """Anonymize sensitive data for ethical research.

    Hashes usernames and optionally removes detected PII.

    Example:
        politext-preprocess anonymize data/processed/cleaned.json --hash-salt mysalt
    """
    config = ctx.obj["config"]

    if hash_salt is None:
        hash_salt = config.ethics.hash_salt

    if not hash_salt:
        click.echo("Warning: No hash salt provided. Using empty salt.", err=True)

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.processed_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.processed_data_path / f"anonymized_{timestamp}.{format}")

    click.echo(f"Loading data from: {input_file}")

    try:
        from politext.ethics.anonymizer import Anonymizer

        with open(input_file) as f:
            data = json.load(f)

        items = data.get("items", data) if isinstance(data, dict) else data

        if not items:
            click.echo("Error: No items found in input file", err=True)
            sys.exit(1)

        click.echo(f"Found {len(items)} items to anonymize")

        anonymizer = Anonymizer(
            hash_salt=hash_salt,
            hash_algorithm=config.ethics.hash_algorithm,
            remove_pii=remove_pii,
        )

        anonymized_items = []
        pii_found_count = 0

        with tqdm(items, desc="Anonymizing") as pbar:
            for item in pbar:
                anon_item = item.copy()

                # Hash username
                if "author_username" in anon_item:
                    anon_item["author_username_hash"] = anonymizer.anonymize_username(
                        anon_item["author_username"]
                    )
                    del anon_item["author_username"]

                if "author_id" in anon_item:
                    anon_item["author_id_hash"] = anonymizer.anonymize_user_id(
                        str(anon_item["author_id"])
                    )
                    del anon_item["author_id"]

                # Remove PII from text
                if remove_pii and "text" in anon_item:
                    result = anonymizer.anonymize_text(anon_item["text"])
                    anon_item["text"] = result.anonymized
                    if result.pii_found:
                        pii_found_count += 1
                        anon_item["pii_types_removed"] = [m.pii_type for m in result.pii_found]

                anonymized_items.append(anon_item)

        click.echo(f"\nAnonymized {len(anonymized_items)} items")
        if remove_pii:
            click.echo(f"PII found and removed in {pii_found_count} items")

        # Save output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "source_file": input_file,
                            "pii_removal": remove_pii,
                            "hash_algorithm": config.ethics.hash_algorithm,
                            "items_processed": len(anonymized_items),
                            "pii_found_count": pii_found_count,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "items": anonymized_items,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        elif format == "parquet":
            df = pd.DataFrame(anonymized_items)
            df.to_parquet(output_path, compression="snappy")

        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Anonymization failed")
        sys.exit(1)


@cli.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "parquet"]),
    default="json",
    help="Output format (default: json)",
)
@click.pass_context
def merge(
    ctx: click.Context,
    input_files: tuple[str, ...],
    output: str | None,
    format: str,
) -> None:
    """Merge multiple preprocessed data files.

    Example:
        politext-preprocess merge file1.json file2.json -o merged.json
    """
    config = ctx.obj["config"]

    if len(input_files) < 2:
        click.echo("Error: At least 2 input files required", err=True)
        sys.exit(1)

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.processed_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.processed_data_path / f"merged_{timestamp}.{format}")

    click.echo(f"Merging {len(input_files)} files...")

    try:
        all_items = []

        for input_file in input_files:
            click.echo(f"  Loading: {input_file}")

            if input_file.endswith(".parquet"):
                df = pd.read_parquet(input_file)
                items = df.to_dict("records")
            else:
                with open(input_file) as f:
                    data = json.load(f)
                items = data.get("items", data) if isinstance(data, dict) else data

            all_items.extend(items)
            click.echo(f"    Added {len(items)} items")

        # Deduplicate by ID if present
        seen_ids = set()
        unique_items = []
        for item in all_items:
            item_id = item.get("id", str(hash(json.dumps(item, sort_keys=True, default=str))))
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_items.append(item)

        duplicates_removed = len(all_items) - len(unique_items)
        click.echo(f"\nTotal items: {len(unique_items)} ({duplicates_removed} duplicates removed)")

        # Save output
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "source_files": list(input_files),
                            "total_items": len(unique_items),
                            "duplicates_removed": duplicates_removed,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "items": unique_items,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        elif format == "parquet":
            df = pd.DataFrame(unique_items)
            df.to_parquet(output_path, compression="snappy")

        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Merge failed")
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.pass_context
def stats(ctx: click.Context, input_file: str) -> None:
    """Show statistics about preprocessed data.

    Example:
        politext-preprocess stats data/processed/cleaned.json
    """
    click.echo(f"Analyzing: {input_file}")

    try:
        if input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
            items = df.to_dict("records")
        else:
            with open(input_file) as f:
                data = json.load(f)
            items = data.get("items", data) if isinstance(data, dict) else data

        click.echo(f"\n{'='*50}")
        click.echo(f"Total items: {len(items)}")

        if not items:
            return

        # Text length stats
        text_lengths = [len(item.get("text", "")) for item in items]
        click.echo(f"\nText length:")
        click.echo(f"  Min: {min(text_lengths)}")
        click.echo(f"  Max: {max(text_lengths)}")
        click.echo(f"  Avg: {sum(text_lengths) / len(text_lengths):.1f}")

        # Language distribution
        languages = [item.get("language", "unknown") for item in items]
        lang_counts = {}
        for lang in languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        click.echo(f"\nLanguage distribution:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:10]:
            click.echo(f"  {lang}: {count} ({count/len(items)*100:.1f}%)")

        # Political score distribution (if present)
        scores = [item.get("political_score") for item in items if "political_score" in item]
        if scores:
            click.echo(f"\nPolitical score:")
            click.echo(f"  Min: {min(scores):.3f}")
            click.echo(f"  Max: {max(scores):.3f}")
            click.echo(f"  Avg: {sum(scores) / len(scores):.3f}")

        # Keyword categories (if present)
        all_categories = []
        for item in items:
            all_categories.extend(item.get("keyword_categories", []))

        if all_categories:
            cat_counts = {}
            for cat in all_categories:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

            click.echo(f"\nKeyword categories:")
            for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
                click.echo(f"  {cat}: {count}")

        click.echo(f"{'='*50}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Stats failed")
        sys.exit(1)


def main() -> None:
    """Main entry point for the preprocess CLI."""
    cli()


if __name__ == "__main__":
    main()
