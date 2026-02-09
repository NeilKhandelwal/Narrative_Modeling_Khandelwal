"""
Data collection CLI for politext.

Provides commands for collecting political text data from various sources
including Twitter and FEC campaign finance data.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
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
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """Politext data collection tool.

    Collect political text data from Twitter and FEC campaign finance APIs.
    """
    ctx.ensure_object(dict)

    # Load configuration
    ctx.obj["config"] = load_config(config_path=config) if config else load_config()

    # Set logging level
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
@click.option(
    "--query",
    "-q",
    required=True,
    help="Search query string",
)
@click.option(
    "--max-results",
    "-n",
    default=1000,
    help="Maximum tweets to collect (default: 1000)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: data/raw/tweets_<timestamp>.json)",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["api", "scraper"]),
    default="api",
    help="Collection method: api (official) or scraper (snscrape)",
)
@click.option(
    "--since",
    type=str,
    help="Start date (YYYY-MM-DD) for tweet collection",
)
@click.option(
    "--until",
    type=str,
    help="End date (YYYY-MM-DD) for tweet collection",
)
@click.option(
    "--language",
    "-l",
    default="en",
    help="Filter by language code (default: en)",
)
@click.pass_context
def twitter(
    ctx: click.Context,
    query: str,
    max_results: int,
    output: str | None,
    method: str,
    since: str | None,
    until: str | None,
    language: str,
) -> None:
    """Collect tweets matching a search query.

    Example:
        politext-collect twitter -q "election 2024" -n 500 -o tweets.json
    """
    config = ctx.obj["config"]

    # Set default output path
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.raw_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.raw_data_path / f"tweets_{timestamp}.json")

    click.echo(f"Collecting tweets with query: {query}")
    click.echo(f"Method: {method}")
    click.echo(f"Max results: {max_results}")

    try:
        if method == "api":
            from politext.collectors import TwitterAPICollector

            collector = TwitterAPICollector(
                bearer_token=config.twitter.bearer_token,
                api_tier=config.twitter.api_tier,
                checkpoint_dir=config.storage.raw_data_path / "checkpoints",
            )

            # Validate credentials
            if not collector.validate_credentials():
                click.echo("Error: Invalid Twitter API credentials", err=True)
                sys.exit(1)

            result = collector.collect(
                query=query,
                max_results=max_results,
            )

        else:  # scraper
            from politext.collectors import TwitterScraperCollector

            collector = TwitterScraperCollector(
                checkpoint_dir=config.storage.raw_data_path / "checkpoints",
            )

            if not collector.validate_credentials():
                click.echo(
                    "Error: snscrape not installed. Install with: pip install snscrape",
                    err=True,
                )
                sys.exit(1)

            result = collector.collect(
                query=query,
                max_results=max_results,
                since=since,
                until=until,
                language=language,
            )

        # Check for errors
        if result.errors:
            for error in result.errors:
                click.echo(f"Warning: {error}", err=True)

        # Save results
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": result.to_dict(),
            "items": result.items,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        click.echo(f"\nCollected {result.total_collected} tweets")
        click.echo(f"Duration: {result.duration_seconds:.1f} seconds")
        click.echo(f"Output saved to: {output_path}")

        if result.has_more:
            click.echo("\nNote: More results available. Increase --max-results to collect more.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Twitter collection failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--username",
    "-u",
    required=True,
    help="Twitter username (without @)",
)
@click.option(
    "--max-results",
    "-n",
    default=500,
    help="Maximum tweets to collect (default: 500)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--include-replies",
    is_flag=True,
    help="Include reply tweets",
)
@click.option(
    "--include-retweets",
    is_flag=True,
    default=True,
    help="Include retweets (default: True)",
)
@click.pass_context
def user_tweets(
    ctx: click.Context,
    username: str,
    max_results: int,
    output: str | None,
    include_replies: bool,
    include_retweets: bool,
) -> None:
    """Collect tweets from a specific Twitter user.

    Example:
        politext-collect user-tweets -u potus -n 200
    """
    config = ctx.obj["config"]

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.raw_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.raw_data_path / f"user_{username}_{timestamp}.json")

    click.echo(f"Collecting tweets from user: @{username}")

    try:
        from politext.collectors import TwitterScraperCollector

        collector = TwitterScraperCollector(
            checkpoint_dir=config.storage.raw_data_path / "checkpoints",
        )

        if not collector.validate_credentials():
            click.echo(
                "Error: snscrape not installed. Install with: pip install snscrape",
                err=True,
            )
            sys.exit(1)

        result = collector.collect_user_tweets(
            username=username,
            max_results=max_results,
            include_replies=include_replies,
            include_retweets=include_retweets,
        )

        if result.errors:
            for error in result.errors:
                click.echo(f"Warning: {error}", err=True)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": result.to_dict(),
            "items": result.items,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        click.echo(f"\nCollected {result.total_collected} tweets from @{username}")
        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("User tweet collection failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--query",
    "-q",
    required=True,
    help="Search query (candidate/committee name)",
)
@click.option(
    "--query-type",
    "-t",
    type=click.Choice(["candidates", "committees", "contributions"]),
    default="candidates",
    help="Type of FEC data to collect",
)
@click.option(
    "--max-results",
    "-n",
    default=500,
    help="Maximum results to collect (default: 500)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--election-year",
    "-y",
    type=int,
    help="Filter by election year",
)
@click.option(
    "--party",
    "-p",
    type=str,
    help="Filter by party code (DEM, REP, etc.)",
)
@click.option(
    "--state",
    "-s",
    type=str,
    help="Filter by state code",
)
@click.pass_context
def fec(
    ctx: click.Context,
    query: str,
    query_type: str,
    max_results: int,
    output: str | None,
    election_year: int | None,
    party: str | None,
    state: str | None,
) -> None:
    """Collect FEC campaign finance data.

    Example:
        politext-collect fec -q "Biden" -t candidates -y 2024
        politext-collect fec -q "ActBlue" -t committees
    """
    config = ctx.obj["config"]

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.raw_data_path.mkdir(parents=True, exist_ok=True)
        output = str(config.storage.raw_data_path / f"fec_{query_type}_{timestamp}.json")

    click.echo(f"Collecting FEC {query_type} data for: {query}")

    try:
        from politext.collectors import FECAPICollector

        collector = FECAPICollector(
            api_key=config.fec.api_key,
            rate_limit_requests=config.fec.rate_limit_requests,
            checkpoint_dir=config.storage.raw_data_path / "checkpoints",
        )

        if not collector.validate_credentials():
            click.echo("Error: FEC API validation failed", err=True)
            sys.exit(1)

        kwargs = {}
        if election_year:
            kwargs["election_year"] = election_year
        if party:
            kwargs["party"] = party
        if state:
            kwargs["state"] = state

        result = collector.collect(
            query=query,
            max_results=max_results,
            query_type=query_type,
            **kwargs,
        )

        if result.errors:
            for error in result.errors:
                click.echo(f"Warning: {error}", err=True)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": result.to_dict(),
            "items": result.items,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        click.echo(f"\nCollected {result.total_collected} {query_type}")
        click.echo(f"Duration: {result.duration_seconds:.1f} seconds")
        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("FEC collection failed")
        sys.exit(1)


@cli.command()
@click.option(
    "--committee-id",
    "-c",
    required=True,
    help="FEC committee ID (e.g., C00703975)",
)
@click.option(
    "--max-results",
    "-n",
    default=1000,
    help="Maximum contributions to collect (default: 1000)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--min-amount",
    type=float,
    help="Minimum contribution amount",
)
@click.option(
    "--max-amount",
    type=float,
    help="Maximum contribution amount",
)
@click.pass_context
def contributions(
    ctx: click.Context,
    committee_id: str,
    max_results: int,
    output: str | None,
    min_amount: float | None,
    max_amount: float | None,
) -> None:
    """Collect contributions to a specific FEC committee.

    Example:
        politext-collect contributions -c C00703975 -n 500 --min-amount 1000
    """
    config = ctx.obj["config"]

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.storage.raw_data_path.mkdir(parents=True, exist_ok=True)
        output = str(
            config.storage.raw_data_path / f"contributions_{committee_id}_{timestamp}.json"
        )

    click.echo(f"Collecting contributions to committee: {committee_id}")

    try:
        from politext.collectors import FECAPICollector

        collector = FECAPICollector(
            api_key=config.fec.api_key,
            rate_limit_requests=config.fec.rate_limit_requests,
            checkpoint_dir=config.storage.raw_data_path / "checkpoints",
        )

        result = collector.collect_contributions(
            committee_id=committee_id,
            max_results=max_results,
            min_amount=min_amount,
            max_amount=max_amount,
        )

        if result.errors:
            for error in result.errors:
                click.echo(f"Warning: {error}", err=True)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": result.to_dict(),
            "items": result.items,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        click.echo(f"\nCollected {result.total_collected} contributions")
        click.echo(f"Duration: {result.duration_seconds:.1f} seconds")
        click.echo(f"Output saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Contributions collection failed")
        sys.exit(1)


@cli.command()
@click.argument("keywords_file", type=click.Path(exists=True))
@click.option(
    "--max-per-keyword",
    "-n",
    default=100,
    help="Maximum tweets per keyword (default: 100)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for collected data",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["api", "scraper"]),
    default="scraper",
    help="Collection method (default: scraper)",
)
@click.pass_context
def batch(
    ctx: click.Context,
    keywords_file: str,
    max_per_keyword: int,
    output_dir: str | None,
    method: str,
) -> None:
    """Batch collect tweets using keywords from a JSON file.

    The keywords file should be a JSON file with keyword categories.

    Example:
        politext-collect batch configs/keywords/topics.json -n 50
    """
    config = ctx.obj["config"]

    if output_dir is None:
        output_dir = str(config.storage.raw_data_path / "batch")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading keywords from: {keywords_file}")

    try:
        with open(keywords_file) as f:
            keywords_data = json.load(f)

        # Extract all keywords from categories
        all_keywords = []
        for category in keywords_data.get("categories", []):
            for subcat in category.get("subcategories", []):
                all_keywords.extend(subcat.get("keywords", []))

        if not all_keywords:
            # Try flat keyword list
            all_keywords = keywords_data.get("keywords", [])

        if not all_keywords:
            click.echo("Error: No keywords found in file", err=True)
            sys.exit(1)

        click.echo(f"Found {len(all_keywords)} keywords")

        # Initialize collector
        if method == "scraper":
            from politext.collectors import TwitterScraperCollector

            collector = TwitterScraperCollector(
                checkpoint_dir=output_path / "checkpoints",
            )
        else:
            from politext.collectors import TwitterAPICollector

            collector = TwitterAPICollector(
                bearer_token=config.twitter.bearer_token,
                checkpoint_dir=output_path / "checkpoints",
            )

        if not collector.validate_credentials():
            click.echo("Error: Collector validation failed", err=True)
            sys.exit(1)

        # Collect for each keyword
        all_results = []
        with tqdm(all_keywords, desc="Collecting") as pbar:
            for keyword in pbar:
                pbar.set_postfix(keyword=keyword[:20])

                result = collector.collect(
                    query=keyword,
                    max_results=max_per_keyword,
                )

                all_results.extend(result.items)

                if result.errors:
                    for error in result.errors:
                        logger.warning(f"Error for '{keyword}': {error}")

        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_output = output_path / f"batch_collection_{timestamp}.json"

        with open(combined_output, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "keywords_file": keywords_file,
                        "total_keywords": len(all_keywords),
                        "max_per_keyword": max_per_keyword,
                        "method": method,
                        "timestamp": timestamp,
                    },
                    "items": all_results,
                },
                f,
                indent=2,
                default=str,
            )

        click.echo(f"\nBatch collection complete!")
        click.echo(f"Total tweets collected: {len(all_results)}")
        click.echo(f"Output saved to: {combined_output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Batch collection failed")
        sys.exit(1)


def main() -> None:
    """Main entry point for the collect CLI."""
    cli()


if __name__ == "__main__":
    main()
