# politext

**Political Text Data Collection System for Sentiment Bias Research**

A modular Python package for collecting, processing, and organizing political text data for academic sentiment bias research. Designed for measuring and contextualizing political bias in sentiment models for election narrative modeling.

## Features

- **Multi-source Data Collection**
  - Twitter/X API (Official v2 API with Basic, Pro, and Academic tiers)
  - Twitter/X Scraping (Alternative method using snscrape)
  - FEC Campaign Finance Data (Candidates, committees, contributions)

- **Text Preprocessing**
  - URL, mention, and hashtag normalization
  - Language detection and filtering
  - Unicode normalization and HTML entity handling
  - Configurable cleaning pipelines

- **Political Content Detection**
  - Keyword matching with Aho-Corasick algorithm
  - Named entity recognition for political figures
  - Political relevance scoring
  - Pre-built keyword dictionaries (politicians, parties, topics)

- **Privacy and Ethics**
  - Automatic PII detection and removal
  - Username hashing and anonymization
  - Presidio integration for advanced PII detection
  - Comprehensive ethical guidelines

- **Dataset Management**
  - Class balancing (undersample/oversample)
  - Political orientation balancing
  - Train/val/test splitting with stratification
  - Parquet and JSON export formats

## Installation

```bash
# Clone the repository
git clone https://github.com/neilkhandelwal/politext.git
cd politext

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt

# Download spaCy model (required for NER)
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Configuration

Copy the example credentials file and add your API keys:

```bash
cp configs/credentials.yaml.example configs/credentials.yaml
```

Edit `configs/credentials.yaml`:

```yaml
twitter:
  bearer_token: "your_twitter_bearer_token"
  api_tier: "basic"  # basic, pro, or academic

fec:
  api_key: "your_fec_api_key"  # or use DEMO_KEY

ethics:
  hash_salt: "your_secure_random_salt"
```

### 2. Collect Data

#### Twitter Data (Official API)

```bash
# Search for election-related tweets
politext-collect twitter -q "election 2024 policy" -n 1000 -o data/raw/election_tweets.json

# Collect from a specific user
politext-collect user-tweets -u potus -n 500
```

#### Twitter Data (Scraper - No API Required)

```bash
# Use snscrape for collection without API
politext-collect twitter -q "climate change policy" -n 500 -m scraper --since 2024-01-01
```

#### FEC Campaign Finance Data

```bash
# Search for candidates
politext-collect fec -q "Biden" -t candidates -y 2024

# Get committee contributions
politext-collect contributions -c C00703975 -n 1000 --min-amount 1000
```

#### Batch Collection with Keywords

```bash
# Collect using keyword file
politext-collect batch configs/keywords/topics.json -n 100 --method scraper
```

### 3. Preprocess Data

```bash
# Clean and normalize text
politext-preprocess clean data/raw/tweets.json -o data/processed/cleaned.json -l en

# Detect political content
politext-preprocess detect data/processed/cleaned.json -o data/annotated/detected.json --min-score 0.3

# Anonymize for privacy
politext-preprocess anonymize data/processed/cleaned.json --remove-pii

# View statistics
politext-preprocess stats data/processed/cleaned.json
```

### 4. Create Balanced Datasets

```bash
# Balance by keyword categories
politext-balance by-class data.json -b keyword_categories -n 500

# Create politically balanced dataset
politext-balance political data.json \
    -l "Democrat" -l "liberal" -l "Biden" \
    -r "Republican" -r "conservative" -r "Trump" \
    -n 500

# Split into train/val/test
politext-balance split data.json --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

## Python API Usage

### Collecting Data

```python
from politext.collectors import TwitterAPICollector, FECAPICollector
from politext.config import load_config

# Load configuration
config = load_config()

# Twitter collection
twitter = TwitterAPICollector(
    bearer_token=config.twitter.bearer_token,
    api_tier=config.twitter.api_tier,
)

result = twitter.collect(
    query="climate policy",
    max_results=500,
)

print(f"Collected {result.total_collected} tweets")

# FEC collection
fec = FECAPICollector(api_key=config.fec.api_key)

candidates = fec.collect(
    query="Smith",
    query_type="candidates",
    election_year=2024,
)
```

### Preprocessing

```python
from politext.preprocessing import TextCleaner
from politext.detection import KeywordMatcher

# Clean text
cleaner = TextCleaner(
    remove_urls=True,
    normalize_hashtags=True,
    detect_language=True,
)

result = cleaner.clean("Check out https://example.com #Election2024 @user")
print(result.cleaned)  # "Check out Election2024 @user"
print(result.language)  # "en"

# Detect political content
matcher = KeywordMatcher()
matcher.load_keywords("configs/keywords/topics.json")

matches = matcher.find_matches("The Senate passed the new healthcare bill")
score = matcher.calculate_political_score("The Senate passed the new healthcare bill")
```

### Anonymization

```python
from politext.ethics.anonymizer import Anonymizer

anonymizer = Anonymizer(
    hash_salt="your_secure_salt",
    remove_pii=True,
    use_presidio=True,
)

# Anonymize text
result = anonymizer.anonymize_text("Contact john@email.com or call 555-123-4567")
print(result.anonymized)  # "Contact [REDACTED]_EMAIL or call [REDACTED]_PHONE"

# Anonymize full tweet
anonymized_tweet = anonymizer.anonymize_tweet(tweet_dict)
```

### Manual Annotation

```python
from politext.preprocessing import Annotator, PoliticalLeaning, Sentiment

annotator = Annotator(storage_path="annotations.json")

# Add annotation
annotator.annotate(
    item_id="tweet_123",
    annotator_id="annotator_1",
    political_leaning=PoliticalLeaning.LEFT,
    sentiment=Sentiment.NEGATIVE,
    is_political=True,
    confidence=0.9,
)

# Get gold labels
gold = annotator.get_gold_label("tweet_123", method="majority")

# Calculate inter-annotator agreement
agreement = annotator.calculate_agreement()
```

## Project Structure

```
politext/
├── configs/
│   ├── config.yaml              # Main configuration
│   ├── credentials.yaml.example # API credentials template
│   └── keywords/
│       ├── politicians.json     # Political figures (50 keywords)
│       ├── parties.json         # Parties/ideologies (42 keywords)
│       └── topics.json          # Political topics (128 keywords)
├── src/politext/
│   ├── cli/                     # Command-line interfaces
│   │   ├── collect.py           # Data collection CLI
│   │   ├── preprocess.py        # Preprocessing CLI
│   │   └── balance.py           # Dataset balancing CLI
│   ├── collectors/              # Data collectors
│   │   ├── twitter_api.py       # Official Twitter API
│   │   ├── twitter_scraper.py   # snscrape-based scraper
│   │   └── fec_api.py           # FEC campaign finance
│   ├── preprocessing/           # Text preprocessing
│   │   ├── cleaner.py           # Text cleaning
│   │   ├── tokenizer.py         # Tokenization
│   │   └── annotator.py         # Manual annotation
│   ├── detection/               # Political detection
│   │   ├── keyword_matcher.py   # Keyword matching
│   │   ├── entity_recognizer.py # Named entity recognition
│   │   └── political_classifier.py
│   ├── storage/                 # Data storage
│   │   ├── parquet_store.py     # Parquet format
│   │   └── sqlite_store.py      # SQLite database
│   └── ethics/                  # Privacy & ethics
│       ├── anonymizer.py        # PII removal
│       └── compliance.py        # Compliance utilities
├── docs/
│   └── ETHICS.md                # Ethical guidelines
├── tests/                       # Unit tests
├── pyproject.toml               # Package configuration
└── requirements.txt             # Dependencies
```

## Keyword Dictionaries

The package includes pre-built keyword dictionaries for political content detection:

### Politicians (50 keywords)
- Democrats: Biden, Harris, Pelosi, Schumer, AOC, etc.
- Republicans: Trump, DeSantis, McConnell, McCarthy, etc.

### Parties & Ideologies (42 keywords)
- Parties: Democrat, Republican, Libertarian, Green, etc.
- Ideologies: liberal, conservative, progressive, moderate, etc.

### Topics (128 keywords across 9 categories)
- Elections: vote, ballot, poll, election, primary, etc.
- Healthcare: medicare, medicaid, ACA, insurance, etc.
- Immigration: border, immigration, asylum, visa, etc.
- Economy: inflation, jobs, GDP, unemployment, etc.
- And more...

## Ethical Guidelines

This library is designed for academic research with privacy and ethics in mind. Please review the [Ethics Guidelines](docs/ETHICS.md) before use.

Key principles:
- Collect only publicly available data
- Always anonymize before analysis
- Minimize data collection and retention
- Document your methods transparently
- Obtain IRB approval when required

## Environment Variables

Configuration can be overridden with environment variables:

```bash
export POLITEXT_TWITTER_BEARER_TOKEN="your_token"
export POLITEXT_TWITTER_API_TIER="academic"
export POLITEXT_FEC_API_KEY="your_key"
export POLITEXT_ETHICS_HASH_SALT="your_salt"
export POLITEXT_LOG_LEVEL="DEBUG"
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/ --fix

# Type checking
mypy src/
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{politext2024,
  author = {Khandelwal, Neil},
  title = {politext: Political Text Data Collection for Sentiment Bias Research},
  year = {2024},
  url = {https://github.com/neilkhandelwal/politext}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Twitter API v2 access provided by Twitter Developer Platform
- FEC data provided by the Federal Election Commission
- Named entity recognition powered by spaCy
- PII detection enhanced by Microsoft Presidio
