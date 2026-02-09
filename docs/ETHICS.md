# Ethical Guidelines for Political Text Data Collection and Analysis

This document outlines the ethical considerations, best practices, and compliance requirements for using the politext library in academic research involving political text data.

## Overview

Political text data research carries unique ethical responsibilities due to:
- The sensitive nature of political opinions and affiliations
- Privacy concerns surrounding social media users
- Potential for misuse in voter manipulation or surveillance
- Power imbalances between researchers and subjects
- The potential impact on democratic discourse

## Core Principles

### 1. Informed Consent and Public Data

**Principle**: Respect user expectations about how their data will be used.

**Guidelines**:
- Only collect publicly available data (public tweets, public posts)
- Do not attempt to circumvent privacy settings
- Recognize that "public" does not mean "consented to research"
- Consider whether users would reasonably expect their data to be used for research

**Best Practices**:
```python
# Good: Collect only public political discourse
collector.collect("election policy debate", max_results=1000)

# Avoid: Targeting specific individuals without consent
# collector.collect_user_tweets("@private_citizen", ...)
```

### 2. Anonymization and De-identification

**Principle**: Protect individual privacy through robust anonymization.

**Requirements**:
- Always hash or remove usernames before analysis
- Remove or redact personally identifiable information (PII)
- Use consistent salting for hash functions within a study
- Never publish data that could identify individuals

**Implementation**:
```python
from politext.ethics.anonymizer import Anonymizer

# Configure anonymizer with a secure salt
anonymizer = Anonymizer(
    hash_algorithm="sha256",
    hash_salt="your_secure_salt",  # Store securely, never in code
    remove_pii=True,
    use_presidio=True,  # Enhanced PII detection
)

# Anonymize before any analysis
anonymized_data = anonymizer.anonymize_batch(collected_tweets)
```

**PII Types Detected and Removed**:
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses
- Personal names (with Presidio)
- Location data (with Presidio)

### 3. Purpose Limitation

**Principle**: Use data only for stated research purposes.

**Guidelines**:
- Clearly define research questions before data collection
- Collect only the minimum data necessary
- Do not repurpose data for commercial use
- Document the intended use in your IRB application

**Configuration**:
```yaml
# configs/config.yaml
ethics:
  anonymize_usernames: true
  remove_pii: true
  data_retention_days: 365  # Delete after study completion
  require_consent: false    # For public data studies
```

### 4. Data Minimization

**Principle**: Collect and retain only what is necessary.

**Guidelines**:
- Define specific keywords and time periods
- Set appropriate result limits
- Delete raw data after anonymization
- Retain only aggregated results when possible

**Example**:
```python
# Good: Targeted collection with limits
result = collector.collect(
    query="2024 election policy",
    max_results=5000,  # Only what's needed
    since="2024-01-01",
    until="2024-03-31",
)

# After processing, delete raw data
shutil.rmtree("data/raw")
```

### 5. Transparency and Reproducibility

**Principle**: Be open about methods and enable verification.

**Requirements**:
- Document all collection parameters
- Preserve collection metadata
- Make analysis code available (when possible)
- Report limitations and potential biases

**Metadata Preservation**:
```python
# Output files automatically include metadata
{
    "metadata": {
        "collection_date": "2024-01-15",
        "query": "election 2024",
        "total_collected": 5000,
        "method": "twitter_api_v2",
        "filters_applied": ["language:en", "is_retweet:false"]
    },
    "items": [...]
}
```

## Institutional Review Board (IRB) Considerations

### Preparing Your IRB Application

When using politext for research, include the following in your IRB application:

1. **Data Sources**: Specify which APIs and collection methods you'll use
2. **Anonymization Procedures**: Detail how PII will be protected
3. **Data Security**: Describe storage and access controls
4. **Retention Period**: Specify how long data will be kept
5. **Destruction Procedures**: How data will be securely deleted

### Sample IRB Language

> This study will collect publicly posted political content from Twitter/X
> using the politext library. All data will be anonymized using SHA-256
> hashing with secure salting before any analysis. Personally identifiable
> information including usernames, email addresses, and phone numbers will
> be automatically detected and removed. Raw data will be deleted within
> 30 days of collection, and only anonymized data will be retained for
> analysis. The anonymized dataset will be destroyed within 1 year of
> study completion.

### Common IRB Concerns and Responses

| Concern | Response |
|---------|----------|
| "Users didn't consent" | Data is public and fully anonymized. No individual can be identified. |
| "Political opinions are sensitive" | Analysis is aggregate-level only. Individual opinions are not reported. |
| "Data could be misused" | Strict access controls, secure storage, and deletion after study completion. |

## Platform-Specific Guidelines

### Twitter/X

1. **Terms of Service Compliance**:
   - Comply with Twitter's Developer Agreement
   - Do not exceed rate limits
   - Do not sell or redistribute raw data

2. **Academic Research Access**:
   - Apply for Academic Research access tier for historical data
   - Document academic affiliation and research purpose

3. **Collection Practices**:
   - Use official API when possible
   - If using alternative methods, implement ethical rate limiting
   - Respect robots.txt guidelines

### FEC Data

1. **Public Record Status**:
   - FEC data is public by law
   - Still practice data minimization
   - Be careful with contributor personal information

2. **Contributor Privacy**:
   - Aggregate contribution data when possible
   - Avoid publishing individual donor names
   - Consider the difference between public figures and private citizens

## Bias in Political Text Research

### Recognizing and Addressing Bias

1. **Selection Bias**:
   - Twitter users are not representative of the population
   - Politically active users are overrepresented
   - Document these limitations

2. **Keyword Bias**:
   - Your search terms affect what you find
   - Use balanced keywords across political spectrum
   - Document keyword selection rationale

3. **Temporal Bias**:
   - Political discourse varies by news cycles
   - Collect over extended periods when possible
   - Document collection timeframes

### Using the Balance Tool

```bash
# Create politically balanced dataset
politext-balance political data.json \
    -l "Democrat" -l "liberal" -l "progressive" \
    -r "Republican" -r "conservative" -r "MAGA" \
    -n 1000 --include-neutral
```

## Data Security Requirements

### Storage

- Use encrypted storage for all collected data
- Implement access controls (who can access what)
- Keep secure backups with same protections

### Transmission

- Use HTTPS for all API calls (automatic)
- Encrypt data transfers between systems
- Use secure file sharing for collaboration

### Retention and Deletion

```python
# Configure retention period
# configs/config.yaml
ethics:
  data_retention_days: 365

# Secure deletion after study
import shutil
shutil.rmtree("data/raw", ignore_errors=True)
shutil.rmtree("data/processed", ignore_errors=True)
```

## Reporting Requirements

When publishing research using politext:

1. **Methodology Section**:
   - Describe data sources and collection methods
   - Report sample sizes and date ranges
   - Document preprocessing steps

2. **Limitations Section**:
   - Acknowledge platform biases
   - Note anonymization may affect some analyses
   - Discuss generalizability limits

3. **Ethics Statement**:
   - State IRB approval or exemption
   - Describe anonymization procedures
   - Note data availability (or why not available)

### Sample Ethics Statement

> This research was approved by [Institution] IRB (Protocol #XXXX).
> All Twitter data was collected from public posts and anonymized using
> cryptographic hashing before analysis. No personally identifiable
> information was retained. The anonymized dataset is not publicly
> available due to platform terms of service, but analysis code is
> available at [repository].

## Checklist Before Data Collection

- [ ] IRB approval or exemption obtained
- [ ] Research questions clearly defined
- [ ] Minimum necessary data scope determined
- [ ] Anonymization procedures configured
- [ ] Secure storage prepared
- [ ] Retention period defined
- [ ] Deletion procedures planned
- [ ] Keywords balanced across political spectrum
- [ ] Rate limiting configured appropriately
- [ ] Platform terms of service reviewed

## Checklist Before Publication

- [ ] All data fully anonymized
- [ ] No individual identifiable from results
- [ ] Methodology documented
- [ ] Limitations acknowledged
- [ ] Ethics statement included
- [ ] Code availability addressed
- [ ] Raw data securely deleted

## Resources

- [Association of Internet Researchers Ethics Guidelines](https://aoir.org/ethics/)
- [Twitter Academic Research Access](https://developer.twitter.com/en/products/twitter-api/academic-research)
- [FEC Data Availability](https://www.fec.gov/data/)
- [GDPR Considerations for Research](https://ec.europa.eu/research/participants/data/ref/h2020/grants_manual/hi/ethics/h2020_hi_ethics-data-protection_en.pdf)

## Contact

For questions about ethical use of this library, please open an issue on GitHub or contact the maintainers.

---

*This document is intended as guidance and does not constitute legal advice. Researchers should consult with their institutional ethics boards and legal counsel for specific situations.*
