"""
Tokenization utilities for political text.

Provides spaCy-based tokenization with political term handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


# Political terms to preserve (not treat as stop words)
POLITICAL_PRESERVE_TERMS = {
    "not", "no", "never", "none",  # Negation
    "against", "for", "with", "without",  # Stance indicators
    "but", "however", "although", "despite",  # Contrast
    "should", "must", "need", "will", "would",  # Modals
    "all", "every", "both", "either", "neither",  # Quantifiers
    "left", "right",  # Political spectrum
    "us", "we", "they", "them",  # Political "us vs them"
}


@dataclass
class TokenizationResult:
    """Result of tokenization."""

    original: str
    tokens: list[str]
    lemmas: list[str]
    pos_tags: list[str]
    filtered_tokens: list[str]  # After stop word removal
    sentences: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class Tokenizer:
    """spaCy-based tokenizer for political text.

    Provides tokenization, lemmatization, and POS tagging with
    special handling for political terms.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        remove_stop_words: bool = True,
        remove_punctuation: bool = True,
        lemmatize: bool = True,
        preserve_political_terms: bool = True,
        min_token_length: int = 2,
        lowercase: bool = True,
    ):
        """Initialize tokenizer.

        Args:
            model_name: spaCy model to use.
            remove_stop_words: Whether to remove stop words.
            remove_punctuation: Whether to remove punctuation tokens.
            lemmatize: Whether to return lemmas instead of tokens.
            preserve_political_terms: Whether to preserve political terms.
            min_token_length: Minimum token length.
            lowercase: Whether to lowercase tokens.
        """
        self.model_name = model_name
        self.remove_stop_words = remove_stop_words
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize
        self.preserve_political_terms = preserve_political_terms
        self.min_token_length = min_token_length
        self.lowercase = lowercase

        self._nlp: Language | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load spaCy model."""
        try:
            self._nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(f"Model {self.model_name} not found. Downloading...")
            spacy.cli.download(self.model_name)
            self._nlp = spacy.load(self.model_name)

        # Disable unnecessary pipeline components for speed
        disabled = []
        for pipe in ["ner", "parser"]:
            if pipe in self._nlp.pipe_names:
                disabled.append(pipe)

        if disabled:
            self._nlp.disable_pipes(*disabled)
            logger.debug(f"Disabled pipeline components: {disabled}")

    def _should_keep_token(self, token) -> bool:
        """Determine if a token should be kept.

        Args:
            token: spaCy token.

        Returns:
            True if token should be kept.
        """
        # Always remove whitespace
        if token.is_space:
            return False

        # Check punctuation
        if self.remove_punctuation and token.is_punct:
            return False

        # Check length
        if len(token.text) < self.min_token_length:
            return False

        # Get text for comparison
        text = token.text.lower()

        # Check political preservation
        if self.preserve_political_terms and text in POLITICAL_PRESERVE_TERMS:
            return True

        # Check stop words
        if self.remove_stop_words and token.is_stop:
            return False

        return True

    def tokenize(self, text: str) -> TokenizationResult:
        """Tokenize text.

        Args:
            text: Text to tokenize.

        Returns:
            TokenizationResult with tokens and metadata.
        """
        if not self._nlp:
            raise RuntimeError("spaCy model not loaded")

        doc = self._nlp(text)

        # Extract all tokens
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]

        # Filter tokens
        filtered = []
        for token in doc:
            if self._should_keep_token(token):
                if self.lemmatize:
                    text_out = token.lemma_
                else:
                    text_out = token.text

                if self.lowercase:
                    text_out = text_out.lower()

                filtered.append(text_out)

        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]

        return TokenizationResult(
            original=text,
            tokens=tokens,
            lemmas=lemmas,
            pos_tags=pos_tags,
            filtered_tokens=filtered,
            sentences=sentences,
            metadata={
                "token_count": len(tokens),
                "filtered_count": len(filtered),
                "sentence_count": len(sentences),
            },
        )

    def tokenize_to_list(self, text: str) -> list[str]:
        """Tokenize text and return filtered token list.

        Args:
            text: Text to tokenize.

        Returns:
            List of filtered tokens.
        """
        return self.tokenize(text).filtered_tokens

    def tokenize_batch(
        self,
        texts: list[str],
        n_process: int = 1,
        batch_size: int = 100,
    ) -> list[TokenizationResult]:
        """Tokenize multiple texts efficiently.

        Args:
            texts: List of texts to tokenize.
            n_process: Number of processes.
            batch_size: Batch size for processing.

        Returns:
            List of TokenizationResult objects.
        """
        if not self._nlp:
            raise RuntimeError("spaCy model not loaded")

        results = []
        for doc in self._nlp.pipe(texts, n_process=n_process, batch_size=batch_size):
            tokens = [token.text for token in doc]
            lemmas = [token.lemma_ for token in doc]
            pos_tags = [token.pos_ for token in doc]

            filtered = []
            for token in doc:
                if self._should_keep_token(token):
                    if self.lemmatize:
                        text_out = token.lemma_
                    else:
                        text_out = token.text

                    if self.lowercase:
                        text_out = text_out.lower()

                    filtered.append(text_out)

            sentences = [sent.text.strip() for sent in doc.sents]

            results.append(TokenizationResult(
                original=doc.text,
                tokens=tokens,
                lemmas=lemmas,
                pos_tags=pos_tags,
                filtered_tokens=filtered,
                sentences=sentences,
                metadata={
                    "token_count": len(tokens),
                    "filtered_count": len(filtered),
                    "sentence_count": len(sentences),
                },
            ))

        return results

    def get_ngrams(
        self,
        text: str,
        n: int = 2,
        use_filtered: bool = True,
    ) -> list[tuple[str, ...]]:
        """Extract n-grams from text.

        Args:
            text: Text to process.
            n: Size of n-grams.
            use_filtered: Whether to use filtered tokens.

        Returns:
            List of n-gram tuples.
        """
        result = self.tokenize(text)
        tokens = result.filtered_tokens if use_filtered else result.tokens

        if len(tokens) < n:
            return []

        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def get_doc(self, text: str) -> Doc:
        """Get full spaCy Doc for custom processing.

        Args:
            text: Text to process.

        Returns:
            spaCy Doc object.
        """
        if not self._nlp:
            raise RuntimeError("spaCy model not loaded")
        return self._nlp(text)

    def get_vocab_size(self) -> int:
        """Get vocabulary size of loaded model.

        Returns:
            Vocabulary size.
        """
        if not self._nlp:
            return 0
        return len(self._nlp.vocab)


def create_default_tokenizer() -> Tokenizer:
    """Create a default tokenizer for political text.

    Returns:
        Configured Tokenizer instance.
    """
    return Tokenizer(
        model_name="en_core_web_sm",
        remove_stop_words=True,
        remove_punctuation=True,
        lemmatize=True,
        preserve_political_terms=True,
        min_token_length=2,
        lowercase=True,
    )
