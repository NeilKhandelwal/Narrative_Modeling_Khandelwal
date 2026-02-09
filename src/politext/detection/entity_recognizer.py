"""
Named Entity Recognition for political content.

Uses spaCy for identifying political entities like politicians,
organizations, and geopolitical entities in text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


@dataclass
class PoliticalEntity:
    """A recognized political entity."""

    text: str
    label: str
    start: int
    end: int
    start_char: int
    end_char: int
    is_political: bool = False
    political_type: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int]:
        """Get character span."""
        return (self.start_char, self.end_char)


# Political entity types mapping
POLITICAL_ENTITY_TYPES = {
    "PERSON": ["politician", "official", "candidate"],
    "ORG": ["party", "government", "agency", "committee", "pac"],
    "GPE": ["country", "state", "city", "jurisdiction"],
    "EVENT": ["election", "summit", "hearing", "debate"],
    "LAW": ["bill", "act", "amendment", "regulation"],
    "NORP": ["political_group", "ideology"],  # Nationalities, religious, political groups
}

# Known political organizations (partial list for entity validation)
KNOWN_POLITICAL_ORGS = {
    "democratic party", "republican party", "gop", "dnc", "rnc",
    "congress", "senate", "house of representatives",
    "supreme court", "white house", "pentagon",
    "fbi", "cia", "nsa", "dhs", "doj",
    "fec", "ftc", "sec", "epa", "fda",
}

# Political title patterns
POLITICAL_TITLES = {
    "president", "vice president", "senator", "representative",
    "congressman", "congresswoman", "governor", "mayor",
    "secretary", "attorney general", "speaker", "leader",
    "justice", "judge", "ambassador", "commissioner",
}


class EntityRecognizer:
    """spaCy-based named entity recognizer for political content.

    Wraps spaCy's NER pipeline and adds political entity classification.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        entity_types: list[str] | None = None,
        custom_patterns: list[dict[str, Any]] | None = None,
    ):
        """Initialize entity recognizer.

        Args:
            model_name: spaCy model to use.
            entity_types: Entity types to extract. Defaults to political types.
            custom_patterns: Custom entity ruler patterns.
        """
        self.model_name = model_name
        self.entity_types = entity_types or list(POLITICAL_ENTITY_TYPES.keys())
        self._nlp: Language | None = None
        self._custom_patterns = custom_patterns or []

        self._load_model()

    def _load_model(self) -> None:
        """Load spaCy model and configure pipeline."""
        try:
            self._nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(f"Model {self.model_name} not found. Downloading...")
            spacy.cli.download(self.model_name)
            self._nlp = spacy.load(self.model_name)

        # Add entity ruler for custom patterns
        if self._custom_patterns:
            self._add_entity_ruler()

    def _add_entity_ruler(self) -> None:
        """Add entity ruler with custom patterns."""
        if not self._nlp:
            return

        # Add ruler before NER
        if "entity_ruler" not in self._nlp.pipe_names:
            ruler = self._nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(self._custom_patterns)
            logger.info(f"Added {len(self._custom_patterns)} custom entity patterns")

    def add_patterns(self, patterns: list[dict[str, Any]]) -> None:
        """Add custom entity patterns.

        Args:
            patterns: List of entity ruler patterns.
                Format: [{"label": "PERSON", "pattern": "John Doe"}]
        """
        self._custom_patterns.extend(patterns)
        if self._nlp and "entity_ruler" in self._nlp.pipe_names:
            ruler = self._nlp.get_pipe("entity_ruler")
            ruler.add_patterns(patterns)
        elif self._nlp:
            self._add_entity_ruler()

    def _is_political_entity(self, ent: Span) -> tuple[bool, str | None]:
        """Determine if an entity is political.

        Args:
            ent: spaCy entity span.

        Returns:
            Tuple of (is_political, political_type).
        """
        text_lower = ent.text.lower()
        label = ent.label_

        # Check organization against known political orgs
        if label == "ORG":
            if text_lower in KNOWN_POLITICAL_ORGS:
                return True, "government_org"
            # Check for political keywords
            if any(kw in text_lower for kw in ["party", "committee", "commission", "congress"]):
                return True, "political_org"

        # Check for political titles in PERSON entities
        if label == "PERSON":
            # Check surrounding context for political titles
            doc = ent.doc
            start_idx = max(0, ent.start - 3)
            context = doc[start_idx:ent.start].text.lower()
            if any(title in context for title in POLITICAL_TITLES):
                return True, "politician"

        # GPE (countries, states) are generally political
        if label == "GPE":
            return True, "jurisdiction"

        # LAW entities are political
        if label == "LAW":
            return True, "legislation"

        # NORP (nationalities, religious, political groups)
        if label == "NORP":
            return True, "political_group"

        # EVENT - check for political keywords
        if label == "EVENT":
            if any(kw in text_lower for kw in ["election", "debate", "summit", "hearing"]):
                return True, "political_event"

        return False, None

    def recognize(self, text: str) -> list[PoliticalEntity]:
        """Recognize named entities in text.

        Args:
            text: Text to analyze.

        Returns:
            List of PoliticalEntity objects.
        """
        if not self._nlp:
            raise RuntimeError("spaCy model not loaded")

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            # Filter by entity type
            if ent.label_ not in self.entity_types:
                continue

            # Check if political
            is_political, political_type = self._is_political_entity(ent)

            entities.append(PoliticalEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start,
                end=ent.end,
                start_char=ent.start_char,
                end_char=ent.end_char,
                is_political=is_political,
                political_type=political_type,
            ))

        return entities

    def recognize_political(self, text: str) -> list[PoliticalEntity]:
        """Recognize only political entities in text.

        Args:
            text: Text to analyze.

        Returns:
            List of political PoliticalEntity objects.
        """
        return [ent for ent in self.recognize(text) if ent.is_political]

    def recognize_batch(
        self,
        texts: list[str],
        n_process: int = 1,
        batch_size: int = 100,
    ) -> list[list[PoliticalEntity]]:
        """Recognize entities in multiple texts efficiently.

        Args:
            texts: List of texts to analyze.
            n_process: Number of processes for parallel processing.
            batch_size: Batch size for processing.

        Returns:
            List of entity lists, one per input text.
        """
        if not self._nlp:
            raise RuntimeError("spaCy model not loaded")

        results = []
        for doc in self._nlp.pipe(texts, n_process=n_process, batch_size=batch_size):
            entities = []
            for ent in doc.ents:
                if ent.label_ not in self.entity_types:
                    continue

                is_political, political_type = self._is_political_entity(ent)
                entities.append(PoliticalEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start,
                    end=ent.end,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    is_political=is_political,
                    political_type=political_type,
                ))
            results.append(entities)

        return results

    def extract_entity_summary(self, text: str) -> dict[str, Any]:
        """Extract summary of entities in text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with entity summary.
        """
        entities = self.recognize(text)
        political_entities = [e for e in entities if e.is_political]

        # Group by label
        by_label: dict[str, list[str]] = {}
        for ent in entities:
            if ent.label not in by_label:
                by_label[ent.label] = []
            if ent.text not in by_label[ent.label]:
                by_label[ent.label].append(ent.text)

        # Group political by type
        political_by_type: dict[str, list[str]] = {}
        for ent in political_entities:
            ptype = ent.political_type or "unknown"
            if ptype not in political_by_type:
                political_by_type[ptype] = []
            if ent.text not in political_by_type[ptype]:
                political_by_type[ptype].append(ent.text)

        return {
            "total_entities": len(entities),
            "political_entities": len(political_entities),
            "by_label": by_label,
            "political_by_type": political_by_type,
            "unique_entities": list(set(e.text for e in entities)),
            "unique_political": list(set(e.text for e in political_entities)),
        }

    def get_doc(self, text: str) -> Doc:
        """Get the full spaCy Doc for advanced analysis.

        Args:
            text: Text to analyze.

        Returns:
            spaCy Doc object.
        """
        if not self._nlp:
            raise RuntimeError("spaCy model not loaded")
        return self._nlp(text)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model info.
        """
        if not self._nlp:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self.model_name,
            "pipeline": self._nlp.pipe_names,
            "entity_types": self.entity_types,
            "custom_patterns": len(self._custom_patterns),
        }


def create_political_recognizer(
    model_name: str = "en_core_web_sm",
    politicians: list[str] | None = None,
) -> EntityRecognizer:
    """Create an entity recognizer configured for political content.

    Args:
        model_name: spaCy model to use.
        politicians: List of politician names to add as patterns.

    Returns:
        Configured EntityRecognizer.
    """
    patterns = []

    # Add politician patterns if provided
    if politicians:
        for name in politicians:
            patterns.append({"label": "PERSON", "pattern": name})

    recognizer = EntityRecognizer(
        model_name=model_name,
        entity_types=["PERSON", "ORG", "GPE", "EVENT", "LAW", "NORP"],
        custom_patterns=patterns,
    )

    return recognizer
