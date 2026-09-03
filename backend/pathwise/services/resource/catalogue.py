"""The curated resource catalogue.

The spec's rule is blunt: **do not ask an LLM to invent resource URLs.** Models
produce confident, plausible, dead links — a real-looking course URL on a real domain
that has never existed — and a learner who clicks three of those stops trusting
everything else the system says.

So resources come from hand-curated YAML, are HTTP-checked before they can be
recommended, and the model is only ever allowed to *rank and explain* rows that
already exist. The `known_resources` validator enforces that at the AI boundary.

This module owns loading and identity. Identity is the subtle part: the same page
reached three different ways must collapse to one row, or the catalogue silently
accumulates duplicates that then compete with each other in every ranking.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pathwise.api.errors import ValidationError
from pathwise.models.enums import ResourceType

RESOURCE_DATA_DIR: Final = Path(__file__).resolve().parents[2] / "data" / "resources"

#: Query parameters that identify a campaign rather than a document. Stripped before
#: hashing, so the same article shared from a newsletter and from search collapses to
#: one row instead of competing with itself in every ranking.
_TRACKING_PARAMS: Final = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "ref",
        "referrer",
        "source",
    }
)

#: Publishers whose material starts with a higher quality prior. Not a judgement that
#: everything they publish is good — it is that an official language reference is a
#: safer default recommendation than an unknown blog, before anyone has rated either.
PUBLISHER_PRIORS: Final[dict[str, float]] = {
    "docs.python.org": 0.95,
    "numpy.org": 0.92,
    "pandas.pydata.org": 0.90,
    "pytorch.org": 0.92,
    "scikit-learn.org": 0.92,
    "developer.mozilla.org": 0.93,
    "postgresql.org": 0.92,
    "git-scm.com": 0.90,
    "ocw.mit.edu": 0.90,
    "cs231n.github.io": 0.88,
    "www.3blue1brown.com": 0.90,
    "course.fast.ai": 0.88,
    "d2l.ai": 0.87,
    "arxiv.org": 0.80,
    "distill.pub": 0.90,
    "www.deeplearningbook.org": 0.88,
    "spinningup.openai.com": 0.85,
    "realpython.com": 0.82,
    "www.youtube.com": 0.70,
    "en.wikipedia.org": 0.65,
}

DEFAULT_QUALITY_PRIOR: Final = 0.55


def canonical_url(url: str) -> str:
    """Reduce a URL to a stable identity for deduplication.

    Normalises scheme and host case, drops a default port, strips tracking
    parameters, sorts what remains, and removes a trailing slash and fragment.

    The fragment is deliberately dropped: `#section-3` points *within* a document, so
    treating it as a distinct resource would list the same page many times. YouTube is
    the exception the rules have to accommodate — its `v=` parameter *is* the
    identity, so query parameters cannot simply be discarded wholesale.
    """
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        raise ValidationError("Resource URLs must be absolute.", url=url)

    scheme = parts.scheme.lower()
    host = parts.netloc.lower()
    for default_port in (":80", ":443"):
        if host.endswith(default_port):
            host = host.rsplit(":", 1)[0]

    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=False)
            if key.lower() not in _TRACKING_PARAMS
        )
    )

    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, host, path, query, ""))


def publisher_of(url: str) -> str:
    """The host, as the publisher identity."""
    return urlsplit(url).netloc.lower().removeprefix("www.") or "unknown"


def quality_prior_for(url: str) -> float:
    """A starting quality estimate based on where the resource is published."""
    host = urlsplit(url).netloc.lower()
    if host in PUBLISHER_PRIORS:
        return PUBLISHER_PRIORS[host]
    return PUBLISHER_PRIORS.get(host.removeprefix("www."), DEFAULT_QUALITY_PRIOR)


class ResourceSpec(BaseModel):
    """One curated resource, as authored in YAML."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=300)
    url: str = Field(min_length=8, max_length=2000)
    resource_type: ResourceType
    #: Concept slugs this covers. Validated against the graph at load, so a typo here
    #: fails at seed time rather than producing a resource nothing can ever surface.
    concepts: list[str] = Field(min_length=1, max_length=12)

    description: str = Field(default="", max_length=1000)
    difficulty: int = Field(default=3, ge=1, le=5)
    duration_minutes: int | None = Field(default=None, ge=1, le=100_000)
    authors: list[str] = Field(default_factory=list, max_length=8)
    published_at: date | None = None
    language: str = Field(default="en", max_length=10)
    is_free: bool = True
    #: Objectives this resource actually covers, when known. Lets a recommendation
    #: target the specific capability a learner missed rather than the whole topic.
    covers_objectives: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("url")
    @classmethod
    def _absolute_http_url(cls, value: str) -> str:
        parts = urlsplit(value.strip())
        if parts.scheme not in {"http", "https"}:
            raise ValueError("resource URLs must be http or https")
        if not parts.netloc:
            raise ValueError("resource URLs must include a host")
        return value.strip()

    @property
    def canonical(self) -> str:
        return canonical_url(self.url)

    @property
    def publisher(self) -> str:
        return publisher_of(self.url)

    @property
    def quality_prior(self) -> float:
        return quality_prior_for(self.url)


class ResourceFile(BaseModel):
    """One curated YAML file."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    collection: str = Field(pattern=r"^[a-z0-9-]+$", max_length=60)
    description: str = Field(default="", max_length=500)
    resources: list[ResourceSpec] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class Catalogue:
    """Every curated resource, deduplicated and ready to persist or rank."""

    resources: tuple[ResourceSpec, ...]
    #: Entries dropped as duplicates, with the canonical URL they collided on. Kept
    #: so a curation mistake is visible rather than silently swallowed.
    duplicates: tuple[tuple[str, str], ...] = ()

    def __len__(self) -> int:
        return len(self.resources)

    @property
    def canonical_urls(self) -> frozenset[str]:
        """The set the `known_resources` validator checks generated output against."""
        return frozenset(resource.canonical for resource in self.resources)

    def for_concept(self, slug: str) -> tuple[ResourceSpec, ...]:
        return tuple(r for r in self.resources if slug in r.concepts)

    def concept_slugs(self) -> frozenset[str]:
        return frozenset(slug for resource in self.resources for slug in resource.concepts)

    def by_type(self, resource_type: ResourceType) -> tuple[ResourceSpec, ...]:
        return tuple(r for r in self.resources if r.resource_type is resource_type)

    def coverage_gaps(self, required_slugs: Iterable[str]) -> tuple[str, ...]:
        """Concepts with no resource at all.

        A roadmap step with nothing to read is a dead end, so this is worth surfacing
        at seed time rather than discovering it when a learner arrives there.
        """
        covered = self.concept_slugs()
        return tuple(sorted(slug for slug in required_slugs if slug not in covered))


def load_file(path: Path) -> ResourceFile:
    """Parse and validate one curated file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationError(f"{path.name} is not valid YAML: {exc}", file=path.name) from exc

    if not isinstance(raw, dict):
        raise ValidationError(f"{path.name} must contain a mapping at the top level.")

    try:
        return ResourceFile.model_validate(raw)
    except Exception as exc:
        raise ValidationError(
            f"{path.name} failed schema validation: {exc}", file=path.name
        ) from exc


def load_catalogue(
    directory: Path | None = None, *, known_concepts: Iterable[str] | None = None
) -> Catalogue:
    """Load every curated file, deduplicating by canonical URL.

    Args:
        directory: Where the YAML lives.
        known_concepts: If given, every referenced concept slug must exist. A typo
            here would otherwise create a resource that nothing can ever surface.

    Raises:
        ValidationError: on a malformed file or an unknown concept slug.
    """
    directory = directory or RESOURCE_DATA_DIR
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ValidationError("No resource catalogue files found.", directory=str(directory))

    seen: dict[str, str] = {}
    resources: list[ResourceSpec] = []
    duplicates: list[tuple[str, str]] = []

    for path in paths:
        for resource in load_file(path).resources:
            canonical = resource.canonical
            if canonical in seen:
                duplicates.append((resource.title, canonical))
                continue
            seen[canonical] = resource.title
            resources.append(resource)

    catalogue = Catalogue(
        resources=tuple(sorted(resources, key=lambda r: r.canonical)),
        duplicates=tuple(duplicates),
    )

    if known_concepts is not None:
        allowed = set(known_concepts)
        unknown = sorted(catalogue.concept_slugs() - allowed)
        if unknown:
            raise ValidationError(
                "Resources reference concepts that do not exist in the knowledge graph.",
                slugs=unknown[:20],
            )

    return catalogue


def to_orm_payloads(catalogue: Catalogue) -> tuple[list[dict[str, object]], ...]:
    """Rows for `resources` and `resource_concepts`.

    Plain dicts rather than ORM instances, so the caller chooses between an insert and
    an upsert without this module knowing about sessions.
    """
    resource_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []

    for resource in catalogue.resources:
        resource_rows.append(
            {
                "title": resource.title,
                "url": resource.url,
                "canonical_url": resource.canonical,
                "description": resource.description,
                "resource_type": resource.resource_type,
                "difficulty": resource.difficulty,
                "duration_minutes": resource.duration_minutes,
                "publisher": resource.publisher,
                "authors": resource.authors,
                "published_at": resource.published_at,
                "language": resource.language,
                "is_free": resource.is_free,
                "quality_prior": resource.quality_prior,
            }
        )
        link_rows.extend(
            {
                "canonical_url": resource.canonical,
                "concept_slug": slug,
                "relevance": 1.0,
                "covers_objectives": resource.covers_objectives,
            }
            for slug in resource.concepts
        )

    return resource_rows, link_rows


def embedding_text(resource: ResourceSpec) -> str:
    """The text a resource is embedded from, for semantic rerank."""
    parts: Sequence[str] = (
        resource.title,
        resource.description,
        " ".join(resource.concepts),
    )
    return "\n".join(part for part in parts if part).strip()
