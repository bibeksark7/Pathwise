"""Loading and validating the curated knowledge graph.

The seed graph is hand-authored YAML, not generated. It is the highest-leverage
artifact in the system — a wrong prerequisite edge produces a wrong roadmap, a wrong
blame attribution, and a wrong next-topic decision — so it gets the strictest
validation of anything here:

* Pydantic schema validation on every concept and edge.
* Referential integrity: every ``requires`` target must resolve to a real slug.
* **Acyclicity across all domain files at once.** Files are loaded independently but
  validated together, because a cycle is most likely to be introduced by two
  separately-plausible cross-domain edges.
* Slug uniqueness across the whole corpus.

The same validation path is what LLM-proposed concepts go through, so a model
suggestion cannot enter the graph on weaker terms than a curated one.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pathwise.api.errors import ValidationError
from pathwise.models.enums import ConceptSource, RelationType
from pathwise.services.knowledge.graph import GraphEdge, GraphNode, KnowledgeGraph

#: Concept ids are derived from slugs rather than random, so re-running the seed is
#: idempotent and the same concept keeps the same id across environments. That makes
#: fixtures, eval datasets, and support conversations refer to stable identifiers.
CONCEPT_NAMESPACE: Final = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

SEED_DATA_DIR: Final = Path(__file__).resolve().parents[2] / "data" / "knowledge_graph"

SLUG_PATTERN: Final = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


def concept_id_for(slug: str) -> uuid.UUID:
    """The stable UUID for a concept slug."""
    return uuid.uuid5(CONCEPT_NAMESPACE, slug)


class ObjectiveSpec(BaseModel):
    """One learning objective.

    Objectives are the unit assessment questions bind to. Without them a score is a
    number about a topic; with them it is evidence about a specific capability, which
    is what the adaptive engine actually needs.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^lo-\d+$")
    text: str = Field(min_length=10, max_length=300)
    bloom: str = Field(
        default="understand",
        pattern="^(remember|understand|apply|analyze|evaluate|create)$",
    )


class EdgeSpec(BaseModel):
    """A prerequisite reference from within a concept definition."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=SLUG_PATTERN)
    strength: float = Field(default=1.0, ge=0.1, le=1.0)
    rationale: str | None = Field(default=None, max_length=300)


class ConceptSpec(BaseModel):
    """One concept as authored in a seed file."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=SLUG_PATTERN, max_length=120)
    name: str = Field(min_length=2, max_length=200)
    description: str = Field(min_length=20, max_length=2000)
    difficulty: int = Field(ge=1, le=5)
    estimated_minutes: int = Field(ge=15, le=6000)

    objectives: list[ObjectiveSpec] = Field(min_length=1, max_length=12)
    tags: list[str] = Field(default_factory=list, max_length=12)
    aliases: list[str] = Field(default_factory=list, max_length=8)

    requires: list[EdgeSpec] = Field(default_factory=list, max_length=10)
    related: list[str] = Field(default_factory=list, max_length=10)
    builds_on: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("objectives")
    @classmethod
    def _objective_ids_unique(cls, value: list[ObjectiveSpec]) -> list[ObjectiveSpec]:
        ids = [objective.id for objective in value]
        if len(ids) != len(set(ids)):
            raise ValueError("objective ids must be unique within a concept")
        return value

    @field_validator("requires")
    @classmethod
    def _requirements_unique(cls, value: list[EdgeSpec]) -> list[EdgeSpec]:
        slugs = [edge.slug for edge in value]
        if len(slugs) != len(set(slugs)):
            raise ValueError("a concept may not list the same prerequisite twice")
        return value

    @property
    def id(self) -> uuid.UUID:
        return concept_id_for(self.slug)


class DomainSpec(BaseModel):
    """One seed file: a domain's worth of concepts."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    domain: str = Field(pattern=r"^[a-z0-9-]+$", max_length=60)
    description: str = Field(default="", max_length=500)
    concepts: list[ConceptSpec] = Field(min_length=1)

    @field_validator("concepts")
    @classmethod
    def _slugs_unique_within_file(cls, value: list[ConceptSpec]) -> list[ConceptSpec]:
        slugs = [concept.slug for concept in value]
        duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
        if duplicates:
            raise ValueError(f"duplicate concept slugs in file: {sorted(duplicates)}")
        return value


class SeedCorpus(BaseModel):
    """Every domain file, validated together.

    Cross-file validation is the point: the individually plausible edges
    ``optimization requires calculus`` and ``calculus requires optimization``, authored
    in two different files by two different people, only look like a cycle when the
    corpus is assembled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domains: list[DomainSpec]

    @property
    def concepts(self) -> list[ConceptSpec]:
        return [concept for domain in self.domains for concept in domain.concepts]

    def domain_of(self, slug: str) -> str:
        for domain in self.domains:
            if any(concept.slug == slug for concept in domain.concepts):
                return domain.domain
        raise KeyError(slug)


def load_domain_file(path: Path) -> DomainSpec:
    """Parse and schema-validate one seed file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationError(f"{path.name} is not valid YAML: {exc}", file=path.name) from exc

    if not isinstance(raw, dict):
        raise ValidationError(f"{path.name} must contain a mapping at the top level.")

    try:
        return DomainSpec.model_validate(raw)
    except Exception as exc:
        raise ValidationError(
            f"{path.name} failed schema validation: {exc}", file=path.name
        ) from exc


def load_corpus(directory: Path | None = None) -> SeedCorpus:
    """Load every seed file in a directory and validate them as one corpus.

    Raises:
        ValidationError: on a schema failure, a duplicate slug, or a dangling
            reference.
        CycleError: if the assembled prerequisite edges are not a DAG.
    """
    directory = directory or SEED_DATA_DIR
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ValidationError("No knowledge-graph seed files found.", directory=str(directory))

    corpus = SeedCorpus(domains=[load_domain_file(path) for path in paths])
    validate_corpus(corpus)
    return corpus


def validate_corpus(corpus: SeedCorpus) -> KnowledgeGraph:
    """Check global invariants and return the assembled graph.

    Returning the graph rather than a boolean means the caller that validated is the
    caller that gets the validated object — there is no window in which something
    else could be persisted instead.
    """
    concepts = corpus.concepts
    slugs = [concept.slug for concept in concepts]

    duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if duplicates:
        raise ValidationError(
            "Concept slugs must be unique across the whole corpus.", slugs=duplicates
        )

    known = set(slugs)
    dangling: list[str] = []
    for concept in concepts:
        references = [edge.slug for edge in concept.requires] + concept.related + concept.builds_on
        dangling.extend(
            f"{concept.slug} -> {reference}" for reference in references if reference not in known
        )

    if dangling:
        raise ValidationError(
            "Seed files reference concepts that do not exist.", references=sorted(dangling)[:20]
        )

    # KnowledgeGraph's constructor raises CycleError if the ordering edges are cyclic.
    return build_graph_from_corpus(corpus)


def build_graph_from_corpus(corpus: SeedCorpus) -> KnowledgeGraph:
    """Assemble an in-memory graph from a validated corpus."""
    nodes = [
        GraphNode(
            id=concept.id,
            slug=concept.slug,
            name=concept.name,
            difficulty=concept.difficulty,
            estimated_minutes=concept.estimated_minutes,
            domain=domain.domain,
            description=concept.description,
            objective_ids=tuple(objective.id for objective in concept.objectives),
        )
        for domain in corpus.domains
        for concept in domain.concepts
    ]
    return KnowledgeGraph(nodes, _edges_from(corpus.concepts))


def _edges_from(concepts: Iterable[ConceptSpec]) -> list[GraphEdge]:
    """Flatten authored references into graph edges.

    ``requires`` is authored on the *dependent* because that is how a person thinks
    about it ("to learn X you need Y"), and emitted as ``PREREQUISITE_OF`` pointing
    the other way, which is how the graph stores it.
    """
    edges: list[GraphEdge] = []
    for concept in concepts:
        edges.extend(
            GraphEdge(
                source=concept_id_for(edge.slug),
                target=concept.id,
                relation=RelationType.PREREQUISITE_OF,
                strength=edge.strength,
            )
            for edge in concept.requires
        )
        edges.extend(
            GraphEdge(
                source=concept.id,
                target=concept_id_for(slug),
                relation=RelationType.RELATED_TO,
                strength=0.5,
            )
            for slug in concept.related
        )
        edges.extend(
            GraphEdge(
                source=concept.id,
                target=concept_id_for(slug),
                relation=RelationType.BUILDS_ON,
                strength=0.7,
            )
            for slug in concept.builds_on
        )
    return edges


def to_orm_payloads(
    corpus: SeedCorpus,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a validated corpus into rows for ``concepts`` and ``concept_edges``.

    Returns plain dicts rather than ORM instances so the caller can choose between a
    bulk insert and an upsert without this module knowing about sessions.
    """
    concept_rows: list[dict[str, Any]] = [
        {
            "id": concept.id,
            "slug": concept.slug,
            "name": concept.name,
            "domain": domain.domain,
            "description": concept.description,
            "difficulty": concept.difficulty,
            "estimated_minutes": concept.estimated_minutes,
            "learning_objectives": [objective.model_dump() for objective in concept.objectives],
            "tags": concept.tags,
            "aliases": concept.aliases,
            "source": ConceptSource.SEED,
        }
        for domain in corpus.domains
        for concept in domain.concepts
    ]

    edge_rows: list[dict[str, Any]] = []
    for concept in corpus.concepts:
        edge_rows.extend(
            {
                "source_id": concept_id_for(edge.slug),
                "target_id": concept.id,
                "relation": RelationType.PREREQUISITE_OF,
                "strength": edge.strength,
                "source": ConceptSource.SEED,
                "rationale": edge.rationale,
            }
            for edge in concept.requires
        )
        edge_rows.extend(
            {
                "source_id": concept.id,
                "target_id": concept_id_for(slug),
                "relation": RelationType.RELATED_TO,
                "strength": 0.5,
                "source": ConceptSource.SEED,
                "rationale": None,
            }
            for slug in concept.related
        )
        edge_rows.extend(
            {
                "source_id": concept.id,
                "target_id": concept_id_for(slug),
                "relation": RelationType.BUILDS_ON,
                "strength": 0.7,
                "source": ConceptSource.SEED,
                "rationale": None,
            }
            for slug in concept.builds_on
        )

    return concept_rows, edge_rows


def embedding_text(concept: ConceptSpec) -> str:
    """The text a concept is embedded from.

    Name, aliases, description, and objectives — deliberately not the tags, which are
    navigational rather than semantic and would blur near-neighbours together.
    """
    parts: Sequence[str] = (
        concept.name,
        " ".join(concept.aliases),
        concept.description,
        " ".join(objective.text for objective in concept.objectives),
    )
    return "\n".join(part for part in parts if part).strip()
