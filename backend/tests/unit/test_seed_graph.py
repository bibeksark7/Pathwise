"""Validation of the shipped knowledge graph.

These tests run against the **real** seed files, not a fixture. The seed graph is the
highest-risk artifact in the system — a wrong prerequisite edge silently produces a
wrong roadmap, a wrong blame attribution, and a wrong next-topic decision — and it is
edited by hand, so it needs a guard that fails at commit time rather than in
production.

The cross-file checks matter most: two individually reasonable edges authored in two
different domain files can compose into a cycle that neither file reveals alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pathwise.api.errors import CycleError, ValidationError
from pathwise.services.knowledge.graph import KnowledgeGraph
from pathwise.services.knowledge.seed import (
    SEED_DATA_DIR,
    ConceptSpec,
    DomainSpec,
    SeedCorpus,
    build_graph_from_corpus,
    concept_id_for,
    embedding_text,
    load_corpus,
    load_domain_file,
    to_orm_payloads,
    validate_corpus,
)


@pytest.fixture(scope="module")
def corpus() -> SeedCorpus:
    return load_corpus()


@pytest.fixture(scope="module")
def graph(corpus: SeedCorpus) -> KnowledgeGraph:
    return build_graph_from_corpus(corpus)


# --------------------------------------------------------------------------- #
# The shipped corpus
# --------------------------------------------------------------------------- #


def test_every_seed_file_parses_and_validates() -> None:
    """Catches a malformed edit before it reaches a migration."""
    paths = sorted(SEED_DATA_DIR.glob("*.yaml"))
    assert paths, "no seed files found"
    for path in paths:
        assert isinstance(load_domain_file(path), DomainSpec), path.name


def test_corpus_is_substantial(corpus: SeedCorpus) -> None:
    """A graph too thin to route around cannot demonstrate adaptive behaviour."""
    assert len(corpus.concepts) >= 60
    assert len(corpus.domains) >= 3


def test_the_assembled_graph_is_acyclic(graph: KnowledgeGraph) -> None:
    """The invariant every downstream algorithm assumes."""
    assert len(graph.topological_order()) == len(graph)


def test_slugs_are_unique_across_every_domain(corpus: SeedCorpus) -> None:
    slugs = [concept.slug for concept in corpus.concepts]
    assert len(slugs) == len(set(slugs))


def test_no_dangling_references(corpus: SeedCorpus) -> None:
    """Every `requires`, `related`, and `builds_on` target must resolve."""
    known = {concept.slug for concept in corpus.concepts}
    for concept in corpus.concepts:
        references = [edge.slug for edge in concept.requires] + concept.related + concept.builds_on
        for reference in references:
            assert reference in known, f"{concept.slug} -> {reference}"


def test_concept_ids_are_stable_across_runs() -> None:
    """Ids are derived from slugs, so seeding is idempotent and fixtures stay valid."""
    assert concept_id_for("gradient-descent") == concept_id_for("gradient-descent")
    assert concept_id_for("gradient-descent") != concept_id_for("backpropagation")


def test_every_concept_has_at_least_one_objective(corpus: SeedCorpus) -> None:
    """Objectives are what assessment questions bind to; a concept without them can
    produce a score but never evidence about a specific capability."""
    for concept in corpus.concepts:
        assert concept.objectives, concept.slug


def test_objective_ids_are_unique_within_a_concept(corpus: SeedCorpus) -> None:
    for concept in corpus.concepts:
        ids = [objective.id for objective in concept.objectives]
        assert len(ids) == len(set(ids)), concept.slug


def test_no_concept_is_impossibly_large(corpus: SeedCorpus) -> None:
    """Granularity rule: a concept you cannot finish is a concept you cannot blame."""
    for concept in corpus.concepts:
        assert concept.estimated_minutes <= 360, concept.slug


# --------------------------------------------------------------------------- #
# Structural sanity of the curated content
# --------------------------------------------------------------------------- #


def test_the_graph_has_entry_points(graph: KnowledgeGraph) -> None:
    """Concepts with no prerequisites — a total beginner must be able to start."""
    roots = [c for c in graph.node_ids if not graph.direct_requirements(c)]
    assert len(roots) >= 3


def test_the_graph_has_meaningful_depth(graph: KnowledgeGraph) -> None:
    """A flat graph cannot express prerequisites and makes adaptation trivial."""
    deepest = max(len(graph.prerequisite_closure(concept_id)) for concept_id in graph.node_ids)
    assert deepest >= 15


def test_domains_are_genuinely_interconnected(graph: KnowledgeGraph, corpus: SeedCorpus) -> None:
    """Machine learning must actually depend on the maths and programming beneath it.

    This is what makes cross-file cycle validation necessary, and what lets blame
    attribution reach from a failed ML topic down into a calculus gap.
    """
    domain_of = {concept_id_for(c.slug): corpus.domain_of(c.slug) for c in corpus.concepts}
    cross_domain = [
        (concept_id, requirement.concept_id)
        for concept_id in graph.node_ids
        for requirement in graph.direct_requirements(concept_id)
        if domain_of[concept_id] != domain_of[requirement.concept_id]
    ]
    assert len(cross_domain) >= 10


def test_no_concept_requires_more_than_four_direct_prerequisites(
    graph: KnowledgeGraph,
) -> None:
    """A concept gated behind many prerequisites is usually too coarse, and it makes
    blame attribution ambiguous — too many equally plausible suspects."""
    for concept_id in graph.node_ids:
        assert len(graph.direct_requirements(concept_id)) <= 4, graph.node(concept_id).slug


def test_no_prerequisite_edge_is_pointing_backwards(graph: KnowledgeGraph) -> None:
    """A difficulty-5 prerequisite for a difficulty-2 concept means the edge is
    reversed — something a DAG check alone cannot catch, since the reversed graph is
    equally acyclic.

    The tolerance is one level, not zero: a harder theoretical concept legitimately
    gates an operationally simpler one (understanding evaluation metrics before
    setting up experiment tracking, or gradients before gradient descent). A gap of
    two or more has no such reading and indicates the edge was authored the wrong
    way round.
    """
    for concept_id in graph.node_ids:
        node = graph.node(concept_id)
        for requirement in graph.direct_requirements(concept_id):
            required = graph.node(requirement.concept_id)
            assert required.difficulty <= node.difficulty + 1, (
                f"{required.slug} (d{required.difficulty}) "
                f"gates {node.slug} (d{node.difficulty}) — is this edge reversed?"
            )


# --------------------------------------------------------------------------- #
# The spec's worked example must actually be encoded
# --------------------------------------------------------------------------- #


def test_the_calculus_to_neural_networks_path_exists(graph: KnowledgeGraph) -> None:
    """The spec's example: Calculus -> Gradient Descent -> Neural Networks."""
    closure = graph.prerequisite_closure(concept_id_for("backpropagation"))
    assert concept_id_for("gradient-descent") in closure
    assert concept_id_for("derivatives") in closure
    assert concept_id_for("chain-rule") in closure


def test_optimization_fundamentals_sits_between_calculus_and_gradient_descent(
    graph: KnowledgeGraph,
) -> None:
    """The remediation the spec describes inserting must be a real graph position,
    not something the adaptation engine has to invent."""
    requirements = {
        r.concept_id for r in graph.direct_requirements(concept_id_for("gradient-descent"))
    }
    assert concept_id_for("optimization-fundamentals") in requirements

    optimization_needs = {
        r.concept_id for r in graph.direct_requirements(concept_id_for("optimization-fundamentals"))
    }
    assert concept_id_for("derivatives") in optimization_needs


def test_blame_reaches_from_backpropagation_down_to_the_chain_rule(
    graph: KnowledgeGraph,
) -> None:
    """The system's core claim, on real data: a learner failing backpropagation with
    weak calculus should be pointed at the chain rule, not at neural networks."""
    mastery = {
        concept_id_for("neural-network-fundamentals"): 0.9,
        concept_id_for("gradients-and-jacobians"): 0.9,
        concept_id_for("chain-rule"): 0.15,
    }
    candidates = graph.blame_candidates(concept_id_for("backpropagation"), mastery, max_depth=2)
    assert candidates
    assert candidates[0].concept_id == concept_id_for("chain-rule")


# --------------------------------------------------------------------------- #
# Persistence payloads
# --------------------------------------------------------------------------- #


def test_orm_payloads_cover_every_concept_and_edge(corpus: SeedCorpus) -> None:
    concept_rows, edge_rows = to_orm_payloads(corpus)
    assert len(concept_rows) == len(corpus.concepts)

    expected_edges = sum(
        len(c.requires) + len(c.related) + len(c.builds_on) for c in corpus.concepts
    )
    assert len(edge_rows) == expected_edges


def test_edge_payloads_point_from_requirement_to_dependent(corpus: SeedCorpus) -> None:
    """`requires` is authored on the dependent but stored as PREREQUISITE_OF pointing
    the other way — an inversion here would reverse the entire graph."""
    _, edge_rows = to_orm_payloads(corpus)
    gradient_descent = concept_id_for("gradient-descent")
    optimization = concept_id_for("optimization-fundamentals")

    match = next(
        row
        for row in edge_rows
        if row["source_id"] == optimization and row["target_id"] == gradient_descent
    )
    assert match["strength"] == pytest.approx(1.0)


def test_embedding_text_carries_the_semantic_fields(corpus: SeedCorpus) -> None:
    concept = next(c for c in corpus.concepts if c.slug == "gradient-descent")
    text = embedding_text(concept)
    assert concept.name in text
    assert concept.description[:40] in text
    assert concept.objectives[0].text in text
    assert concept.aliases[0] in text  # aliases help match a learner's own wording


def test_embedding_text_excludes_tags() -> None:
    """Tags are navigational, not semantic; embedding them would pull unrelated
    concepts together purely because they share a label."""
    concept = ConceptSpec.model_validate(
        _minimal("alpha", tags=["zzdistinctivetag"], aliases=["keepthisalias"])
    )
    text = embedding_text(concept)
    assert "zzdistinctivetag" not in text
    assert "keepthisalias" in text


# --------------------------------------------------------------------------- #
# Rejection paths — LLM proposals go through this same validation
# --------------------------------------------------------------------------- #


def _minimal(slug: str, **overrides: object) -> dict[str, object]:
    return {
        "slug": slug,
        "name": slug.title(),
        "description": "A description long enough to satisfy the schema minimum.",
        "difficulty": 3,
        "estimated_minutes": 120,
        "objectives": [{"id": "lo-1", "text": "Do the thing competently."}],
        **overrides,
    }


def test_a_cyclic_corpus_is_rejected() -> None:
    domain = DomainSpec.model_validate(
        {
            "domain": "test",
            "concepts": [
                _minimal("alpha", requires=[{"slug": "beta"}]),
                _minimal("beta", requires=[{"slug": "alpha"}]),
            ],
        }
    )
    with pytest.raises(CycleError):
        validate_corpus(SeedCorpus(domains=[domain]))


def test_a_dangling_reference_is_rejected() -> None:
    domain = DomainSpec.model_validate(
        {"domain": "test", "concepts": [_minimal("alpha", requires=[{"slug": "ghost"}])]}
    )
    with pytest.raises(ValidationError, match="do not exist"):
        validate_corpus(SeedCorpus(domains=[domain]))


def test_duplicate_slugs_across_files_are_rejected() -> None:
    """The failure mode a per-file check cannot see."""
    first = DomainSpec.model_validate({"domain": "one", "concepts": [_minimal("shared")]})
    second = DomainSpec.model_validate({"domain": "two", "concepts": [_minimal("shared")]})
    with pytest.raises(ValidationError, match="unique"):
        validate_corpus(SeedCorpus(domains=[first, second]))


def test_unknown_fields_are_rejected() -> None:
    """A typo in a field name must fail loudly rather than be silently dropped."""
    with pytest.raises(Exception, match="extra"):
        ConceptSpec.model_validate(_minimal("alpha", dificulty=3))


def test_an_invalid_slug_is_rejected() -> None:
    with pytest.raises(Exception, match="pattern"):
        ConceptSpec.model_validate(_minimal("Not A Slug"))


def test_a_concept_may_not_list_the_same_prerequisite_twice() -> None:
    with pytest.raises(Exception, match="twice"):
        ConceptSpec.model_validate(_minimal("alpha", requires=[{"slug": "beta"}, {"slug": "beta"}]))


def test_a_missing_seed_directory_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="No knowledge-graph seed files"):
        load_corpus(tmp_path)
