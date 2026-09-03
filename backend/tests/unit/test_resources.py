"""Resource catalogue, link validation, and recommendation tests.

The spec's rule for this area is blunt: never let a model invent a resource URL.
These tests defend the three things that make that rule enforceable — a curated
catalogue with stable identity, link checking that distinguishes *gone* from
*unreachable*, and ranking that is deterministic and explainable without a model.

The real shipped catalogue is exercised, not a fixture, so a bad edit to the YAML
fails here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pathwise.api.errors import ValidationError
from pathwise.models.enums import LearningStyle, ResourceType
from pathwise.services.knowledge.seed import build_graph_from_corpus, load_corpus
from pathwise.services.recommendation.recommender import (
    WEIGHTS,
    RecommendationContext,
    fallback_explanation,
    recommend,
    summarise,
)
from pathwise.services.resource.catalogue import (
    DEFAULT_QUALITY_PRIOR,
    Catalogue,
    ResourceSpec,
    canonical_url,
    load_catalogue,
    publisher_of,
    quality_prior_for,
    to_orm_payloads,
)
from pathwise.services.resource.validation import (
    FakeUrlChecker,
    LinkStatus,
    usable_resources,
    validate_catalogue,
    validate_urls,
)

TODAY = date(2026, 9, 1)


def spec(
    title: str = "A Resource",
    url: str = "https://example.com/thing",
    *,
    concepts: list[str] | None = None,
    resource_type: ResourceType = ResourceType.ARTICLE,
    difficulty: int = 3,
    duration: int | None = 60,
    objectives: list[str] | None = None,
    published: date | None = None,
    free: bool = True,
) -> ResourceSpec:
    return ResourceSpec(
        title=title,
        url=url,
        resource_type=resource_type,
        concepts=concepts or ["target"],
        difficulty=difficulty,
        duration_minutes=duration,
        covers_objectives=objectives or [],
        published_at=published,
        is_free=free,
    )


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    """The real shipped catalogue."""
    return load_catalogue()


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "variant",
    [
        "https://docs.python.org/3/tutorial/",
        "HTTPS://Docs.Python.ORG/3/tutorial",
        "https://docs.python.org:443/3/tutorial",
        "https://docs.python.org/3/tutorial/#section-2",
        "https://docs.python.org/3/tutorial/?utm_source=newsletter",
    ],
)
def test_equivalent_urls_share_one_identity(variant: str) -> None:
    """The same page submitted five ways must be one row, or the catalogue fills with
    duplicates that then compete with each other in every ranking."""
    assert canonical_url(variant) == canonical_url("https://docs.python.org/3/tutorial")


def test_meaningful_query_parameters_survive() -> None:
    """YouTube's `v=` *is* the identity — stripping all query parameters would
    collapse every video on the site into one resource."""
    first = canonical_url("https://www.youtube.com/watch?v=abc&t=1")
    second = canonical_url("https://www.youtube.com/watch?t=1&v=abc")
    other = canonical_url("https://www.youtube.com/watch?v=xyz&t=1")

    assert first == second  # parameter order is irrelevant
    assert first != other  # but the video id is not


def test_different_pages_stay_distinct() -> None:
    assert canonical_url("https://a.example/one") != canonical_url("https://a.example/two")


def test_a_relative_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        canonical_url("/docs/tutorial")


def test_publisher_is_extracted_without_www() -> None:
    assert publisher_of("https://www.example.com/page") == "example.com"


def test_a_known_publisher_starts_above_an_unknown_one() -> None:
    """Not a claim that everything official is good — it is that an official
    reference is a safer default than an unknown blog before either is rated."""
    assert quality_prior_for("https://docs.python.org/3/") > DEFAULT_QUALITY_PRIOR
    assert quality_prior_for("https://some-blog.example/post") == DEFAULT_QUALITY_PRIOR


# --------------------------------------------------------------------------- #
# The shipped catalogue
# --------------------------------------------------------------------------- #


def test_the_catalogue_loads(catalogue: Catalogue) -> None:
    assert len(catalogue) >= 50


def test_the_catalogue_has_no_duplicates(catalogue: Catalogue) -> None:
    assert catalogue.duplicates == ()
    assert len(catalogue.canonical_urls) == len(catalogue)


def test_every_referenced_concept_exists() -> None:
    """A typo here creates a resource nothing can ever surface, so it must fail at
    load rather than silently."""
    graph = build_graph_from_corpus(load_corpus())
    known = {graph.node(cid).slug for cid in graph.node_ids}
    load_catalogue(known_concepts=known)  # raises if any slug is unknown


def test_an_unknown_concept_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "collection: bad\n"
        "resources:\n"
        "  - title: A Thing\n"
        "    url: https://example.com/a\n"
        "    resource_type: article\n"
        "    concepts: [no-such-concept]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="do not exist"):
        load_catalogue(tmp_path, known_concepts={"real-concept"})


def test_the_catalogue_covers_most_of_the_graph(catalogue: Catalogue) -> None:
    graph = build_graph_from_corpus(load_corpus())
    known = {graph.node(cid).slug for cid in graph.node_ids}
    assert len(catalogue.concept_slugs()) / len(known) >= 0.8


def test_coverage_gaps_are_reported_rather_than_hidden(catalogue: Catalogue) -> None:
    """A roadmap step with nothing to read is a dead end, and it is better to know
    at seed time than when a learner arrives there."""
    graph = build_graph_from_corpus(load_corpus())
    known = {graph.node(cid).slug for cid in graph.node_ids}
    gaps = catalogue.coverage_gaps(known)
    assert all(gap not in catalogue.concept_slugs() for gap in gaps)


def test_the_catalogue_mixes_formats(catalogue: Catalogue) -> None:
    """The recommender selects by learning style, so a catalogue of only
    documentation could never serve someone who learns from video."""
    present = {r.resource_type for r in catalogue.resources}
    assert len(present) >= 5
    assert ResourceType.VIDEO in present
    assert ResourceType.DOCUMENTATION in present


def test_every_resource_uses_https(catalogue: Catalogue) -> None:
    for resource in catalogue.resources:
        assert resource.url.startswith("https://"), resource.title


def test_a_malformed_url_is_rejected() -> None:
    with pytest.raises(Exception, match="http"):
        spec(url="ftp://example.com/file")


def test_orm_payloads_cover_the_catalogue(catalogue: Catalogue) -> None:
    resource_rows, link_rows = to_orm_payloads(catalogue)
    assert len(resource_rows) == len(catalogue)
    assert len(link_rows) == sum(len(r.concepts) for r in catalogue.resources)
    assert all(row["canonical_url"] for row in resource_rows)


# --------------------------------------------------------------------------- #
# Link validation
# --------------------------------------------------------------------------- #


async def test_a_healthy_catalogue_passes() -> None:
    catalogue = Catalogue(resources=(spec(url="https://a.example/x"),))
    report = await validate_catalogue(catalogue, FakeUrlChecker())
    assert report.all_usable
    assert report.summary()["ok"] == 1


async def test_a_dead_link_is_detected() -> None:
    dead = "https://a.example/gone"
    report = await validate_urls([dead], FakeUrlChecker(dead=[dead]))
    assert report.dead
    assert report.dead[0].status is LinkStatus.DEAD


async def test_an_unreachable_link_is_not_treated_as_dead() -> None:
    """A timeout is a statement about this moment, not about the resource. Deleting
    a good link because a server was briefly down would erode the catalogue."""
    url = "https://a.example/slow"
    report = await validate_urls([url], FakeUrlChecker(unreachable=[url]))

    assert not report.dead
    assert report.unreachable
    assert report.unreachable[0].is_usable


async def test_only_confirmed_dead_links_are_removed() -> None:
    dead, slow, fine = (
        "https://a.example/gone",
        "https://a.example/slow",
        "https://a.example/ok",
    )
    catalogue = Catalogue(
        resources=(
            spec("Gone", dead),
            spec("Slow", slow),
            spec("Fine", fine),
        )
    )
    report = await validate_catalogue(catalogue, FakeUrlChecker(dead=[dead], unreachable=[slow]))
    remaining = {r.title for r in usable_resources(catalogue, report)}

    assert remaining == {"Slow", "Fine"}


async def test_a_redirect_is_flagged_for_review() -> None:
    """A deep link redirected to a site root almost always means the page is gone
    and the server is hiding it behind a friendly landing page."""
    url = "https://a.example/deep/page"
    report = await validate_urls([url], FakeUrlChecker(redirects={url: "https://a.example/"}))
    assert report.redirected
    assert report.redirected[0].needs_attention


async def test_results_come_back_in_input_order() -> None:
    """Concurrency must not make a report unreproducible or undiffable."""
    urls = [f"https://a.example/{i}" for i in range(20)]
    report = await validate_urls(urls, FakeUrlChecker(), concurrency=5)
    assert [r.url for r in report.results] == urls


async def test_every_url_is_checked_exactly_once() -> None:
    checker = FakeUrlChecker()
    urls = [f"https://a.example/{i}" for i in range(10)]
    await validate_urls(urls, checker)
    assert sorted(checker.checked) == sorted(urls)


async def test_the_report_reads_clearly() -> None:
    dead = "https://a.example/gone"
    report = await validate_urls([dead], FakeUrlChecker(dead=[dead]))
    assert "DEAD" in report.format_report()


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #


def test_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_only_resources_covering_the_concept_are_returned() -> None:
    resources = [
        spec("Right", "https://a.example/1"),
        spec("Wrong", "https://a.example/2", concepts=["other"]),
    ]
    result = recommend(resources, "target", RecommendationContext(today=TODAY))
    assert [i.resource.title for i in result.ranked] == ["Right"]


def test_irrelevant_resources_are_not_logged_as_exclusions() -> None:
    """Recording every unrelated resource buries the exclusions a person would
    actually ask about — and on the real catalogue that was 60 lines of noise."""
    resources = [spec("Right", "https://a.example/1")] + [
        spec(f"Other {i}", f"https://a.example/o{i}", concepts=["elsewhere"]) for i in range(30)
    ]
    result = recommend(resources, "target", RecommendationContext(today=TODAY))
    assert result.excluded == ()


def test_a_focused_resource_outranks_a_broad_one() -> None:
    """A 20-hour course touching eight topics is a worse answer to "help me with this
    concept" than a short piece written about exactly it."""
    focused = spec("Focused", "https://a.example/focused", concepts=["target"])
    broad = spec(
        "Broad",
        "https://a.example/broad",
        concepts=["target", "a", "b", "c", "d", "e", "f"],
    )
    result = recommend([broad, focused], "target", RecommendationContext(today=TODAY))
    assert result.ranked[0].resource.title == "Focused"


def test_format_follows_the_stated_learning_style() -> None:
    video = spec("Video", "https://a.example/v", resource_type=ResourceType.VIDEO)
    book = spec("Book", "https://a.example/b", resource_type=ResourceType.BOOK)

    for style, expected in (
        (LearningStyle.VIDEO, "Video"),
        (LearningStyle.READING, "Book"),
    ):
        result = recommend(
            [video, book], "target", RecommendationContext(learning_style=style, today=TODAY)
        )
        assert result.ranked[0].resource.title == expected, style


def test_difficulty_is_matched_to_measured_mastery() -> None:
    easy = spec("Easy", "https://a.example/e", difficulty=1)
    hard = spec("Hard", "https://a.example/h", difficulty=5)

    beginner = recommend(
        [easy, hard], "target", RecommendationContext(concept_mastery=0.0, today=TODAY)
    )
    expert = recommend(
        [easy, hard], "target", RecommendationContext(concept_mastery=1.0, today=TODAY)
    )
    assert beginner.ranked[0].resource.title == "Easy"
    assert expert.ranked[0].resource.title == "Hard"


def test_something_that_fits_the_time_available_is_preferred() -> None:
    short = spec("Short", "https://a.example/s", duration=45)
    long_one = spec("Long", "https://a.example/l", duration=2000)
    result = recommend(
        [long_one, short], "target", RecommendationContext(minutes_available=60, today=TODAY)
    )
    assert result.ranked[0].resource.title == "Short"


def test_a_long_resource_is_ranked_down_not_excluded() -> None:
    """A good long resource is still worth surfacing, just not first."""
    result = recommend(
        [spec("Long", "https://a.example/l", duration=5000)],
        "target",
        RecommendationContext(minutes_available=30, today=TODAY),
    )
    assert result.ranked


def test_already_seen_resources_are_not_repeated() -> None:
    """Recommending the same video twice is the fastest way to look like nothing is
    being tracked."""
    seen = spec("Seen", "https://a.example/seen")
    fresh = spec("Fresh", "https://a.example/fresh")
    result = recommend(
        [seen, fresh],
        "target",
        RecommendationContext(already_seen=frozenset({seen.canonical}), today=TODAY),
    )
    assert [i.resource.title for i in result.ranked] == ["Fresh"]
    assert result.excluded == (("Seen", "already seen"),)


def test_paid_resources_can_be_filtered_out() -> None:
    result = recommend(
        [spec("Paid", "https://a.example/p", free=False)],
        "target",
        RecommendationContext(free_only=True, today=TODAY),
    )
    assert result.is_empty
    assert result.excluded == (("Paid", "not free"),)


def test_a_resource_covering_the_missed_objective_is_preferred() -> None:
    """The difference between "study this again" and "here is the part you got
    wrong"."""
    general = spec("General", "https://a.example/g")
    targeted = spec("Targeted", "https://a.example/t", objectives=["lo-2"])
    result = recommend(
        [general, targeted],
        "target",
        RecommendationContext(weak_objectives=frozenset({"lo-2"}), today=TODAY),
    )
    assert result.ranked[0].resource.title == "Targeted"
    assert "lo-2" in result.ranked[0].details["difficulty_fit"]


def test_ranking_is_deterministic(catalogue: Catalogue) -> None:
    context = RecommendationContext(concept_mastery=0.4, today=TODAY)
    first = recommend(catalogue.resources, "gradient-descent", context)
    second = recommend(catalogue.resources, "gradient-descent", context)
    assert [i.resource.canonical for i in first.ranked] == [
        i.resource.canonical for i in second.ranked
    ]


def test_the_score_is_the_sum_of_its_weighted_factors() -> None:
    item = recommend([spec()], "target", RecommendationContext(today=TODAY)).ranked[0]
    expected = sum(value * WEIGHTS[name] for name, value in item.factors.items())
    assert item.score == pytest.approx(expected)


def test_every_factor_carries_a_reason() -> None:
    """These are handed to the explanation prompt verbatim."""
    item = recommend([spec()], "target", RecommendationContext(today=TODAY)).ranked[0]
    assert set(item.details) >= set(item.factors)
    assert all(text.strip() for text in item.details.values())


def test_nothing_suitable_is_reported_honestly() -> None:
    result = recommend([], "target", RecommendationContext(today=TODAY))
    assert result.is_empty
    assert summarise(result)["resources"] == []


def test_the_prompt_payload_contains_only_catalogue_rows(catalogue: Catalogue) -> None:
    """The model can only describe what it is given, and it is given nothing that is
    not already in the catalogue."""
    result = recommend(catalogue.resources, "gradient-descent", RecommendationContext(today=TODAY))
    payload = result.to_prompt_json()
    assert payload
    for entry in payload:
        assert entry["url"] in catalogue.canonical_urls


def test_the_deterministic_explanation_states_a_real_reason() -> None:
    item = recommend([spec()], "target", RecommendationContext(today=TODAY)).ranked[0]
    text = fallback_explanation(item)
    assert item.resource.title in text
    assert item.details[item.dominant_factor] in text
