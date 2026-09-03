"""Checking that resource URLs actually resolve.

Curation alone is not enough. A link that was correct when it was added rots, and a
learner who clicks three dead recommendations stops trusting everything else the
system says. So every URL is fetched before it can be recommended, and re-fetched on
a schedule afterwards.

The distinction that matters is **dead versus unreachable**:

* ``404``/``410`` means the page is gone. Act on it — stop recommending the resource.
* A timeout, a connection error, or a ``5xx`` means *we* could not reach it right now.
  That is a statement about this moment, not about the resource, and deleting a good
  link because a server was briefly down would quietly erode the catalogue over time.

Only the first kind changes a resource's status. The second is recorded and retried.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Protocol

from pathwise.logging_config import get_logger
from pathwise.services.resource.catalogue import Catalogue, ResourceSpec

log = get_logger(__name__)

#: Politeness and self-protection. Checking 60 URLs should not look like an attack,
#: and should not take ten minutes either.
DEFAULT_CONCURRENCY: Final = 8
DEFAULT_TIMEOUT_SECONDS: Final = 15.0

#: Status codes that mean the resource is genuinely gone, as opposed to temporarily
#: unavailable. Only these justify pulling a resource from recommendation.
DEAD_STATUS_CODES: Final = frozenset({404, 410})

#: Some servers reject HEAD outright, or lie about it. A 4xx from HEAD is retried
#: with GET before concluding anything.
_HEAD_UNRELIABLE_CODES: Final = frozenset({400, 401, 403, 405, 406, 501})

#: A plain identifying user agent. Sending none at all gets blocked by several of the
#: hosts in the catalogue.
USER_AGENT: Final = "Pathwise-LinkChecker/1.0 (+https://github.com/bibeksark7/Pathwise)"


class LinkStatus(StrEnum):
    """What a check concluded."""

    OK = "ok"
    #: Definitively gone. Stop recommending it.
    DEAD = "dead"
    #: Could not be reached this time. Retry rather than act.
    UNREACHABLE = "unreachable"
    #: Resolved, but somewhere else. Worth a human look — a redirect to a site root
    #: usually means the specific page is gone.
    REDIRECTED = "redirected"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of checking one URL."""

    url: str
    status: LinkStatus
    status_code: int | None = None
    final_url: str | None = None
    error: str | None = None
    elapsed_ms: float = 0.0
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_usable(self) -> bool:
        """Whether the resource may still be recommended.

        An unreachable result counts as usable: we have no evidence the resource is
        bad, only that we could not confirm it right now.
        """
        return self.status in {LinkStatus.OK, LinkStatus.REDIRECTED, LinkStatus.UNREACHABLE}

    @property
    def needs_attention(self) -> bool:
        return self.status in {LinkStatus.DEAD, LinkStatus.REDIRECTED}


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The result of checking a whole catalogue."""

    results: tuple[CheckResult, ...]

    @property
    def ok(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.status is LinkStatus.OK)

    @property
    def dead(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.status is LinkStatus.DEAD)

    @property
    def redirected(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.status is LinkStatus.REDIRECTED)

    @property
    def unreachable(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.status is LinkStatus.UNREACHABLE)

    @property
    def all_usable(self) -> bool:
        return not self.dead

    def summary(self) -> dict[str, int]:
        return {
            "checked": len(self.results),
            "ok": len(self.ok),
            "redirected": len(self.redirected),
            "dead": len(self.dead),
            "unreachable": len(self.unreachable),
        }

    def format_report(self) -> str:
        """A human-readable summary, for the seeding CLI."""
        counts = self.summary()
        lines = [
            f"Checked {counts['checked']} URLs: "
            f"{counts['ok']} ok, {counts['redirected']} redirected, "
            f"{counts['dead']} dead, {counts['unreachable']} unreachable."
        ]
        for result in self.dead:
            lines.append(f"  DEAD ({result.status_code})  {result.url}")
        for result in self.redirected:
            lines.append(f"  MOVED -> {result.final_url}\n         from {result.url}")
        for result in self.unreachable:
            lines.append(f"  UNREACHABLE  {result.url}  ({result.error})")
        return "\n".join(lines)


class UrlChecker(Protocol):
    """Something that can tell us whether a URL resolves."""

    async def check(self, url: str) -> CheckResult: ...


class FakeUrlChecker:
    """A checker that answers from a table. Used by the test suite.

    Link checking is the one part of this pipeline that genuinely needs the network,
    so it is behind an interface — the suite stays offline and deterministic, and
    every failure mode (dead, moved, timed out) is one line to arrange.
    """

    def __init__(
        self,
        *,
        dead: Iterable[str] = (),
        redirects: dict[str, str] | None = None,
        unreachable: Iterable[str] = (),
    ) -> None:
        self._dead = set(dead)
        self._redirects = redirects or {}
        self._unreachable = set(unreachable)
        self.checked: list[str] = []

    async def check(self, url: str) -> CheckResult:
        self.checked.append(url)

        if url in self._dead:
            return CheckResult(url=url, status=LinkStatus.DEAD, status_code=404)
        if url in self._unreachable:
            return CheckResult(url=url, status=LinkStatus.UNREACHABLE, error="simulated timeout")
        if url in self._redirects:
            return CheckResult(
                url=url,
                status=LinkStatus.REDIRECTED,
                status_code=200,
                final_url=self._redirects[url],
            )
        return CheckResult(url=url, status=LinkStatus.OK, status_code=200, final_url=url)


class HttpUrlChecker:
    """Checks URLs over the network.

    Tries ``HEAD`` first because it is cheap, then falls back to ``GET`` — a
    surprising number of hosts reject or mishandle HEAD, and treating that as a dead
    link would remove perfectly good resources.
    """

    def __init__(
        self,
        client: object | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        # Typed loosely so this module does not import httpx at definition time; the
        # client is supplied by the composition root or created on first use.
        self._client = client
        self._timeout = timeout

    async def _get_client(self) -> object:
        if self._client is None:
            import httpx2  # imported lazily so tests never need it

            self._client = httpx2.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
        return self._client

    async def check(self, url: str) -> CheckResult:
        import time

        client = await self._get_client()
        started = time.perf_counter()

        try:
            response = await client.head(url)  # type: ignore[attr-defined]
            if response.status_code in _HEAD_UNRELIABLE_CODES:
                response = await client.get(url)  # type: ignore[attr-defined]
        except Exception as exc:
            # A network failure says nothing about the resource itself.
            return CheckResult(
                url=url,
                status=LinkStatus.UNREACHABLE,
                error=f"{type(exc).__name__}: {exc}"[:200],
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        code = response.status_code
        final = str(response.url)

        if code in DEAD_STATUS_CODES:
            status = LinkStatus.DEAD
        elif code >= 500:
            status = LinkStatus.UNREACHABLE
        elif code >= 400:
            # 401/403 after a GET retry means it exists but is gated. Not dead.
            status = LinkStatus.OK
        elif _moved_meaningfully(url, final):
            status = LinkStatus.REDIRECTED
        else:
            status = LinkStatus.OK

        return CheckResult(
            url=url,
            status=status,
            status_code=code,
            final_url=final,
            elapsed_ms=elapsed_ms,
        )

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "aclose"):
            await self._client.aclose()


def _moved_meaningfully(original: str, final: str) -> bool:
    """Whether a redirect landed somewhere worth a human look.

    Ignores the cosmetic cases — a trailing slash, http to https, adding `www` — and
    flags the one that matters: a deep link redirected to a site root, which almost
    always means the specific page is gone and the server is hiding it behind a
    friendly landing page.
    """
    from pathwise.services.resource.catalogue import canonical_url

    try:
        if canonical_url(original) == canonical_url(final):
            return False
    except Exception:
        return True

    from urllib.parse import urlsplit

    original_path = urlsplit(original).path.rstrip("/")
    final_path = urlsplit(final).path.rstrip("/")

    # Landed on the root while the original pointed somewhere specific.
    return bool(original_path) and not final_path


async def validate_urls(
    urls: Sequence[str],
    checker: UrlChecker,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> ValidationReport:
    """Check many URLs concurrently, bounded.

    Results come back in input order regardless of completion order, so a report is
    reproducible and diffable between runs.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(url: str) -> CheckResult:
        async with semaphore:
            return await checker.check(url)

    results = await asyncio.gather(*(bounded(url) for url in urls))
    return ValidationReport(results=tuple(results))


async def validate_catalogue(
    catalogue: Catalogue,
    checker: UrlChecker,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> ValidationReport:
    """Check every resource in a catalogue."""
    report = await validate_urls(
        [resource.url for resource in catalogue.resources], checker, concurrency=concurrency
    )

    if report.dead:
        log.warning("catalogue_has_dead_links", count=len(report.dead))
    if report.redirected:
        log.info("catalogue_has_redirects", count=len(report.redirected))

    return report


def usable_resources(catalogue: Catalogue, report: ValidationReport) -> tuple[ResourceSpec, ...]:
    """The resources that may still be recommended.

    Excludes only confirmed-dead links. A resource we merely failed to reach stays in
    — absence of confirmation is not evidence of absence, and dropping it would let a
    transient outage silently shrink the catalogue.
    """
    dead = {result.url for result in report.dead}
    return tuple(r for r in catalogue.resources if r.url not in dead)
