"""Model pricing and cost accounting.

Every call's cost is computed here from its actual token usage and written to
``llm_calls``, so "what does this feature cost to run" is a query rather than an
estimate.

Cached tokens are priced separately, and the difference is large enough to dominate:
a cache read is a tenth of the input rate, a cache write is 1.25x. A system that
collapsed them into one number would report a well-cached feature as expensive and an
uncached one as cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pathwise.ai.providers.base import TokenUsage
from pathwise.logging_config import get_logger

log = get_logger(__name__)

#: A cache read costs about a tenth of a fresh input token.
CACHE_READ_MULTIPLIER: Final = 0.1
#: Writing to the cache costs about 1.25x, which is why caching only pays off when
#: the prefix is actually reused.
CACHE_WRITE_MULTIPLIER: Final = 1.25

_PER_MILLION: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-million-token rates for one model, in USD."""

    input_per_mtok: float
    output_per_mtok: float

    def cost_for(self, usage: TokenUsage) -> float:
        """Total USD for one call's usage."""
        input_rate = self.input_per_mtok / _PER_MILLION
        output_rate = self.output_per_mtok / _PER_MILLION
        return (
            usage.input_tokens * input_rate
            + usage.output_tokens * output_rate
            + usage.cache_read_tokens * input_rate * CACHE_READ_MULTIPLIER
            + usage.cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        )


#: Anthropic first-party rates. Keep in sync when adding a model to configuration —
#: `test_every_configured_model_is_priced` fails if you forget, which is the point.
PRICING: Final[dict[str, ModelPricing]] = {
    "claude-opus-5": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
    "claude-opus-4-8": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
    "claude-sonnet-5": ModelPricing(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00),
    "claude-fable-5-1": ModelPricing(input_per_mtok=10.00, output_per_mtok=50.00),
    # The deterministic test provider. Priced at zero so test runs do not pollute
    # cost dashboards with imaginary spend.
    "fake-model": ModelPricing(input_per_mtok=0.0, output_per_mtok=0.0),
}


def pricing_for(model: str) -> ModelPricing | None:
    """Look up a model's rates, or ``None`` if it is not in the table."""
    return PRICING.get(model)


def estimate_cost(model: str, usage: TokenUsage) -> float:
    """Cost in USD for one call.

    An unpriced model returns 0.0 and logs a warning rather than raising: a missing
    price is an accounting gap, not a reason to fail a user's request. The warning
    and the test above are what stop the gap from going unnoticed.
    """
    pricing = pricing_for(model)
    if pricing is None:
        log.warning("model_not_in_pricing_table", model=model, tokens=usage.total_tokens)
        return 0.0
    return pricing.cost_for(usage)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Cost split by token class, for the AI usage dashboard.

    Splitting it out makes the cache's value legible: ``cache_savings`` answers "what
    would this have cost with no caching?", which is the number that justifies the
    complexity of maintaining a stable prompt prefix.
    """

    model: str
    input_cost: float
    output_cost: float
    cache_read_cost: float
    cache_write_cost: float

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost

    @property
    def cache_savings(self) -> float:
        """What the cached reads would have cost at the full input rate, less what
        they did cost. Zero when nothing was served from cache."""
        if self.cache_read_cost <= 0:
            return 0.0
        full_price = self.cache_read_cost / CACHE_READ_MULTIPLIER
        return full_price - self.cache_read_cost


def breakdown(model: str, usage: TokenUsage) -> CostBreakdown:
    """Cost split by token class."""
    pricing = pricing_for(model) or ModelPricing(0.0, 0.0)
    input_rate = pricing.input_per_mtok / _PER_MILLION
    output_rate = pricing.output_per_mtok / _PER_MILLION
    return CostBreakdown(
        model=model,
        input_cost=usage.input_tokens * input_rate,
        output_cost=usage.output_tokens * output_rate,
        cache_read_cost=usage.cache_read_tokens * input_rate * CACHE_READ_MULTIPLIER,
        cache_write_cost=usage.cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER,
    )


def format_usd(amount: float) -> str:
    """Render a cost readably.

    Individual calls are fractions of a cent, so a plain two-decimal format would
    display every one of them as "$0.00" and make the dashboard useless.
    """
    if amount == 0:
        return "$0.00"
    if amount < 0.01:
        return f"${amount:.6f}"
    if amount < 1:
        return f"${amount:.4f}"
    return f"${amount:.2f}"
