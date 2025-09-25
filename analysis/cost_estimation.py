"""Cost estimation helpers built around Inspect eval logs.

Inspect's :func:`inspect_ai.eval` entrypoint returns :class:`inspect_ai.log.EvalLog`
objects that already include aggregated token usage for each model involved in a
run. This module provides small utilities to turn those logs into per-model
usage summaries and then translate the summaries into projected costs for larger
benchmark runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from inspect_ai.log import EvalLog
from inspect_ai.model._model_output import ModelUsage


@dataclass
class UsageSummary:
    """Aggregate token usage measured for a particular model."""

    model: str
    samples: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all samples."""

        return self.input_tokens + self.output_tokens

    @property
    def mean_input_tokens(self) -> float:
        """Average number of input tokens per completed sample."""

        return self.input_tokens / self.samples if self.samples else 0.0

    @property
    def mean_output_tokens(self) -> float:
        """Average number of output tokens per completed sample."""

        return self.output_tokens / self.samples if self.samples else 0.0

    @property
    def mean_total_tokens(self) -> float:
        """Average number of total tokens per completed sample."""

        return self.total_tokens / self.samples if self.samples else 0.0


@dataclass(frozen=True)
class ModelPricing:
    """Per-million token pricing for a language model."""

    provider: str
    model: str
    input_per_million: float
    output_per_million: float
    currency: str = "USD"


@dataclass(frozen=True)
class CostEstimate:
    """Projected spend for a benchmark run on a specific model."""

    provider: str
    model: str
    currency: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


def summarise_eval_log(log: EvalLog) -> UsageSummary | None:
    """Convert a single Inspect eval log into a :class:`UsageSummary`.

    Parameters
    ----------
    log:
        Evaluation log returned by :func:`inspect_ai.eval` or loaded from disk
        with :func:`inspect_ai.log.read_eval_log`.
    """

    samples = 0
    if log.results is not None:
        samples = log.results.completed_samples or log.results.total_samples or 0

    if samples == 0:
        return None

    usage = _primary_model_usage(log)
    if usage is None:
        return None

    return UsageSummary(
        model=log.eval.model,
        samples=samples,
        input_tokens=usage.input_tokens,
        output_tokens=_output_tokens(usage),
    )


def summarise_eval_logs(logs: Sequence[EvalLog]) -> Mapping[str, UsageSummary]:
    """Aggregate usage summaries for one or more eval logs."""

    summaries: dict[str, UsageSummary] = {}
    for log in logs:
        summary = summarise_eval_log(log)
        if summary is None:
            continue

        existing = summaries.get(summary.model)
        if existing is None:
            summaries[summary.model] = summary
        else:
            existing.samples += summary.samples
            existing.input_tokens += summary.input_tokens
            existing.output_tokens += summary.output_tokens

    return summaries


def scale_usage(summary: UsageSummary, target_games: int) -> UsageSummary:
    """Scale a usage summary to a target number of Hangman games."""

    if summary.samples == 0:
        raise ValueError("Cannot scale usage: no samples provided")

    factor = target_games / summary.samples
    return UsageSummary(
        model=summary.model,
        samples=target_games,
        input_tokens=int(round(summary.input_tokens * factor)),
        output_tokens=int(round(summary.output_tokens * factor)),
    )


def estimate_cost(summary: UsageSummary, pricing: ModelPricing) -> CostEstimate:
    """Estimate monetary cost given model pricing."""

    input_cost = (summary.input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (summary.output_tokens / 1_000_000) * pricing.output_per_million
    return CostEstimate(
        provider=pricing.provider,
        model=pricing.model,
        currency=pricing.currency,
        input_tokens=summary.input_tokens,
        output_tokens=summary.output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
    )


def project_costs(
    summaries: Mapping[str, UsageSummary] | Sequence[UsageSummary],
    target_games: int,
    pricings: Iterable[ModelPricing],
) -> list[CostEstimate]:
    """Estimate evaluation costs for multiple models."""

    if isinstance(summaries, Mapping):
        summary_map = dict(summaries)
    else:
        summary_map = {summary.model: summary for summary in summaries}

    estimates: list[CostEstimate] = []
    for pricing in pricings:
        summary = summary_map.get(pricing.model)
        if summary is None:
            raise KeyError(
                f"No usage summary available for model '{pricing.model}'. "
                "Call `summarise_eval_logs()` on your eval logs first."
            )
        scaled = scale_usage(summary, target_games)
        estimates.append(estimate_cost(scaled, pricing))

    return estimates


def project_costs_from_eval_logs(
    logs: Sequence[EvalLog],
    target_games: int,
    pricings: Iterable[ModelPricing],
) -> list[CostEstimate]:
    """Convenience wrapper around :func:`project_costs` for raw eval logs."""

    summaries = summarise_eval_logs(logs)
    return project_costs(summaries, target_games, pricings)


def _primary_model_usage(log: EvalLog) -> ModelUsage | None:
    """Extract aggregated usage for the primary model associated with a log."""

    if not log.stats or not log.stats.model_usage:
        return None

    primary = log.eval.model
    usage = log.stats.model_usage.get(primary)
    if usage is not None:
        return usage

    return _sum_usage(log.stats.model_usage.values())


def _sum_usage(usages: Iterable[ModelUsage]) -> ModelUsage:
    total = ModelUsage()
    for usage in usages:
        total = total + usage
    return total


def _output_tokens(usage: ModelUsage) -> int:
    if usage.output_tokens:
        return usage.output_tokens

    remainder = usage.total_tokens - usage.input_tokens
    return remainder if remainder > 0 else 0


__all__ = [
    "UsageSummary",
    "ModelPricing",
    "CostEstimate",
    "summarise_eval_log",
    "summarise_eval_logs",
    "scale_usage",
    "estimate_cost",
    "project_costs",
    "project_costs_from_eval_logs",
]
