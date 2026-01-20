"""Console rendering helpers for the review simulation mode."""
from __future__ import annotations

import json
import statistics
from typing import Iterable, Optional

from rich.console import Console
from rich.table import Table

from review_simulation.simulation import AttributeJudgement, AttributeSatisfaction, SimulationResult


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _format_price(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"${value:,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _format_attribute_result(outcome: Optional[AttributeJudgement]) -> str:
    if outcome is None or outcome.satisfied is None:
        return "—"
    return "✅" if outcome.satisfied else "❌"


def _format_followup_score(satisfied: bool, confidence: Optional[float]) -> str:
    status = "✅" if satisfied else "❌"
    if confidence is None:
        return status
    return f"{status} ({confidence:.2f})"


def compute_final_stats(results: Iterable[SimulationResult], metric_k: int) -> dict:
    collected = list(results)
    precision_vals = [r.metrics.precision_at_k for r in collected if r.metrics.precision_at_k is not None]
    precision_confident_vals = [
        r.metrics.precision_at_k_confident for r in collected if r.metrics.precision_at_k_confident is not None
    ]
    infra_vals = [r.metrics.infra_list_diversity for r in collected if r.metrics.infra_list_diversity is not None]
    ndcg_vals = [r.metrics.ndcg_at_k for r in collected if r.metrics.ndcg_at_k is not None]
    ndcg_confident_vals = [
        r.metrics.ndcg_at_k_confident for r in collected if r.metrics.ndcg_at_k_confident is not None
    ]
    satisfied_rates = []
    if metric_k:
        for r in collected:
            satisfied_rates.append(r.metrics.satisfied_count / metric_k)

    attribute_totals: dict[str, AttributeSatisfaction] = {}
    for result in collected:
        for name, stats in result.metrics.attribute_satisfaction.items():
            current = attribute_totals.get(name) or AttributeSatisfaction(0, 0)
            current.satisfied_count += stats.satisfied_count
            current.total_count += stats.total_count
            attribute_totals[name] = current

    attribute_rates = {name: stats.rate for name, stats in attribute_totals.items() if stats.total_count}

    def _avg(values: list[float]) -> Optional[float]:
        return statistics.mean(values) if values else None

    return {
        "precision_at_k": _avg(precision_vals),
        "precision_at_k_confident": _avg(precision_confident_vals),
        "infra_list_diversity": _avg(infra_vals),
        "ndcg_at_k": _avg(ndcg_vals),
        "ndcg_at_k_confident": _avg(ndcg_confident_vals),
        "satisfied_at_k": _avg(satisfied_rates),
        "attribute_satisfaction_rates": attribute_rates,
        "overall_attribute_satisfaction": _avg(list(attribute_rates.values())),
    }


def render_results(results: Iterable[SimulationResult], metric_k: int) -> None:
    console = Console()
    collected_results = list(results)
    final_stats = compute_final_stats(collected_results, metric_k)
    for idx, result in enumerate(collected_results, start=1):
        persona = result.persona
        turn = result.persona_turn
        console.rule(f"Persona {idx}: {persona.make} {persona.model}")
        console.print(f"[bold]Writing style:[/bold] {turn.writing_style}")
        console.print(f"[bold]Interaction style:[/bold] {turn.interaction_style}")
        console.print(f"[bold]Family background:[/bold] {turn.family_background}")
        console.print(f"[bold]Goal summary:[/bold] {turn.goal_summary}")
        console.print(
            "[bold]Preferences:[/bold] "
            f"Makes {', '.join(persona.mentioned_makes) if persona.mentioned_makes else 'unspecified'} | "
            f"Models {', '.join(persona.mentioned_models) if persona.mentioned_models else 'unspecified'} | "
            f"Years {', '.join(str(year) for year in persona.mentioned_years) if persona.mentioned_years else 'unspecified'} | "
            f"Condition {persona.preferred_condition or 'unspecified'} | "
            f"Newness {persona.newness_preference_score or 'unknown'} ({persona.newness_preference_notes or 'no context'}) | "
            f"Type {persona.preferred_vehicle_type or 'unspecified'} | "
            f"Fuel {persona.preferred_fuel_type or 'unspecified'} | "
            f"Openness {persona.alternative_openness or 'unknown'} | "
            f"Notes {persona.misc_notes or 'none'}"
        )
        console.print()
        console.print(f"[bold]User turn:[/bold] {turn.message}")
        diversification_dimension = result.recommendation_response.get("diversification_dimension")
        extracted_filters = result.recommendation_response.get("filters_extracted")
        implicit_preferences = result.recommendation_response.get("preferences_extracted")
        bucket_labels = result.recommendation_response.get("bucket_labels")
        if diversification_dimension:
            console.print(
                "[bold]Diversification dimension:[/bold] "
                + json.dumps(diversification_dimension)
            )
        if bucket_labels:
            console.print("[bold]Bucket labels:[/bold] " + json.dumps(bucket_labels))
        if extracted_filters:
            console.print("[bold]Extracted filters:[/bold] " + json.dumps(extracted_filters))
        if implicit_preferences:
            console.print("[bold]Implicit preferences:[/bold] " + json.dumps(implicit_preferences))
        console.print(f"[bold]Summary:[/bold] {result.summary}")
        console.print()

        if result.follow_up_exchanges:
            followup_table = Table(show_lines=False)
            followup_table.add_column("Question", overflow="fold")
            followup_table.add_column("Quick replies", overflow="fold")
            followup_table.add_column("Answer", overflow="fold")
            followup_table.add_column("Relevance")
            followup_table.add_column("Newness")
            followup_table.add_column("Relevance rationale", overflow="fold")
            followup_table.add_column("Newness rationale", overflow="fold")
            for exchange in result.follow_up_exchanges:
                judgement = exchange.judgement
                followup_table.add_row(
                    exchange.question,
                    ", ".join(exchange.quick_replies or []),
                    exchange.answer,
                    _format_followup_score(
                        judgement.relevance_satisfied, judgement.relevance_confidence
                    ),
                    _format_followup_score(
                        judgement.newness_satisfied, judgement.newness_confidence
                    ),
                    judgement.relevance_rationale,
                    judgement.newness_rationale,
                )
            console.print(followup_table)
            console.print()

        table = Table(show_lines=False)
        table.add_column("#", justify="right")
        table.add_column("Make")
        table.add_column("Model")
        table.add_column("Year", justify="right")
        table.add_column("Condition")
        table.add_column("Price", justify="right")
        table.add_column("Satisfied")
        table.add_column("Rationale", overflow="fold")
        table.add_column("Price match", justify="center")
        table.add_column("Condition match", justify="center")
        table.add_column("Year match", justify="center")
        table.add_column("Make match", justify="center")
        table.add_column("Model match", justify="center")
        table.add_column("Fuel match", justify="center")
        table.add_column("Body match", justify="center")
        table.add_column("Misc match", justify="center")
        table.add_column("Confidence", justify="right")

        for vehicle in result.vehicles:
            satisfaction = "✅" if vehicle.satisfied else "❌"
            attributes = vehicle.attribute_results or {}
            price_match = _format_attribute_result(attributes.get("price"))
            condition_match = _format_attribute_result(attributes.get("condition"))
            year_match = _format_attribute_result(attributes.get("year"))
            make_match = _format_attribute_result(attributes.get("make"))
            model_match = _format_attribute_result(attributes.get("model"))
            fuel_match = _format_attribute_result(attributes.get("fuel_type"))
            body_match = _format_attribute_result(attributes.get("body_type"))
            misc_match = _format_attribute_result(attributes.get("all_misc"))
            confidence_text = _format_metric(vehicle.confidence) if vehicle.confidence is not None else "—"
            table.add_row(
                str(vehicle.index),
                str(vehicle.make or "?"),
                str(vehicle.model or "?"),
                str(vehicle.year or "-"),
                str(vehicle.condition or "-"),
                _format_price(vehicle.price),
                satisfaction,
                str(vehicle.rationale or ""),
                price_match,
                condition_match,
                year_match,
                make_match,
                model_match,
                fuel_match,
                body_match,
                misc_match,
                confidence_text,
            )
        console.print(table)
        console.print(
            f"Precision@{metric_k}: {_format_metric(result.metrics.precision_at_k)} | "
            f"Precision@{metric_k} (conf>0.6): {_format_metric(result.metrics.precision_at_k_confident)} | "
            f"Infra-list diversity: {_format_metric(result.metrics.infra_list_diversity)} | "
            f"NDCG@{metric_k}: {_format_metric(result.metrics.ndcg_at_k)} | "
            f"NDCG@{metric_k} (conf>0.6): {_format_metric(result.metrics.ndcg_at_k_confident)} | "
            f"Satisfied in top {metric_k}: {result.metrics.satisfied_count}"
        )
        if result.metrics.attribute_satisfaction:
            attr_parts = []
            for key, stats in sorted(result.metrics.attribute_satisfaction.items()):
                rate = _format_metric(stats.rate)
                attr_parts.append(f"{key}: {rate} ({stats.satisfied_count}/{stats.total_count})")
            console.print("Attribute satisfied@k: " + " | ".join(attr_parts))
        console.print()

    console.rule("Simulation complete")
    console.print("[bold]Final averages across personas:[/bold]")
    console.print(
        f"Precision@{metric_k}: {_format_metric(final_stats['precision_at_k'])} | "
        f"Precision@{metric_k} (conf>0.6): {_format_metric(final_stats['precision_at_k_confident'])} | "
        f"Infra-list diversity: {_format_metric(final_stats['infra_list_diversity'])} | "
        f"NDCG@{metric_k}: {_format_metric(final_stats['ndcg_at_k'])} | "
        f"NDCG@{metric_k} (conf>0.6): {_format_metric(final_stats['ndcg_at_k_confident'])} | "
        f"Satisfied@{metric_k}: {_format_metric(final_stats['satisfied_at_k'])}"
    )
    if final_stats["attribute_satisfaction_rates"]:
        attr_parts = []
        for key, rate in sorted(final_stats["attribute_satisfaction_rates"].items()):
            attr_parts.append(f"{key}: {_format_metric(rate)}")
        console.print("Attribute satisfied@k averages: " + " | ".join(attr_parts))
    if final_stats.get("overall_attribute_satisfaction") is not None:
        console.print(
            f"Overall attribute satisfied@k: {_format_metric(final_stats['overall_attribute_satisfaction'])}"
        )
