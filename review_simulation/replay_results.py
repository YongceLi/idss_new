"""Recreate review simulation UI output from an exported CSV file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from review_simulation.exporter import RESULT_EXPORT_COLUMNS
from review_simulation.persona import ReviewPersona, load_personas_from_frame
from review_simulation.simulation import (
    AttributeJudgement,
    AttributeSatisfaction,
    FollowUpExchange,
    FollowUpJudgement,
    PersonaTurn,
    SimulationMetrics,
    SimulationResult,
    VehicleJudgement,
)
from review_simulation.ui import compute_final_stats, render_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV produced by review_simulation.run --export")
    parser.add_argument(
        "--metric-k",
        type=int,
        default=20,
        help="k value for metric display (defaults to 20)",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional path to persist aggregated stats as JSON",
    )
    return parser.parse_args()


def _parse_confidence(value: object) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _attribute_judgement_from_export(value: dict | None) -> AttributeJudgement:
    if value is None:
        return AttributeJudgement(satisfied=None, rationale=None)

    satisfied = value.get("satisfied") if isinstance(value, dict) else None
    mentioned = value.get("mentioned") if isinstance(value, dict) else None
    if mentioned is False:
        satisfied_value = None
    else:
        satisfied_value = satisfied

    rationale = value.get("rationale") if isinstance(value, dict) else None
    return AttributeJudgement(satisfied=satisfied_value, rationale=rationale)


def _followup_exchanges_from_export(value: str) -> List[FollowUpExchange]:
    if not value or (isinstance(value, float) and pd.isna(value)):
        return []
    raw = json.loads(value)
    exchanges: List[FollowUpExchange] = []
    for entry in raw or []:
        judgement_raw = entry.get("judgement") or {}
        exchanges.append(
            FollowUpExchange(
                question=entry.get("question", ""),
                quick_replies=entry.get("quick_replies"),
                answer=entry.get("answer", ""),
                judgement=FollowUpJudgement(
                    relevance_satisfied=judgement_raw.get("relevance_satisfied", False),
                    relevance_confidence=_parse_confidence(
                        judgement_raw.get("relevance_confidence")
                    ),
                    relevance_rationale=judgement_raw.get("relevance_rationale", ""),
                    newness_satisfied=judgement_raw.get("newness_satisfied", False),
                    newness_confidence=_parse_confidence(
                        judgement_raw.get("newness_confidence")
                    ),
                    newness_rationale=judgement_raw.get("newness_rationale", ""),
                ),
            )
        )
    return exchanges


def _load_results(frame: pd.DataFrame) -> List[SimulationResult]:
    personas: List[ReviewPersona] = load_personas_from_frame(frame)
    turns: List[PersonaTurn] = []

    for _, row in frame.iterrows():
        turns.append(
            PersonaTurn(
                message=str(row.get("persona_query", "")),
                writing_style=str(row.get("persona_writing_style", "")),
                interaction_style=str(row.get("persona_interaction_style", "")),
                family_background=str(row.get("persona_family_background", "")),
                goal_summary=str(row.get("persona_goal_summary", "")),
                upper_price_limit=row.get("persona_upper_price_limit"),
            )
        )

    results: List[SimulationResult] = []
    for persona, turn, (_, row) in zip(personas, turns, frame.iterrows()):
        vehicle_judgements = json.loads(row.get("vehicle_judgements", "[]"))
        vehicles: List[VehicleJudgement] = []
        for vehicle in vehicle_judgements:
            attribute_results = {
                key: _attribute_judgement_from_export(value)
                for key, value in (vehicle.get("attribute_results") or {}).items()
            }
            vehicles.append(
                VehicleJudgement(
                    index=vehicle.get("index"),
                    make=vehicle.get("make"),
                    model=vehicle.get("model"),
                    year=vehicle.get("year"),
                    condition=vehicle.get("condition"),
                    location=vehicle.get("location"),
                    vin=vehicle.get("vin"),
                    price=vehicle.get("price"),
                    satisfied=vehicle.get("satisfied", False),
                    rationale=vehicle.get("rationale", ""),
                    attribute_results=attribute_results,
                    confidence=_parse_confidence(vehicle.get("confidence")),
                )
            )

        attribute_stats_raw = json.loads(row.get("attribute_satisfaction_at_k") or "{}")
        attribute_satisfaction = {
            key: AttributeSatisfaction(
                satisfied_count=value.get("satisfied_count", 0),
                total_count=value.get("total_count", 0),
            )
            for key, value in attribute_stats_raw.items()
        }

        metrics = SimulationMetrics(
            precision_at_k=row.get("precision_at_k"),
            precision_at_k_confident=row.get("precision_at_k_confident"),
            satisfied_count=int(row.get("satisfied_in_top_k", 0)),
            infra_list_diversity=row.get("infra_list_diversity"),
            ndcg_at_k=row.get("ndcg_at_k"),
            ndcg_at_k_confident=row.get("ndcg_at_k_confident"),
            attribute_satisfaction=attribute_satisfaction,
        )

        recommendation_response = {
            "diversification_dimension": row.get("diversification_dimension"),
            "filters_extracted": json.loads(row.get("filters_extracted") or "null"),
            "preferences_extracted": json.loads(row.get("preferences_extracted") or "null"),
            "bucket_labels": json.loads(row.get("bucket_labels") or "null"),
            "sql_query": row.get("sql_query"),
            "overall_attribute_satisfaction": row.get("overall_attribute_satisfaction"),
        }
        conversation_history = json.loads(row.get("conversation_history") or "[]")
        follow_up_exchanges = _followup_exchanges_from_export(
            row.get("follow_up_exchanges") or "[]"
        )

        results.append(
            SimulationResult(
                persona=persona,
                persona_turn=turn,
                vehicles=vehicles,
                metrics=metrics,
                recommendation_response=recommendation_response,
                summary=row.get("summary", ""),
                conversation_history=conversation_history,
                follow_up_exchanges=follow_up_exchanges,
            )
        )
    return results


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    missing_columns = [col for col in RESULT_EXPORT_COLUMNS if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"Input CSV missing expected columns: {', '.join(missing_columns)}")

    results = _load_results(frame)
    render_results(results, args.metric_k)

    if args.stats_output:
        stats = compute_final_stats(results, args.metric_k)
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
