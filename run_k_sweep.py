#!/usr/bin/env python3
"""Sweep k values and ablation configurations, save results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from review_simulation.persona import ReviewPersona, load_personas_from_frame
from review_simulation.simulation import PersonaTurn, SimulationResult, evaluate_persona
from review_simulation.ui import compute_final_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-sweep and ablation evaluation")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_eval/review_queries_car_v2.csv"),
        help="CSV file with persona queries",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="coverage_risk",
        help="Recommendation method (default: coverage_risk)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results_k_sweep.txt"),
        help="Output file for results",
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        default=None,
        help="Limit number of personas to test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Chat model for persona synthesis",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="0,1,2,3",
        help="Comma-separated k values to test (default: 0,1,2,3)",
    )
    # Ablation flags
    parser.add_argument(
        "--no-mmr",
        action="store_true",
        help="Disable MMR diversification (ablation)",
    )
    parser.add_argument(
        "--no-entropy-bucketing",
        action="store_true",
        help="Disable entropy-based bucketing (ablation)",
    )
    parser.add_argument(
        "--no-progressive-relaxation",
        action="store_true",
        help="Disable progressive filter relaxation (ablation)",
    )
    parser.add_argument(
        "--no-entropy-questions",
        action="store_true",
        help="Disable entropy-based question selection (ablation)",
    )
    parser.add_argument(
        "--ablation-sweep",
        action="store_true",
        help="Run full ablation sweep (test all combinations)",
    )
    return parser.parse_args()


def load_personas_and_turns(csv_path: Path) -> List[tuple]:
    frame = pd.read_csv(csv_path)
    personas = load_personas_from_frame(frame)
    turns = []
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
    return list(zip(personas, turns))


def format_stats(stats: dict, metric_k: int, k: int) -> str:
    """Format stats into the desired output format."""
    lines = []

    # Main metrics line
    def _fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    main_line = (
        f"Precision@{metric_k}: {_fmt(stats['precision_at_k'])} | "
        f"Precision@{metric_k} (conf>0.6): {_fmt(stats['precision_at_k_confident'])} | "
        f"Infra-list diversity: {_fmt(stats['infra_list_diversity'])} | "
        f"NDCG@{metric_k}: {_fmt(stats['ndcg_at_k'])} | "
        f"NDCG@{metric_k} (conf>0.6): {_fmt(stats['ndcg_at_k_confident'])} | "
        f"Satisfied@{metric_k}: {_fmt(stats['satisfied_at_k'])}"
    )
    lines.append(main_line)

    # Follow-up line (only if k > 0)
    if k > 0 and stats.get("follow_up_relevance_rate") is not None:
        followup_line = (
            f"Follow-up relevance avg: {_fmt(stats['follow_up_relevance_rate'])} | "
            f"Follow-up newness avg: {_fmt(stats['follow_up_newness_rate'])}"
        )
        lines.append(followup_line)

    # Attribute satisfaction line
    if stats.get("attribute_satisfaction_rates"):
        attr_parts = []
        for key, rate in sorted(stats["attribute_satisfaction_rates"].items()):
            attr_parts.append(f"{key}: {_fmt(rate)}")
        lines.append("Attribute satisfied@k averages: " + " | ".join(attr_parts))

    # Overall attribute satisfaction
    if stats.get("overall_attribute_satisfaction") is not None:
        lines.append(f"Overall attribute satisfied@k: {_fmt(stats['overall_attribute_satisfaction'])}")

    return "\n".join(lines)


def run_evaluation(
    persona_pairs: List[tuple],
    llm: ChatOpenAI,
    method: str,
    k: int,
    use_mmr: bool,
    use_entropy: bool,
    use_relaxation: bool,
    use_entropy_questions: bool = True,
    metric_k: int = 9,
) -> dict:
    """Run evaluation with specific configuration and return stats."""
    results: List[SimulationResult] = []

    config_desc = f"k={k}, mmr={use_mmr}, entropy={use_entropy}, relax={use_relaxation}, entropy_q={use_entropy_questions}"

    for persona, turn in tqdm(persona_pairs, desc=config_desc):
        result = evaluate_persona(
            persona,
            turn,
            llm,
            recommendation_limit=9,
            metric_k=metric_k,
            recommendation_method=method,
            k=k,
            n_per_row=3,
            confidence_threshold=0.51,
            max_assessment_attempts=3,
            use_mmr_diversification=use_mmr,
            use_entropy_bucketing=use_entropy,
            use_progressive_relaxation=use_relaxation,
            use_entropy_questions=use_entropy_questions,
        )
        results.append(result)

    return compute_final_stats(results, metric_k)


def main() -> None:
    args = parse_args()

    # Load personas
    persona_pairs = load_personas_and_turns(args.input)
    if args.max_personas:
        persona_pairs = persona_pairs[:args.max_personas]

    llm = ChatOpenAI(model=args.model, temperature=0.4)
    metric_k = 9

    # Parse k values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    results_text = []

    if args.ablation_sweep:
        # Full ablation sweep: test all combinations
        ablation_configs = [
            {"mmr": True, "entropy": True, "relax": True, "entropy_q": True, "name": "Full"},
            {"mmr": False, "entropy": True, "relax": True, "entropy_q": True, "name": "No MMR"},
            {"mmr": True, "entropy": False, "relax": True, "entropy_q": True, "name": "No Entropy Bucketing"},
            {"mmr": True, "entropy": True, "relax": False, "entropy_q": True, "name": "No Progressive Relaxation"},
            {"mmr": True, "entropy": True, "relax": True, "entropy_q": False, "name": "No Entropy Questions"},
            {"mmr": False, "entropy": False, "relax": True, "entropy_q": True, "name": "No Diversification"},
            {"mmr": False, "entropy": False, "relax": False, "entropy_q": False, "name": "Minimal (No Div, No Relax, No Entropy Q)"},
        ]

        for config in ablation_configs:
            results_text.append(f"=" * 60)
            results_text.append(f"Configuration: {config['name']}")
            results_text.append(f"  MMR: {config['mmr']}, Entropy: {config['entropy']}, Relaxation: {config['relax']}, Entropy Questions: {config['entropy_q']}")
            results_text.append(f"=" * 60)
            results_text.append("")

            for k in k_values:
                print(f"\n{'='*60}")
                print(f"Running {config['name']} with k={k}, method={args.method}")
                print(f"{'='*60}")

                stats = run_evaluation(
                    persona_pairs,
                    llm,
                    args.method,
                    k,
                    use_mmr=config["mmr"],
                    use_entropy=config["entropy"],
                    use_relaxation=config["relax"],
                    use_entropy_questions=config["entropy_q"],
                    metric_k=metric_k,
                )

                k_text = f"k = {k}\n\nFinal averages across personas:\n"
                k_text += format_stats(stats, metric_k, k)
                results_text.append(k_text)
                results_text.append("")

                print(f"\n{k_text}\n")

            results_text.append("")
    else:
        # Standard k-sweep with optional ablation flags
        use_mmr = not args.no_mmr
        use_entropy = not args.no_entropy_bucketing
        use_relaxation = not args.no_progressive_relaxation
        use_entropy_questions = not args.no_entropy_questions

        config_desc = []
        if not use_mmr:
            config_desc.append("no-mmr")
        if not use_entropy:
            config_desc.append("no-entropy")
        if not use_relaxation:
            config_desc.append("no-relaxation")
        if not use_entropy_questions:
            config_desc.append("no-entropy-questions")
        config_name = ", ".join(config_desc) if config_desc else "full"

        results_text.append(f"Method: {args.method}")
        results_text.append(f"Configuration: {config_name}")
        results_text.append(f"MMR: {use_mmr}, Entropy Bucketing: {use_entropy}, Progressive Relaxation: {use_relaxation}, Entropy Questions: {use_entropy_questions}")
        results_text.append("")

        for k in k_values:
            print(f"\n{'='*60}")
            print(f"Running k={k} with method={args.method}")
            print(f"{'='*60}")

            stats = run_evaluation(
                persona_pairs,
                llm,
                args.method,
                k,
                use_mmr=use_mmr,
                use_entropy=use_entropy,
                use_relaxation=use_relaxation,
                use_entropy_questions=use_entropy_questions,
                metric_k=metric_k,
            )

            k_text = f"k = {k}\n\nFinal averages across personas:\n"
            k_text += format_stats(stats, metric_k, k)
            results_text.append(k_text)
            results_text.append("")

            print(f"\n{k_text}\n")

    # Write results to file
    with open(args.output, "w") as f:
        f.write("\n".join(results_text))

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
