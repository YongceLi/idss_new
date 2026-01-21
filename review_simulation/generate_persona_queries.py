#!/usr/bin/env python3
"""Generate persona prompts and queries for downstream evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from langchain_openai import ChatOpenAI

from review_simulation.exporter import PERSONA_EXPORT_COLUMNS, persona_to_row
from review_simulation.persona import ReviewPersona, load_personas_from_frame
from review_simulation.simulation import build_persona_turn, build_persona_turn_short


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV produced by review_enricher.py")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to store the generated persona dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Chat model to use for persona synthesis",
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        default=None,
        help="Optional cap on number of personas processed",
    )

    parser.add_argument(
        "--short",
        action="store_true",
        help="Use the shortened persona prompt variant",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    personas: List[ReviewPersona] = load_personas_from_frame(frame)
    if args.max_personas is not None:
        personas = personas[: args.max_personas]

    llm = ChatOpenAI(model=args.model, temperature=0.4)

    rows = []
    for persona in personas:
        if args.short:
            turn = build_persona_turn_short(persona, llm)
        else:
            turn = build_persona_turn(persona, llm)
        rows.append(persona_to_row(persona, turn))

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows, columns=PERSONA_EXPORT_COLUMNS)
    frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
