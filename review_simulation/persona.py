"""Utilities for loading and representing review-derived personas."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class VehicleAffinity:
    """Represents a liked or disliked vehicle configuration."""

    make: Optional[str]
    model: Optional[str]
    year: Optional[int]
    condition: Optional[str]
    rationale: Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> "VehicleAffinity":
        return cls(
            make=data.get("make"),
            model=data.get("model"),
            year=data.get("year"),
            condition=(data.get("condition") or None),
            rationale=data.get("rationale"),
        )


@dataclass
class ReviewPersona:
    """Persona constructed from a single enriched review entry."""

    make: str
    model: str
    review: str
    rating: Optional[float]
    date: Optional[str]
    liked: List[VehicleAffinity] = field(default_factory=list)
    disliked: List[VehicleAffinity] = field(default_factory=list)
    intention: str = ""
    mentioned_makes: List[str] = field(default_factory=list)
    mentioned_models: List[str] = field(default_factory=list)
    mentioned_years: List[int] = field(default_factory=list)
    preferred_condition: Optional[str] = None
    newness_preference_score: Optional[int] = None
    newness_preference_notes: Optional[str] = None
    preferred_vehicle_type: Optional[str] = None
    preferred_fuel_type: Optional[str] = None
    alternative_openness: Optional[int] = None
    misc_notes: Optional[str] = None
    upper_price_limit: Optional[float] = None

    @property
    def rating_value(self) -> Optional[float]:
        try:
            return float(self.rating) if self.rating is not None else None
        except (TypeError, ValueError):
            return None


def _parse_affinity_column(value: str) -> List[VehicleAffinity]:
    if not value or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        raw_list = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(raw_list, dict):
        raw_list = [raw_list]
    affinities: List[VehicleAffinity] = []
    for item in raw_list:
        if isinstance(item, dict):
            affinities.append(VehicleAffinity.from_dict(item))
    return affinities


def _parse_json_list(value: Optional[str]) -> List[str]:
    if not value or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(data, list):
        return [str(item) for item in data if item is not None and str(item).strip()]
    if isinstance(data, (str, int, float)) and str(data).strip():
        return [str(data)]
    return []


def _parse_json_int_list(value: Optional[str]) -> List[int]:
    items = _parse_json_list(value)
    parsed: List[int] = []
    for item in items:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed


def _try_parse_int(value: Optional[object]) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _try_parse_float(value: Optional[object]) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_personas_from_frame(frame: pd.DataFrame) -> List[ReviewPersona]:
    personas: List[ReviewPersona] = []
    for _, row in frame.iterrows():
        personas.append(
            ReviewPersona(
                make=str(row.get("Make", "")),
                model=str(row.get("Model", "")),
                review=str(row.get("Review", "")),
                rating=row.get("ratings"),
                date=row.get("date"),
                liked=_parse_affinity_column(row.get("liked_options", "")),
                disliked=_parse_affinity_column(row.get("disliked_options", "")),
                intention=str(row.get("user_intention", "")),
                mentioned_makes=_parse_json_list(row.get("mentioned_makes")),
                mentioned_models=_parse_json_list(row.get("mentioned_models")),
                mentioned_years=_parse_json_int_list(row.get("mentioned_years")),
                preferred_condition=(row.get("preferred_condition") or None),
                newness_preference_score=_try_parse_int(row.get("newness_preference_score")),
                newness_preference_notes=(row.get("newness_preference_notes") or None),
                preferred_vehicle_type=(row.get("preferred_vehicle_type") or None),
                preferred_fuel_type=(row.get("preferred_fuel_type") or None),
                alternative_openness=_try_parse_int(row.get("openness_to_alternatives")),
                misc_notes=(row.get("misc_notes") or None),
                upper_price_limit=_try_parse_float(row.get("upper_price_limit")),
            )
        )
    return personas


def load_personas(csv_path: Path) -> List[ReviewPersona]:
    df = pd.read_csv(csv_path)
    return load_personas_from_frame(df)
