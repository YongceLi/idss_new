"""Core logic for running review-driven single-turn simulations."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from idss import create_controller
from review_simulation.persona import ReviewPersona, VehicleAffinity


@dataclass
class PersonaTurn:
    """Describes the simulated single-turn message for a persona."""

    message: str
    writing_style: str
    interaction_style: str
    family_background: str
    goal_summary: str
    upper_price_limit: Optional[float] = None


@dataclass
class AttributeJudgement:
    """Result of assessing a specific vehicle attribute."""

    satisfied: Optional[bool]
    rationale: Optional[str]


@dataclass
class VehicleJudgement:
    """Result of assessing a recommended vehicle."""

    index: int
    make: str
    model: str
    year: Optional[int]
    condition: str
    location: str
    vin: Optional[str]
    price: Optional[float]
    satisfied: bool
    rationale: str
    attribute_results: Dict[str, AttributeJudgement]
    confidence: Optional[float]


@dataclass
class SimulationMetrics:
    precision_at_k: Optional[float]
    precision_at_k_confident: Optional[float]
    satisfied_count: int
    infra_list_diversity: Optional[float]
    ndcg_at_k: Optional[float]
    ndcg_at_k_confident: Optional[float]
    attribute_satisfaction: Dict[str, "AttributeSatisfaction"]


@dataclass
class AttributeSatisfaction:
    """Aggregate satisfaction data for a single attribute."""

    satisfied_count: int
    total_count: int

    @property
    def rate(self) -> Optional[float]:
        if self.total_count == 0:
            return None
        return self.satisfied_count / self.total_count


@dataclass
class SimulationResult:
    persona: ReviewPersona
    persona_turn: PersonaTurn
    vehicles: List[VehicleJudgement]
    metrics: SimulationMetrics
    recommendation_response: Dict[str, Any]
    summary: str
    conversation_history: List[Dict[str, str]]
    follow_up_exchanges: List["FollowUpExchange"]


@dataclass
class FollowUpJudgement:
    relevance_satisfied: bool
    relevance_confidence: Optional[float]
    relevance_rationale: str
    newness_satisfied: bool
    newness_confidence: Optional[float]
    newness_rationale: str


@dataclass
class FollowUpExchange:
    question: str
    quick_replies: Optional[List[str]]
    answer: str
    judgement: FollowUpJudgement


class PersonaDraft(BaseModel):
    writing_style: str
    interaction_style: str
    family_background: str
    goal_summary: str
    user_message: str
    upper_price_limit: Optional[float]


class AttributeAssessment(BaseModel):
    satisfied: Optional[bool] = Field(
        None, description="Whether this attribute matches the persona preferences (None if not mentioned)"
    )
    rationale: Optional[str] = Field(
        None, description="<=10 words explaining the decision or None if not evaluated"
    )


class VehicleAssessment(BaseModel):
    index: int = Field(..., description="1-based index of the vehicle in the presented list")
    satisfied: bool = Field(..., description="Whether the persona would be satisfied overall")
    rationale: str = Field(..., description="Short textual explanation")
    confidence: Optional[float] = Field(
        None,
        description="Confidence in the overall satisfaction judgement between 0 and 1",
        ge=0.0,
        le=1.0,
    )
    price: Optional[AttributeAssessment] = None
    condition: Optional[AttributeAssessment] = None
    year: Optional[AttributeAssessment] = None
    make: Optional[AttributeAssessment] = None
    model: Optional[AttributeAssessment] = None
    fuel_type: Optional[AttributeAssessment] = None
    body_type: Optional[AttributeAssessment] = None
    all_misc: Optional[AttributeAssessment] = None


class VehicleAssessmentList(BaseModel):
    assessments: List[VehicleAssessment]


class SummaryResponse(BaseModel):
    satisfied_summary: str = Field(
        ..., description="One-sentence reason the persona would be satisfied with the list"
    )
    unsatisfied_summary: str = Field(
        ..., description="One-sentence reason the persona would be unsatisfied with the list"
    )


class YearPreferenceExtraction(BaseModel):
    year_mentioned: Optional[int] = Field(
        None, description="The exact year mentioned in the query or null"
    )
    year_sentiment: str = Field(
        ..., description="One of older/newer/around/exact describing the direction"
    )
    year_alternative: bool = Field(
        ..., description="Whether the shopper is open to years outside the preference"
    )


class VehiclePreferenceExtraction(BaseModel):
    make_like: List[str]
    make_dislike: List[str]
    make_alternative: bool
    model_like: List[str]
    model_dislike: List[str]
    model_alternative: bool
    fuel_type_preference: List[str]
    fuel_type_alternative: bool
    body_type_preference: List[str]
    body_type_alternative: bool


class FollowUpJudgeResponse(BaseModel):
    relevance_satisfied: bool = Field(
        ..., description="Whether the follow-up question is relevant to the persona and history"
    )
    relevance_confidence: float = Field(
        ..., description="Confidence 0-1 for relevance judgement"
    )
    relevance_rationale: str = Field(
        ..., description="Short rationale (<=20 words) for relevance"
    )
    newness_satisfied: bool = Field(
        ..., description="Whether the follow-up question asks for new information"
    )
    newness_confidence: float = Field(
        ..., description="Confidence 0-1 for newness judgement"
    )
    newness_rationale: str = Field(
        ..., description="Short rationale (<=20 words) for newness"
    )


YEAR_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract year preferences from a shopper's single-turn query.",
        ),
        (
            "human",
            """
Persona query: {persona_query}

Identify explicit preferences and respond with JSON containing:
- year_mentioned: The exact year as an integer if stated, otherwise null.
- year_sentiment: Return 4 values (older, newer, around, exact). This means that the query shows the year to be older than the year mentioned (example: I want the find vehicles that is older than 2025), newer (example: I want the find vehicles that is newer than 2025), or around  (example: I want the find vehicles that is around than 2025), or exact (example: I want the find a 2025 model vehicles.). If there is no mention, return around. One of [older, newer, around, exact]. Default to around when no direction is given.
- year_alternative: Whether the user are fine with options that are not aligned with their year requirement or not. True if the query allows years outside the stated intent; false otherwise.
""",
        ),
    ]
)


VEHICLE_PREFERENCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract explicit make/model/fuel/body preferences from a query.",
        ),
        (
            "human",
            """
Persona query: {persona_query}

Identify explicit preferences and respond with JSON containing:

- condition_like: List the preferred conditions (new/used) that are explicitly mentioned. DO NOT infer preferred conditions beyond what is directly stated in the query.
- condition_alternative: Whether the user are fine with options that are not aligned with their condition like or not. True if alternatives outside condition_like are acceptable; false otherwise. Phrases like "open to both new or used" indicate true.
- make_like: List the preferred makes (empty if none) that are explicitly mentioned. DO NOT infer preferred makes beyond what is directly stated in the query.
- make_dislike: List the makes to avoid (empty if none) that are explicitly mentioned. DO NOT infer makes to avoid beyond what is directly stated in the query.
- make_alternative: Whether the user are fine with options that are not aligned with their make like or not. True if alternatives outside make_like are acceptable; false otherwise. Phrases like "open to other makes" or "open to options" or "something similar" indicate true.
- model_like: List the preferred models (empty if none) that are explicitly mentioned. DO NOT infer preferred models beyond what is directly stated in the query.
- model_dislike: List the models to avoid (empty if none) that are explicitly mentioned. DO NOT infer models to avoid beyond what is directly stated in the query.
- model_alternative: Whether the user are fine with options that are not aligned with their model like or not. True if alternatives outside model_like are acceptable; false otherwise. Phrases like "open to other models" or "open to options" or "something similar" indicate true.
- fuel_type_preference: List the preferred fuel types exactly as stated (empty if none).
- fuel_type_alternative: Whether the user are fine with options that are not aligned with their fuel type like or not. True if other fuel types are acceptable; false otherwise. Phrases like "open to other fuel types" or "open to options" indicate true.
- body_type_preference: List the preferred body types exactly as stated (empty if none).
- body_type_alternative: Whether the user are fine with options that are not aligned with their body type like or not. True if other body types are acceptable; false otherwise. Phrases like "open to other body types" or "open to options" indicate true.

When the query is silent about flexibility, set the corresponding alternative flag to True.
""",
        ),
    ]
)


PERSONA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You craft single-turn user utterances for a car recommendation demo."
            "Read the reviewer's background and produce a concise, natural query."
            "Every generated query must restate the shopper's concrete preferences."
            "Identify the highest price the shopper would pay and include it.",
        ),
        (
            "human",
            """
Review summary:
Make/Model owned or discussed: {make} {model}
Rating given: {rating}
Review excerpt: "{review}"
Likes: {likes}
Dislikes: {dislikes}
Stated intention: {intention}
Mentioned makes: {mentioned_makes}
Mentioned models: {mentioned_models}
Mentioned years: {mentioned_years}
Preferred condition: {preferred_condition}
Newness preference (1-10): {newness_preference_score} — {newness_preference_notes}
Preferred vehicle type: {preferred_vehicle_type}
Preferred fuel type: {preferred_fuel_type}
Openness to alternatives (1-10): {openness_to_alternatives}
Other priorities: {misc_notes}
Current year is 2025 (assume this for the most newness context).

Create a JSON object with keys writing_style, interaction_style, family_background,
goal_summary, upper_price_limit, and user_message. The user_message must be the exact text the
persona will send to the assistant in a single turn. It should reflect their
writing style and refer to their family/life context when appropriate. Keep it
under 120 words and avoid lists/bullets. In details:
- writing_style: A brief description of the user's writing style.
- interaction_style: A brief description of how the user prefers to interact.
- family_background: A brief summary of the user's family/life context relevant to car buying.
- goal_summary: A concise summary of the user's goal when interacting with a car recommendation agent.
- upper_price_limit: Your best estimate of the shopper's maximum acceptable price in USD (numbers only). Use null if unknown.
- user_message: Must clearly mention the desired make/model (if any), relevant years, whether the car should be new or used, how
  new they want the search to be, body style, preferred fuel type, willingness to consider alternatives, the maximum price to pay,
  and any additional priorities highlighted above.
""",
        ),
    ]
)


ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You evaluate car recommendations for a simulated shopper."
            "Decide if each vehicle matches the persona's expressed likes/dislikes."
            "Only use make, model, year, condition (new/used), body style, fuel type, and price to judge."
            "All vehicle information provided is accurate. DO NOT override them with assumptions. Use it to make informed judgements."
            "Return concise rationales (10 words or fewer).",
        ),
        (
            "human",
            """
Persona query: {persona_query}

Vehicle preference extraction:
{vehicle_preferences}

Current year is 2025 (assume this for the most newness context).
Do NOT make assumptions beyond the provided information. Only use the data given to make your judgements.
Do NOT assume the modernity of the vehicle beyond the year provided (The closer the year to 2025, the newer and more modern the vehicle).
Do NOT make assumptions about the make or model beyond what is explicitly given.
Only consider the information given in the Persona Query and the vehicle details provided to evaluate satisfaction.
Use the rest of the information only as a guideline, as they might not reflect the actual preferences expressed in the Persona Query.
Use the provided year judgement included with each vehicle entry instead of recalculating year fit.
Use the provided price judgement included with each vehicle entry instead of recalculating year fit.

For each vehicle below decide if the persona would be satisfied. Judge only the
criteria explicitly mentioned in the persona_query; if a criterion was not
mentioned, return null for that criterion. Respond with JSON using this shape:
{{"assessments": [{{"index": <number>, "satisfied": <bool>, "rationale": <string>, "confidence": <float 0-1>,
"price": {{...}}, "condition": {{...}}, "year": {{...}},
"make": {{...}}, "model": {{...}}, "fuel_type": {{...}}, "body_type": {{...}}, "all_misc": {{...}}}}, ...]}}.
In details, for each attribute, to determine overall satisfaction, consider:
condition: (whether the condition satisfies the users' preference in the query or not.
Compare Preferred conditions (condition_like) + Open to other conditions (condition_alternative) with the condition of the vehicle to determine this.
If the users are fine with both new and used or do not have preferences (condition_alternative = True), return True).
make: (whether the make satisfies the users' preference in the query or not. 
Compare Preferred makes (make_like) + Avoided makes (make_dislike) + Open to other makes (make_alternative) with the make of the vehicle to determine this.
If the users are open to other makes (make_alternative = True), return True.).
model: (whether the model satisfies the users' preference in the query or not. 
Compare Preferred models (model_like) + Avoided models (model_dislike) + Open to other models (model_alternative) with the model of the vehicle to determine this.
If the users are open to other models (model_alternative = True), return True.).
fuel_type: (whether the Fuel Type satisfies the users' preference in the query or not.
Compare Preferred fuel types (fuel_type_like) + Open to other fuel types (fuel_type_alternative) with the fuel_type of the vehicle to determine this.
If the users are open to other fuel types (fuel_type_alternative = True), return True.
Do NOT assume fuel efficiency in here, only compare fuel_type.
For example, if the query wants Gasoline, and the fuel_type returns Gasoline,
Premium Unleaded, E85, or any other type of Gasoline, return True. Return None if there is no mention of fuel type in the query).
body_type: (whether the body type satisfies the users' preference in the query or not, return None if there is no mention of body type in the query. 
Compare Preferred body types (body_type_like) + Open to other body types (body_type_alternative) with the body_type of the vehicle to determine this.
If the users are fine with alternatives (body_type_alternative = True), return True.).
all_misc: (whether others' preference of the users mentioned in the query satisfies the users' query or not. The preference checks are safetyness and luxury.
. Only return value if there is an obvious mismatch with the highest confidence level, else return None. Example of obvious mismatch: user wants family SUV, but return industrial trucks.).
For satisfied: only return true/false for each attribute when persona_query mentions it; otherwise set
that attribute to null. Use the provided price judgement included with each vehicle entry instead of recalculating price fit.

Confidence should be a number between 0 and 1 indicating how confident you are in the overall satisfaction judgement.
If there are a lot of conflicts between attributes and the final satisfaction judgement, the confidence score should be lower.
Vice versa, if there are a lot of attributes that align with the final satisfaction judgement, the confidence score should be higher.
The delta of confidence should be proportional to the number of attributes that align with the final satisfaction judgement.
Vehicles:
{vehicles}
""",
        ),
    ]
)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You condense vehicle satisfaction rationales into concise summaries."
            "Return JSON with two short sentences describing why the persona is satisfied or unsatisfied."
        ),
        (
            "human",
            """
Satisfied rationales:
{satisfied_reasons}

Unsatisfied rationales:
{unsatisfied_reasons}

Respond with JSON {{"satisfied_summary": <string>, "unsatisfied_summary": <string>}}.
Each value must be a single short sentence (no more than 20 words).
If there are no reasons, respond None.
""",
        ),
    ]
)


FOLLOW_UP_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge the quality of a follow-up question asked by a car recommendation assistant."
            "Evaluate relevance to the persona query and conversation history, and whether it asks for new information.",
        ),
        (
            "human",
            """
Persona query:
{persona_query}

Conversation history (most recent last):
{conversation_history}

Follow-up question:
{follow_up_question}

Return JSON with:
- relevance_satisfied: true/false
- relevance_confidence: 0-1
- relevance_rationale: <=20 words
- newness_satisfied: true/false (true if it asks for information not already stated)
- newness_confidence: 0-1
- newness_rationale: <=20 words
""",
        ),
    ]
)


FOLLOW_UP_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are simulating a user persona answering a follow-up question about car preferences."
            "Respond in the persona's writing style and interaction style. Be concise and natural.",
        ),
        (
            "human",
            """
Persona writing style: {persona_writing_style}
Persona interaction style: {persona_interaction_style}
Persona family background: {persona_family_background}
Persona goal summary: {persona_goal_summary}
Persona query: {persona_query}

Conversation history (most recent last):
{conversation_history}

Follow-up question:
{follow_up_question}

Quick replies offered: {quick_replies}

Answer the follow-up question as the persona. Use quick replies if appropriate, but you may rephrase.
""",
        ),
    ]
)


def _affinities_to_text(items: List[VehicleAffinity]) -> str:
    if not items:
        return "None"
    parts = []
    for item in items:
        attributes = []
        if item.make:
            attributes.append(item.make)
        if item.model:
            attributes.append(item.model)
        if item.year:
            attributes.append(str(item.year))
        if item.condition:
            attributes.append(item.condition)
        summary = " ".join(attributes) if attributes else "unspecified vehicle"
        if item.rationale:
            summary += f" ({item.rationale})"
        parts.append(summary)
    return "; ".join(parts)


def _list_to_text(values: List[str]) -> str:
    if not values:
        return "None"
    return ", ".join(values)


def _format_vehicle_entry(vehicle: dict, index: int) -> Dict[str, Optional[str]]:
    car = vehicle.get("vehicle", {}) if isinstance(vehicle, dict) else {}
    listing = vehicle.get("retailListing", {}) if isinstance(vehicle, dict) else {}
    build = vehicle.get("build", {}) if isinstance(vehicle, dict) else {}

    make = car.get("make")
    model = car.get("model")
    year = car.get("year")
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    used_flag = car.get("norm_is_used")
    if used_flag is None:
        used_flag = vehicle.get("norm_is_used")
    condition = "used" if used_flag else "new"

    city = listing.get("city") or vehicle.get("dealer_city")
    state = listing.get("state") or vehicle.get("dealer_state")
    location = ", ".join([part for part in [city, state] if part]) or "Unknown"

    price = listing.get("price")
    if price is None:
        price = listing.get("list_price") or vehicle.get("price")
    try:
        price = float(price) if price is not None else None
    except (TypeError, ValueError):
        price = None
    miles = listing.get("miles")
    fuel_type = car.get("norm_fuel_type") or build.get("fuel_type")
    body_type = car.get("norm_body_type") or build.get("body_type")


    return {
        "index": index,
        "make": make,
        "model": model,
        "year": year,
        "condition": condition,
        "location": location,
        "vin": vehicle.get("vin") or car.get("vin"),
        "price": price,
        "miles": miles,
        "fuel_type": fuel_type,
        "body_type": body_type,
    }


def _determine_price_judgement(
    upper_price_limit: Optional[float], vehicle_price: Optional[float]
) -> AttributeJudgement:
    """Return a deterministic price satisfaction judgement.

    If either value is missing, no judgement is made. Otherwise satisfied is
    True when the persona's extracted price ceiling is greater than or equal to
    the vehicle's price.
    """

    if upper_price_limit is None or vehicle_price is None:
        return AttributeJudgement(satisfied=None, rationale=None)

    if upper_price_limit >= vehicle_price:
        return AttributeJudgement(
            satisfied=True,
            rationale=f"Within budget: {vehicle_price} <= {upper_price_limit}",
        )

    return AttributeJudgement(
        satisfied=False,
        rationale=f"Over budget: {vehicle_price} > {upper_price_limit}",
    )


def _extract_year_preferences(
    persona_query: str, model: ChatOpenAI
) -> YearPreferenceExtraction:
    structured_model = model.with_structured_output(YearPreferenceExtraction)
    prompt = YEAR_EXTRACTION_PROMPT.format_prompt(persona_query=persona_query)
    return structured_model.invoke(prompt.to_messages())


def _determine_year_judgement(
    year_preferences: YearPreferenceExtraction, vehicle_year: Optional[int]
) -> AttributeJudgement:
    if year_preferences.year_alternative:
        return AttributeJudgement(
            satisfied=True, rationale="User open to year alternatives"
        )

    if year_preferences.year_mentioned is None or vehicle_year is None:
        return AttributeJudgement(satisfied=None, rationale=None)

    target_year = year_preferences.year_mentioned
    sentiment = (year_preferences.year_sentiment or "").lower()

    if sentiment == "older":
        satisfied = vehicle_year < target_year
        rationale = f"Vehicle year {vehicle_year} is older than {target_year}"
    elif sentiment == "newer":
        satisfied = vehicle_year > target_year
        rationale = f"Vehicle year {vehicle_year} is newer than {target_year}"
    elif sentiment == "exact":
        satisfied = vehicle_year == target_year
        rationale = f"Vehicle year {vehicle_year} matches {target_year}"
    else:
        satisfied = target_year - 4 < vehicle_year < target_year + 4
        rationale = f"Vehicle year {vehicle_year} within ±4 of {target_year}"

    return AttributeJudgement(satisfied=satisfied, rationale=rationale)


def _extract_vehicle_preferences(
    persona_query: str, model: ChatOpenAI
) -> VehiclePreferenceExtraction:
    structured_model = model.with_structured_output(VehiclePreferenceExtraction)
    prompt = VEHICLE_PREFERENCE_PROMPT.format_prompt(persona_query=persona_query)
    return structured_model.invoke(prompt.to_messages())


def build_persona_turn(persona: ReviewPersona, model: ChatOpenAI) -> PersonaTurn:
    structured_model = model.with_structured_output(PersonaDraft)
    likes_text = _affinities_to_text(persona.liked)
    dislikes_text = _affinities_to_text(persona.disliked)
    rating_text = persona.rating_value if persona.rating_value is not None else "unknown"

    prompt = PERSONA_PROMPT.format_prompt(
        make=persona.make,
        model=persona.model,
        rating=rating_text,
        review=persona.review,
        likes=likes_text,
        dislikes=dislikes_text,
        intention=persona.intention or "",
        mentioned_makes=_list_to_text(persona.mentioned_makes),
        mentioned_models=_list_to_text(persona.mentioned_models),
        mentioned_years=_list_to_text([str(year) for year in persona.mentioned_years]),
        preferred_condition=persona.preferred_condition or "unspecified",
        newness_preference_score=persona.newness_preference_score or "unknown",
        newness_preference_notes=persona.newness_preference_notes or "",
        preferred_vehicle_type=persona.preferred_vehicle_type or "unspecified",
        preferred_fuel_type=persona.preferred_fuel_type or "unspecified",
        openness_to_alternatives=persona.alternative_openness or "unknown",
        misc_notes=persona.misc_notes or "None stated",
    )
    draft = structured_model.invoke(prompt.to_messages())
    return PersonaTurn(
        message=draft.user_message.strip(),
        writing_style=draft.writing_style.strip(),
        interaction_style=draft.interaction_style.strip(),
        family_background=draft.family_background.strip(),
        goal_summary=draft.goal_summary.strip(),
        upper_price_limit=draft.upper_price_limit,
    )


def _assess_vehicles(
    persona: ReviewPersona,
    persona_turn: PersonaTurn,
    vehicles: List[dict],
    model: ChatOpenAI,
) -> List[VehicleJudgement]:
    structured_model = model.with_structured_output(VehicleAssessmentList)
    year_preferences = _extract_year_preferences(persona_turn.message, model)
    vehicle_preferences = _extract_vehicle_preferences(persona_turn.message, model)

    vehicle_entries = []
    price_judgements: Dict[int, AttributeJudgement] = {}
    year_judgements: Dict[int, AttributeJudgement] = {}
    for idx, vehicle in enumerate(vehicles):
        entry = _format_vehicle_entry(vehicle, idx + 1)
        price_judgement = _determine_price_judgement(
            persona_turn.upper_price_limit, entry.get("price")
        )
        year_judgement = _determine_year_judgement(
            year_preferences, entry.get("year")
        )
        entry["price_judgement"] = {
            "satisfied": price_judgement.satisfied,
            "rationale": price_judgement.rationale,
        }
        entry["year_judgement"] = {
            "satisfied": year_judgement.satisfied,
            "rationale": year_judgement.rationale,
        }
        vehicle_entries.append(entry)
        price_judgements[entry["index"]] = price_judgement
        year_judgements[entry["index"]] = year_judgement
    prompt = ASSESSMENT_PROMPT.format_prompt(
        persona_query=persona_turn.message,
        year_preferences=year_preferences.model_dump_json(indent=2),
        vehicle_preferences=vehicle_preferences.model_dump_json(indent=2),
        vehicles=json.dumps(vehicle_entries, indent=2),
    )
    response = structured_model.invoke(prompt.to_messages())

    assessment_map = {item.index: item for item in response.assessments}

    results: List[VehicleJudgement] = []
    attribute_keys = [
        "price",
        "condition",
        "year",
        "make",
        "model",
        "fuel_type",
        "body_type",
        "all_misc",
    ]

    def _attribute_judgement(value: Optional[AttributeAssessment]) -> AttributeJudgement:
        if value is None:
            return AttributeJudgement(satisfied=None, rationale=None)
        return AttributeJudgement(
            satisfied=value.satisfied,
            rationale=value.rationale.strip() if value.rationale else None,
        )

    for entry in vehicle_entries:
        assessment = assessment_map.get(entry["index"])
        if assessment is None:
            # Default to dissatisfaction if the model omitted an entry.
            assessment = VehicleAssessment(index=entry["index"], satisfied=False, rationale="No evaluation")
        attribute_results = {
            key: _attribute_judgement(getattr(assessment, key)) for key in attribute_keys
        }
        attribute_results["price"] = price_judgements.get(
            entry["index"], AttributeJudgement(satisfied=None, rationale=None)
        )
        attribute_results["year"] = year_judgements.get(
            entry["index"], AttributeJudgement(satisfied=None, rationale=None)
        )
        results.append(
            VehicleJudgement(
                index=entry["index"],
                make=entry.get("make"),
                model=entry.get("model"),
                year=entry.get("year"),
                condition=str(entry.get("condition")),
                location=str(entry.get("location")),
                vin=entry.get("vin"),
                price=entry.get("price"),
                satisfied=assessment.satisfied,
                rationale=assessment.rationale.strip(),
                attribute_results=attribute_results,
                confidence=assessment.confidence,
            )
        )
    return results


def _compute_metrics(judgements: List[VehicleJudgement], persona: ReviewPersona, k: int) -> SimulationMetrics:
    _ = persona
    if k <= 0:
        return SimulationMetrics(
            precision_at_k=None,
            precision_at_k_confident=None,
            satisfied_count=0,
            infra_list_diversity=None,
            ndcg_at_k=None,
            ndcg_at_k_confident=None,
            attribute_satisfaction={},
        )

    top_k = [j for j in judgements if j.index <= k]
    if not top_k:
        return SimulationMetrics(
            precision_at_k=None,
            precision_at_k_confident=None,
            satisfied_count=0,
            infra_list_diversity=None,
            ndcg_at_k=None,
            ndcg_at_k_confident=None,
            attribute_satisfaction={},
        )

    satisfied = [j for j in top_k if j.satisfied]
    precision = len(satisfied) / len(top_k)

    confident_top_k = [j for j in top_k if j.confidence is not None and j.confidence > 0.6]
    if confident_top_k:
        confident_satisfied = [j for j in confident_top_k if j.satisfied]
        precision_confident = len(confident_satisfied) / len(confident_top_k)
    else:
        precision_confident = None

    make_models = [
        (str(j.make).strip().lower() if j.make else "", str(j.model).strip().lower() if j.model else "")
        for j in top_k
    ]
    unique_make_models = {item for item in make_models if any(item)}
    infra_list_diversity = None
    infra_list_diversity = len(unique_make_models) / len(top_k)

    def _dcg(items: List[VehicleJudgement]) -> float:
        value = 0.0
        for idx, judgement in enumerate(items, start=1):
            rel = 1.0 if judgement.satisfied else 0.0
            if rel == 0.0:
                continue
            value += (2 ** rel - 1) / math.log2(idx + 1)
        return value

    dcg = _dcg(top_k)
    ideal_count = min(len(satisfied), len(top_k))
    ideal_items = top_k[:]
    ideal_items.sort(key=lambda item: item.satisfied, reverse=True)
    idcg = _dcg(ideal_items[:ideal_count])
    if idcg:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0 if not satisfied else None

    confident_dcg = _dcg(confident_top_k)
    confident_ideal_count = min(
        len(confident_top_k), len([j for j in confident_top_k if j.satisfied])
    )
    confident_ideal_items = confident_top_k[:]
    confident_ideal_items.sort(key=lambda item: item.satisfied, reverse=True)
    confident_idcg = _dcg(confident_ideal_items[:confident_ideal_count])
    if confident_idcg:
        ndcg_confident = confident_dcg / confident_idcg
    elif confident_top_k:
        ndcg_confident = 0.0 if not confident_satisfied else None
    else:
        ndcg_confident = None

    attribute_satisfaction: Dict[str, AttributeSatisfaction] = {}
    for judgement in top_k:
        for attribute, outcome in judgement.attribute_results.items():
            if outcome.satisfied is None:
                continue
            current = attribute_satisfaction.get(attribute) or AttributeSatisfaction(
                satisfied_count=0, total_count=0
            )
            current.total_count += 1
            if outcome.satisfied:
                current.satisfied_count += 1
            attribute_satisfaction[attribute] = current

    return SimulationMetrics(
        precision_at_k=precision,
        precision_at_k_confident=precision_confident,
        satisfied_count=len(satisfied),
        infra_list_diversity=infra_list_diversity,
        ndcg_at_k=ndcg,
        ndcg_at_k_confident=ndcg_confident,
        attribute_satisfaction=attribute_satisfaction,
    )


def _summarize_judgements(
    judgements: List[VehicleJudgement], model: ChatOpenAI
) -> str:
    satisfied_reasons = [j.rationale for j in judgements if j.satisfied and j.rationale]
    unsatisfied_reasons = [j.rationale for j in judgements if not j.satisfied and j.rationale]

    if not satisfied_reasons and not unsatisfied_reasons:
        return "No rationales provided."

    structured_model = model.with_structured_output(SummaryResponse)
    satisfied_text = (
        "\n".join(f"- {reason}" for reason in satisfied_reasons) or "None"
    )
    unsatisfied_text = (
        "\n".join(f"- {reason}" for reason in unsatisfied_reasons) or "None"
    )
    prompt = SUMMARY_PROMPT.format_prompt(
        satisfied_reasons=satisfied_text,
        unsatisfied_reasons=unsatisfied_text,
    )
    response = structured_model.invoke(prompt.to_messages())

    parts: List[str] = []
    if response.satisfied_summary:
        parts.append(f"Satisfied: {response.satisfied_summary.strip()}")
    if response.unsatisfied_summary:
        parts.append(f"Unsatisfied: {response.unsatisfied_summary.strip()}")
    if not parts:
        return "Summary unavailable."
    return " | ".join(parts)


def _judge_follow_up_question(
    persona_turn: PersonaTurn,
    conversation_history: List[Dict[str, str]],
    follow_up_question: str,
    model: ChatOpenAI,
) -> FollowUpJudgement:
    structured_model = model.with_structured_output(FollowUpJudgeResponse)
    prompt = FOLLOW_UP_JUDGE_PROMPT.format_prompt(
        persona_query=persona_turn.message,
        conversation_history=json.dumps(conversation_history, indent=2),
        follow_up_question=follow_up_question,
    )
    response = structured_model.invoke(prompt.to_messages())
    return FollowUpJudgement(
        relevance_satisfied=response.relevance_satisfied,
        relevance_confidence=response.relevance_confidence,
        relevance_rationale=response.relevance_rationale.strip(),
        newness_satisfied=response.newness_satisfied,
        newness_confidence=response.newness_confidence,
        newness_rationale=response.newness_rationale.strip(),
    )


def _answer_follow_up_question(
    persona_turn: PersonaTurn,
    conversation_history: List[Dict[str, str]],
    follow_up_question: str,
    quick_replies: Optional[List[str]],
    model: ChatOpenAI,
) -> str:
    prompt = FOLLOW_UP_ANSWER_PROMPT.format_prompt(
        persona_writing_style=persona_turn.writing_style,
        persona_interaction_style=persona_turn.interaction_style,
        persona_family_background=persona_turn.family_background,
        persona_goal_summary=persona_turn.goal_summary,
        persona_query=persona_turn.message,
        conversation_history=json.dumps(conversation_history, indent=2),
        follow_up_question=follow_up_question,
        quick_replies=", ".join(quick_replies) if quick_replies else "None",
    )
    response = model.invoke(prompt.to_messages())
    return response.content.strip()


def run_simulation(
    persona: ReviewPersona,
    llm: ChatOpenAI,
    recommendation_limit: int = 20,
    metric_k: Optional[int] = None,
    recommendation_method: str = "embedding_similarity",
    k: int = 3,
    n_per_row: int = 3,
    confidence_threshold: float = 0.5,
    max_assessment_attempts: int = 3,
) -> SimulationResult:
    persona_turn = build_persona_turn(persona, llm)

    return evaluate_persona(
        persona,
        persona_turn,
        llm,
        recommendation_limit=recommendation_limit,
        metric_k=metric_k,
        recommendation_method=recommendation_method,
        k=k,
        n_per_row=n_per_row,
        confidence_threshold=confidence_threshold,
        max_assessment_attempts=max_assessment_attempts,
    )


def evaluate_persona(
    persona: ReviewPersona,
    persona_turn: PersonaTurn,
    llm: ChatOpenAI,
    recommendation_limit: int = 20,
    metric_k: Optional[int] = None,
    recommendation_method: str = "embedding_similarity",
    k: int = 3,
    n_per_row: int = 3,
    confidence_threshold: float = 0.5,
    max_assessment_attempts: int = 3,
) -> SimulationResult:
    controller = create_controller(
        k=k,
        n_vehicles_per_row=n_per_row,
        recommendation_method=recommendation_method,
    )
    response = controller.process_input(persona_turn.message)
    conversation_history: List[Dict[str, str]] = [
        {"role": "user", "content": persona_turn.message}
    ]
    follow_up_exchanges: List[FollowUpExchange] = []
    max_followups = max(1, k) + 2

    while response.response_type == "question" and len(follow_up_exchanges) < max_followups:
        conversation_history.append({"role": "assistant", "content": response.message})
        judgement = _judge_follow_up_question(
            persona_turn,
            conversation_history,
            response.message,
            llm,
        )
        answer = _answer_follow_up_question(
            persona_turn,
            conversation_history,
            response.message,
            response.quick_replies,
            llm,
        )
        conversation_history.append({"role": "user", "content": answer})
        follow_up_exchanges.append(
            FollowUpExchange(
                question=response.message,
                quick_replies=response.quick_replies,
                answer=answer,
                judgement=judgement,
            )
        )
        response = controller.process_input(answer)

    if response.response_type == "question":
        conversation_history.append({"role": "assistant", "content": response.message})

    if response.response_type == "recommendations":
        conversation_history.append({"role": "assistant", "content": response.message})

    recommendation_rows = response.recommendations or []
    vehicles = [item for row in recommendation_rows for item in row]
    vehicles = vehicles[:recommendation_limit]

    judgements = _assess_vehicles(persona, persona_turn, vehicles, llm)
    attempts: List[List[VehicleJudgement]] = [judgements]

    def _avg_confidence(items: List[VehicleJudgement]) -> Optional[float]:
        confidences = [j.confidence for j in items if j.confidence is not None]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    def _select_majority(attempt_sets: List[List[VehicleJudgement]]) -> List[VehicleJudgement]:
        if not attempt_sets:
            return []

        by_index: Dict[int, List[VehicleJudgement]] = {}
        for attempt in attempt_sets:
            for judgement in attempt:
                by_index.setdefault(judgement.index, []).append(judgement)

        final: List[VehicleJudgement] = []
        for index, entries in sorted(by_index.items()):
            true_count = sum(1 for entry in entries if entry.satisfied is True)
            false_count = sum(1 for entry in entries if entry.satisfied is False)
            if true_count > false_count:
                majority_value: Optional[bool] = True
            elif false_count > true_count:
                majority_value = False
            else:
                majority_value = None

            def _confidence_score(entry: VehicleJudgement) -> float:
                return entry.confidence if entry.confidence is not None else -1.0

            candidates = (
                [entry for entry in entries if entry.satisfied == majority_value]
                if majority_value is not None
                else entries
            )
            selected = max(candidates or entries, key=_confidence_score)
            final.append(selected)

        return sorted(final, key=lambda item: item.index)

    best_confidence = _avg_confidence(judgements)
    threshold_met = best_confidence is not None and best_confidence >= confidence_threshold

    remaining_attempts = max_assessment_attempts - 1
    while not threshold_met and remaining_attempts > 0:
        new_attempt = _assess_vehicles(persona, persona_turn, vehicles, llm)
        attempts.append(new_attempt)
        attempt_confidence = _avg_confidence(new_attempt)
        if attempt_confidence is not None and attempt_confidence >= confidence_threshold:
            judgements = new_attempt
            threshold_met = True
            best_confidence = attempt_confidence
            break
        if best_confidence is None or (
            attempt_confidence is not None and attempt_confidence > best_confidence
        ):
            judgements = new_attempt
            best_confidence = attempt_confidence
        remaining_attempts -= 1

    if not threshold_met:
        judgements = _select_majority(attempts)

    metrics = _compute_metrics(judgements, persona, metric_k or recommendation_limit)
    summary = _summarize_judgements(judgements, llm)

    return SimulationResult(
        persona=persona,
        persona_turn=persona_turn,
        vehicles=judgements,
        metrics=metrics,
        recommendation_response={
            "diversification_dimension": response.diversification_dimension,
            "filters_extracted": response.filters_extracted,
            "preferences_extracted": response.preferences_extracted,
            "bucket_labels": response.bucket_labels,
        },
        summary=summary,
        conversation_history=conversation_history,
        follow_up_exchanges=follow_up_exchanges,
    )
