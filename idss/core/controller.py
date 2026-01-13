"""
Main IDSS Controller.

Orchestrates the interview and recommendation flow with configurable k parameter.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from idss.utils.logger import get_logger
from idss.core.config import IDSSConfig, get_config
from idss.data.vehicle_store import LocalVehicleStore
from idss.parsing.semantic_parser import (
    parse_user_input,
    merge_filters,
    merge_preferences,
    ParsedInput
)
from idss.interview.question_generator import (
    generate_question,
    generate_recommendation_intro,
    QuestionResponse
)
from idss.diversification.entropy import (
    select_diversification_dimension,
    compute_entropy_report
)
from idss.diversification.bucketing import diversify_with_entropy_bucketing
from idss.recommendation.embedding_similarity import rank_with_embedding_similarity
from idss.recommendation.coverage_risk import rank_with_coverage_risk

logger = get_logger("core.controller")


@dataclass
class SessionState:
    """State for an IDSS session."""
    explicit_filters: Dict[str, Any] = field(default_factory=dict)
    implicit_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)
    question_count: int = 0


@dataclass
class IDSSResponse:
    """Response from the IDSS controller."""
    response_type: str  # 'question' or 'recommendations'
    message: str
    quick_replies: Optional[List[str]] = None
    recommendations: Optional[List[List[Dict[str, Any]]]] = None  # 2D grid
    bucket_labels: Optional[List[str]] = None
    diversification_dimension: Optional[str] = None
    filters_extracted: Optional[Dict[str, Any]] = None
    preferences_extracted: Optional[Dict[str, Any]] = None


class IDSSController:
    """
    Main controller for the Simplified IDSS.

    Handles the interview → recommendation flow with:
    - Configurable k (number of questions)
    - LLM-based impatience detection
    - Entropy-based diversification
    """

    def __init__(self, config: Optional[IDSSConfig] = None):
        """
        Initialize the controller.

        Args:
            config: Configuration object. Uses default config if not provided.
        """
        self.config = config or get_config()
        self.state = SessionState()
        self.store = LocalVehicleStore(require_photos=True)

        logger.info(f"IDSS Controller initialized: k={self.config.k}, method={self.config.recommendation_method}")

    def process_input(self, user_message: str) -> IDSSResponse:
        """
        Process user input and return appropriate response.

        Args:
            user_message: The user's message

        Returns:
            IDSSResponse with either a question or recommendations
        """
        logger.info(f"Processing input: {user_message[:100]}...")

        # Step 1: Parse the user input
        parsed = parse_user_input(
            user_message=user_message,
            conversation_history=self.state.conversation_history,
            existing_filters=self.state.explicit_filters,
            question_count=self.state.question_count
        )

        # Step 2: Update state with extracted information
        self._update_state(user_message, parsed)

        # Step 3: Decide whether to ask a question or recommend
        should_recommend = self._should_recommend(parsed)

        if should_recommend:
            return self._generate_recommendations()
        else:
            return self._generate_question()

    def get_recommendations(self) -> IDSSResponse:
        """
        Force recommendation generation (bypass interview).

        Returns:
            IDSSResponse with recommendations
        """
        return self._generate_recommendations()

    def reset_session(self) -> None:
        """Reset the session state for a new conversation."""
        self.state = SessionState()
        logger.info("Session reset")

    def _update_state(self, user_message: str, parsed: ParsedInput) -> None:
        """Update session state with parsed information."""
        # Merge filters
        self.state.explicit_filters = merge_filters(
            self.state.explicit_filters,
            parsed.explicit_filters
        )

        # Merge preferences
        self.state.implicit_preferences = merge_preferences(
            self.state.implicit_preferences,
            parsed.implicit_preferences
        )

        # Add to conversation history
        self.state.conversation_history.append({
            "role": "user",
            "content": user_message
        })

    def _should_recommend(self, parsed: ParsedInput) -> bool:
        """
        Determine if we should generate recommendations now.

        Returns True if:
        - k=0 (direct recommendation mode)
        - We've asked k questions already
        - User is impatient or explicitly wants recommendations
        """
        # k=0 mode: always recommend immediately
        if self.config.k == 0:
            logger.info("k=0 mode: generating recommendations immediately")
            return True

        # Hit question limit
        if self.state.question_count >= self.config.k:
            logger.info(f"Hit question limit (k={self.config.k}): generating recommendations")
            return True

        # User impatience detected
        if parsed.is_impatient:
            logger.info("Impatience detected: generating recommendations")
            return True

        # User explicitly wants recommendations
        if parsed.wants_recommendations:
            logger.info("User requested recommendations: generating recommendations")
            return True

        return False

    def _generate_question(self) -> IDSSResponse:
        """Generate a clarifying question."""
        question_response = generate_question(
            conversation_history=self.state.conversation_history,
            explicit_filters=self.state.explicit_filters,
            implicit_preferences=self.state.implicit_preferences,
            questions_asked=self.state.questions_asked
        )

        # Update state
        self.state.question_count += 1
        self.state.questions_asked.append(question_response.topic)
        self.state.conversation_history.append({
            "role": "assistant",
            "content": question_response.question
        })

        logger.info(f"Generated question #{self.state.question_count}: {question_response.question}")

        return IDSSResponse(
            response_type="question",
            message=question_response.question,
            quick_replies=question_response.quick_replies,
            filters_extracted=self.state.explicit_filters,
            preferences_extracted=self.state.implicit_preferences
        )

    def _generate_recommendations(self) -> IDSSResponse:
        """Generate recommendations with ranking method + entropy-based diversification."""
        logger.info("Generating recommendations...")
        logger.info(f"Filters: {self.state.explicit_filters}")
        logger.info(f"Preferences: {self.state.implicit_preferences}")

        # Step 1: Get candidate vehicles from database (larger pool for ranking)
        candidates = self._get_candidates(limit=500)

        if not candidates:
            return IDSSResponse(
                response_type="recommendations",
                message="I couldn't find any vehicles matching your criteria. Try broadening your search.",
                recommendations=[],
                bucket_labels=[],
                filters_extracted=self.state.explicit_filters,
                preferences_extracted=self.state.implicit_preferences
            )

        logger.info(f"Found {len(candidates)} candidate vehicles from SQL")

        # Step 2: Rank candidates using configured method
        ranked_candidates = self._rank_candidates(candidates)
        logger.info(f"Ranked to {len(ranked_candidates)} candidates using {self.config.recommendation_method}")

        # Step 3: Log entropy report for analysis
        entropy_report = compute_entropy_report(ranked_candidates)
        logger.info(f"Entropy report: {entropy_report}")

        # Step 4: Select diversification dimension (based on ranked candidates)
        div_dimension = select_diversification_dimension(
            candidates=ranked_candidates,
            explicit_filters=self.state.explicit_filters
        )

        # Step 5: Bucket vehicles using entropy-based diversification
        buckets, bucket_labels, _ = diversify_with_entropy_bucketing(
            vehicles=ranked_candidates,
            dimension=div_dimension,
            n_rows=self.config.num_rows,
            n_per_row=self.config.n_vehicles_per_row
        )

        # Step 5: Generate introduction message
        intro_message = generate_recommendation_intro(
            explicit_filters=self.state.explicit_filters,
            implicit_preferences=self.state.implicit_preferences,
            diversification_dimension=div_dimension,
            bucket_labels=bucket_labels
        )

        # Add to conversation history
        self.state.conversation_history.append({
            "role": "assistant",
            "content": intro_message
        })

        return IDSSResponse(
            response_type="recommendations",
            message=intro_message,
            recommendations=buckets,
            bucket_labels=bucket_labels,
            diversification_dimension=div_dimension,
            filters_extracted=self.state.explicit_filters,
            preferences_extracted=self.state.implicit_preferences
        )

    def _get_candidates(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get candidate vehicles from the database.

        Args:
            limit: Maximum number of candidates to retrieve

        Returns:
            List of vehicle dictionaries
        """
        # Clean filters for database query
        db_filters = {}
        for key, value in self.state.explicit_filters.items():
            if value is not None:
                db_filters[key] = value

        # If no filters, add a default year range to avoid returning everything
        if not db_filters:
            db_filters['year'] = '2018-2025'
            logger.info("No filters specified, using default year range: 2018-2025")

        try:
            candidates = self.store.search_listings(
                filters=db_filters,
                limit=limit,
                order_by="price",
                order_dir="ASC"
            )
            return candidates
        except Exception as e:
            logger.error(f"Failed to get candidates: {e}")
            return []

    def _rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank candidates using configured recommendation method.

        Flow:
        1. SQL Filter → Candidates (500+)
        2. Embedding Similarity or Coverage-Risk Ranking → Top-100 ranked by relevance
        3. Returns ranked candidates for entropy bucketing

        Args:
            candidates: Raw candidates from SQL query

        Returns:
            Ranked list of candidates
        """
        method = self.config.recommendation_method
        top_k = 100  # Rank down to top 100 for entropy bucketing

        if method == "embedding_similarity":
            # Embedding Similarity: Dense Vector + MMR
            logger.info(f"Ranking with Embedding Similarity (Dense + MMR)...")
            ranked = rank_with_embedding_similarity(
                vehicles=candidates,
                explicit_filters=self.state.explicit_filters,
                implicit_preferences=self.state.implicit_preferences,
                top_k=top_k,
                lambda_param=self.config.embedding_similarity_lambda_param,
                use_mmr=True
            )
        elif method == "coverage_risk":
            # Coverage-Risk Optimization
            logger.info(f"Ranking with Coverage-Risk Optimization...")
            ranked = rank_with_coverage_risk(
                vehicles=candidates,
                explicit_filters=self.state.explicit_filters,
                implicit_preferences=self.state.implicit_preferences,
                top_k=top_k,
                lambda_risk=self.config.coverage_risk_lambda_risk,
                mode=self.config.coverage_risk_mode,
                tau=self.config.coverage_risk_tau,
                alpha=self.config.coverage_risk_alpha
            )
        else:
            # Fallback: no ranking, just use first top_k
            logger.warning(f"Unknown method '{method}', returning unranked candidates")
            ranked = candidates[:top_k]

        return ranked


def create_controller(k: Optional[int] = None, **kwargs) -> IDSSController:
    """
    Factory function to create a controller with custom settings.

    Args:
        k: Number of questions to ask (overrides config)
        **kwargs: Other config parameters to override

    Returns:
        Configured IDSSController
    """
    config = get_config()

    # Override with provided parameters
    if k is not None:
        config.k = k
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return IDSSController(config)
