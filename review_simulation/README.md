# Review Simulation

Tools for generating persona-based evaluation runs of the Interactive Decision Support System (IDSS). This workflow synthesizes personas from reviews, runs them through the IDSS controller, simulates follow-up Q&A, and scores the resulting recommendations and follow-up questions.

## Workflow Overview

1. **Generate personas and queries** from review data.
2. **Run the simulation** to:
   - Ask follow-up questions when the controller requests more info.
   - Use an LLM to answer those questions in the persona voice.
   - Judge the follow-up questions for relevance and newness.
   - Score recommendation quality (precision@k, NDCG@k, attribute satisfaction).
3. **Replay results** from exported CSVs for consistent reporting.

## Scripts and Responsibilities

### `generate_persona_queries.py`
Builds persona prompts and single-turn user queries (including inferred upper price limits) from enriched review CSVs.

**Usage**
```bash
python review_simulation/generate_persona_queries.py data/reviews_enriched.csv --output data/personas.csv
```

### `run.py`
Runs the full simulation against the IDSS controller. It:
- Sends the persona query into the controller.
- If a follow-up question is returned:
  - Captures the question, quick replies, and LLM-generated answer.
  - Judges the question for **relevance** and **newness** (binary with confidence + rationale).
  - Appends the Q&A to the conversation history.
- If recommendations are returned:
  - Evaluates each vehicle against the persona query.
  - Computes precision@k, NDCG@k, infra-list diversity, and attribute-level satisfaction rates.
  - Prints the follow-up Q&A + judge table above the recommendation evaluation table.

**Usage**
```bash
python review_simulation/run.py data/personas.csv \
  --limit 20 \
  --metric-k 20 \
  --k 3 \
  --n-per-row 3 \
  --method embedding_similarity \
  --export data/persona_results.csv
```

**Key arguments**
- `--k`: Number of follow-up questions the controller may ask before recommending.
- `--n-per-row`: Vehicles per row in the recommendation grid.
- `--method`: Recommendation method (`embedding_similarity` or `coverage_risk`).
- `--limit`: Max vehicles to evaluate.
- `--metric-k`: k for precision@k and NDCG@k.
- `--export`: Optional CSV path for results.

### `replay_results.py`
Recreates the console UI from an exported CSV and can persist aggregated metrics as JSON. This ensures repeatable reporting without re-running the simulation.

**Usage**
```bash
python review_simulation/replay_results.py data/persona_results.csv \
  --metric-k 20 \
  --stats-output data/persona_results_stats.json
```
