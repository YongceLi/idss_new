# Review Simulation

Tools for generating persona-based evaluation runs of the interactive decision support system.

## Components
- **generate_persona_queries.py**: Builds persona prompts and single-turn user queries (including inferred upper price limits) from enriched review CSVs.
- **run.py**: Evaluates generated personas against recommendation pipelines, renders console UI, and optionally exports detailed metrics and vehicle judgements (with per-attribute checks and confidence-aware scoring).
- **replay_results.py**: Recreates the console UI from an exported CSV and can persist aggregated metrics (precision, diversity, NDCG, precision/NDCG with confidence > 0.6, satisfied@k, and attribute-level satisfied@k averages).

## Usage

### Generate personas and queries
```bash
python review_simulation/generate_persona_queries.py data/reviews_enriched.csv --output data/personas.csv
```

### Run simulation
```bash
python review_simulation/run.py data/personas.csv --limit 20 --metric-k 20 --export data/persona_results.csv
```
The UI displays per-vehicle attribute checks (price, condition, year, make, model, fuel type, body type, and misc preferences), overall satisfaction with rationale, and confidence. Final averages across personas are shown after all rows.

### Replay exported results
```bash
python review_simulation/replay_results.py data/persona_results.csv --metric-k 20 --stats-output data/persona_results_stats.json
```
This recreates the UI from the CSV and saves run-level aggregates to the optional stats path.
