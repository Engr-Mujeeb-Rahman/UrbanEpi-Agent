# UrbanEpi-Agent

Code, derived results, and scripts for the paper *Privacy-Budgeted Federated Release with LLM Question Parsing for District-Level Disease Surveillance*.

UrbanEpi-Agent is the research prototype studied in the paper. It lets several health institutions share district-level disease signals without sharing records. Each institution keeps its records and one differential-privacy budget, which pays for both continuous monitoring releases and ad-hoc questions. A language model parses plain-language questions into validated typed plans, and never writes the queries that run. A call rule that reads only private releases decides when the language model is called.

The evaluation uses 40,963 real surveillance records from Mexico City in a simulated federation (16 boroughs, five institutions split from one public file) with injected outbreaks.

## Repository layout

| Path | Contents |
|------|----------|
| `src/urbanepi/` | The system: institutions, aggregator, privacy accounting, gate, LLM agent, evaluation code |
| `scripts/` | Data preparation, experiments, and figure scripts |
| `tests/` | Unit tests (`pytest`) |
| `configs/base.yaml` | All settings used in the paper |
| `experiments/frozen/` | Settings and gate parameters frozen before the test runs |
| `experiments/results/final/` | Result files behind every table and figure in the paper |
| `experiments/llm_cache/` | Cached LLM responses, so reruns repeat the paper's calls without an API key |
| `data/benchmark/` | The UrbanEpi-Q question benchmark and its review sheets |
| `data/public/` | Public borough metadata: names, populations, boundaries, adjacency, ontology |
| `figures/` | The paper's figures, as produced by the scripts |

## Setup

Python 3.11 or later.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # Windows: .venv\Scripts\pip
.venv/bin/pip install -e .
```

Scripts are run from the repository root. Calls not found in `experiments/llm_cache/` need a Gemini API key in a `.env` file (see `.env.example`).

## Data

Raw records are not included. Download the open influenza and COVID-19 dataset of Mexico's Secretaría de Salud:

- Source: `https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_influenza_covid19.zip`
- The paper uses the Internet Archive snapshot of 30 April 2025.
- Save it as `data/raw/mexico/datos_abiertos_influenza_covid19_20250430.zip` and check it against `data/raw/mexico/SHA256SUMS.txt`.

Then build the events table:

```bash
python scripts/build_events.py
```

## Reproducing the paper

| Paper item | Script | Result file |
|------------|--------|-------------|
| Question parsing (E1) | `run_parse_benchmark.py`, `run_e1_baselines.py`, `run_e1_cypher_v2.py` | `e1_baselines.json`, `e1_cypher_v2.json` |
| Answer quality and allocation (E2) | `run_e2.py`, `run_e2_ci.py`, `run_e2_crn.py` | `e2.json`, `e2_ci.json`, `e2_crn.json` |
| Detection, federation, budget designs, gate (E3, E4, E5, E7, E8) | `run_detection_experiments.py e3 e4 e5 e7 e8` | `e3.json` to `e8.json` |
| Matched chance overlap and federation table | `run_matched_rate.py` | `matched_rate.json` |
| LLM triage and worked example (E6) | `run_e6.py`, `run_triage_power_and_tokens.py` | `e6.json`, `triage_power_tokens.json` |
| Privacy audit | `run_audit.py` | `audit.json` |
| Threshold checks | `run_calibration_spread.py`, `run_threshold_power.py` | `calibration_spread.json`, `threshold_power.json` |
| Latency, tokens, and overhead | `run_system_metrics.py` | `system_metrics.json` |
| Review checks | `run_review_checks.py` | `review_checks.json` |
| Second annotator | `score_second_annotator.py` | `second_annotator.json` |
| Bootstrap intervals, alert load, validation-chosen thresholds, Farrington-type and scan baselines, shorter test period, run without IMSS | `run_supervisor_checks.py` | `supervisor_checks.json` |
| Curves over 15 threshold draws (Figures 2 to 4) | `run_threshold_draws.py` | `threshold_draws.json` |
| Rule-based pre-filter | `run_prefilter.py` | `prefilter.json` |
| Figures 1 to 5 | `make_diagrams.py`, `make_figures.py` | `figures/` |

Experiments use fixed seeds, and the event-merging gate visits boroughs in sorted order, so reruns give the same numbers. Set `PYTHONHASHSEED=0` as well if you change the code. The calls to the language model were made with `gemini-3.5-flash-lite` and are cached. The triage experiment (E6) ran before the gate was made order-independent, so rerunning it may need new LLM calls.

## Tests

```bash
python -m pytest
```

Some tests are skipped when the raw data has not been downloaded.

## Limits

This is a research prototype. Institutions are simulated from one public file, secure aggregation is simulated, and the system is not intended for clinical or autonomous public-health decisions.

## License

Code is released under the MIT License (see `LICENSE`). The source data remain under the terms of their publisher.
