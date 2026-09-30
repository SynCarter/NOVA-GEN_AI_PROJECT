# NOVA — AI Data Analyst

**Developer:** S Larens  
**Roll No:** 23053071  

NOVA is a free-to-run GenAI data-analysis assistant. Users upload CSV data, ask questions in natural language, receive calculation-backed answers and explore interactive visualizations.

## Architecture

CSV → Streamlit UI → Natural-language operation router → Pandas calculation engine → verified result → local FLAN-T5 explanation → answer/chart.

The language model is not trusted with numerical computation. Pandas performs the calculation first, which reduces hallucination risk.

## Features
- CSV upload and automatic profiling
- Missing values and duplicate detection
- Natural-language averages, totals, min/max, rankings and category filters
- Correlation analysis
- Automated insights
- Interactive Plotly charts
- Local open-source GenAI explanation using `google/flan-t5-small`
- Deterministic fallback if the model cannot load
- No paid API required
- No API key required
- Test suite and evaluation script

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first AI explanation may download the FLAN-T5-small model from Hugging Face. After that it is cached locally.

## Tests

```bash
pytest -q
python evaluate.py
```

## Free deployment

Deploy `app.py` from GitHub with Streamlit Community Cloud. The first model load can take longer because the open model must be downloaded.

## Submission safety

Never commit passwords, tokens, API keys or private datasets. Keep sample data only.

## Project claim

This project uses an open-source local language model for explanation and a deterministic Pandas engine for calculations. It should be evaluated using the supplied test/evaluation scripts rather than assumed metrics.
