# Google Photos Discovery Engine

An AI-powered research pipeline that ingests user feedback at scale, processes it, and surfaces structured insights about photo retrieval failures — enabling evidence-based product decisions.

## Project Goal

> Increase the percentage of users who successfully retrieve a photo they remember but cannot precisely describe when they start searching.

## Documentation

| Document | Purpose |
|---|---|
| [`docs/problemStatement.md`](docs/problemStatement.md) | Project context and strategic goal |
| [`docs/architecture.md`](docs/architecture.md) | System architecture and data flow |
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | Phase-wise implementation plan |
| [`docs/edge-cases.md`](docs/edge-cases.md) | Edge cases and handling strategies |

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd "Google Photos Discovery Engine"

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure API keys
cp .env.example .env
# Edit .env and fill in your API keys

# 4. Run Phase 0 smoke test
python smoke_test.py
```

## Pipeline Phases

| Phase | Command | Description |
|---|---|---|
| 1 — Ingestion | `python -m src.ingestion.<source>` | Collect raw data from 7 sources |
| 2 — Processing | `python -m src.processing.<module>` | Clean, filter, chunk, embed |
| 3 — Analysis | `python -m src.analysis.<module>` | LLM extraction + clustering |
| 4 — Decomposition | `python -m src.decomposition.<module>` | Gap mapping + scorecard |
| 5 — Output | `python -m src.output.report_generator` | Generate reports |

## Tech Stack

- **LLM**: Groq (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Vector Store**: ChromaDB
- **Clustering**: HDBSCAN
- **Scraping**: `apify-client` (Reddit), `google-play-scraper`, YouTube Data API v3, `BeautifulSoup`
