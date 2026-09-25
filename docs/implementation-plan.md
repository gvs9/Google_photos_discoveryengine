# Implementation Plan: Google Photos Discovery Engine

## Goal

Build a phase-wise, end-to-end AI-powered discovery pipeline that:
1. Ingests user feedback at scale from 7+ public sources
2. Processes and embeds it into a queryable vector store
3. Runs AI analysis to extract retrieval pain points, memory cues, and search behaviors
4. Decomposes the business metric into evidence-backed opportunity areas

---

## Phase Overview

| Phase | Name | Deliverable | Complexity |
|---|---|---|---|
| **0** | Project Setup & Config | Scaffolded repo, env, dependencies | 🟢 Low |
| **1** | Data Ingestion | Raw data from all 7 sources in `data/raw/` | 🟡 Medium |
| **2** | Processing Pipeline | Cleaned, embedded chunks in ChromaDB | 🟡 Medium |
| **3** | AI Analysis Engine | Structured extractions per chunk in Parquet | 🔴 High |
| **4** | Metric Decomposition | Gap mapping + opportunity scorecard | 🔴 High |
| **5** | Output & Reporting | `insight_report.md`, `opportunity_map.json`, dashboard | 🟡 Medium |

---

## Phase 0: Project Setup & Configuration

**Objective**: Scaffold the project, install all dependencies, and wire up configuration.

### Tasks

- [x] Initialize git repo and create `.gitignore` (exclude `data/`, `.env`, `__pycache__`)
- [x] Create directory structure as defined in `architecture.md`
  ```
  src/ingestion/   src/processing/   src/analysis/
  src/decomposition/   src/output/
  data/raw/   data/processed/   data/vector_store/
  outputs/   config/   tests/   docs/
  ```
- [x] Create `requirements.txt` with all dependencies
- [x] Create `config/settings.py` using `pydantic-settings` to manage:
  - API keys: `GROQ_API_KEY`, `APIFY_API_TOKEN`, `YOUTUBE_API_KEY`
  - Thresholds: `RELEVANCE_THRESHOLD`, `MIN_CHUNK_LENGTH`, `HDBSCAN_MIN_CLUSTER_SIZE`
  - Paths: `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `VECTOR_STORE_DIR`
  - Date range: `SCRAPE_FROM_DATE` (default: 3 years ago)
- [x] Create `.env.example` template
- [x] Verify all imports work with a simple smoke test script (`python smoke_test.py` → PASS)

### Dependencies (`requirements.txt`)

```
# Scraping
requests>=2.31
beautifulsoup4>=4.12
google-play-scraper>=1.2
app-store-scraper>=0.3
apify-client>=1.7      # Reddit scraping via Apify cloud actor
tweepy>=4.14
youtube-data-api>=0.0.21

# NLP & Embeddings
spacy>=3.7
sentence-transformers>=2.6
groq>=0.9

# Vector Store
chromadb>=0.5

# Clustering
hdbscan>=0.8
scikit-learn>=1.4
numpy>=1.26
pandas>=2.2
pyarrow>=16.0

# Deduplication
datasketch>=1.6      # MinHash LSH

# LLM Orchestration
tenacity>=8.3        # Retry with backoff
tqdm>=4.66

# Config
pydantic-settings>=2.2
python-dotenv>=1.0

# Output
streamlit>=1.35
jinja2>=3.1
```

### Exit Criteria
- `python -c "import chromadb, apify_client, groq, hdbscan"` runs without error
- `.env` populated with valid API keys
- Directory structure created

---

## Phase 1: Data Ingestion

**Objective**: Collect raw text data from all 7 sources and persist as partitioned JSON in `data/raw/`.

**Architecture layer**: Layer 1 — Data Ingestion

### Raw Data Schema (per record)

```json
{
  "id": "sha256_of_content",
  "source": "reddit",
  "url": "https://...",
  "author_type": "user",
  "date": "2024-03-15T10:22:00Z",
  "rating": null,
  "title": "...",
  "content": "...",
  "metadata": {}
}
```

### Ingestion Rules (all scrapers must enforce)
- Skip records with `content` length < 30 characters
- Dedup by `(source, sha256(content))`
- Store to `data/raw/<source>/YYYY-MM-DD.json` (daily partitioned)
- Log: records fetched, records saved, records skipped (duplicate/short)

---

### 1.1 — `src/ingestion/play_store.py`

**Source**: Google Play Store reviews for Google Photos (`com.google.android.apps.photos`)  
**Tool**: `google-play-scraper`

| Parameter | Value |
|---|---|
| App ID | `com.google.android.apps.photos` |
| Count | 5,000 (configurable) |
| Filter | All ratings (1–5 stars) |
| Sort | Most relevant |

**Key steps**:
1. Call `reviews()` with pagination via `continuation_token`
2. Extract: `reviewId`, `content`, `score` (rating), `at` (date), `thumbsUpCount`
3. Map to raw schema and save

---

### 1.2 — `src/ingestion/app_store.py`

**Source**: Apple App Store reviews  
**Tool**: `app-store-scraper`

| Parameter | Value |
|---|---|
| App Name | `google-photos` |
| App ID | `962194608` |
| Country | `us` |
| Count | 3,000 |

**Key steps**: Same pattern as Play Store scraper.

---

### 1.3 — `src/ingestion/reddit.py`

**Source**: Reddit posts + comments  
**Tool**: `apify-client` — Apify cloud actor [`trudax/reddit-scraper`](https://apify.com/trudax/reddit-scraper)

> **Why Apify?** Reddit's API now requires a policy agreement + credentials approval process. Apify provides a managed Reddit actor that works immediately with just an API token — no Reddit credentials needed.

| Subreddits | Query terms |
|---|---|
| `r/googlephotos` | All posts (past 3 years) |
| `r/androidapps` | `"google photos" search OR find OR remember` |
| `r/ios` | `"google photos" can't find OR lost photo` |
| `r/mildlyinfuriating` | `"google photos"` |

**Apify Actor**: `trudax/reddit-scraper`  
**Run input schema**:
```json
{
  "startUrls": [
    {"url": "https://www.reddit.com/r/googlephotos/"},
    {"url": "https://www.reddit.com/r/androidapps/"}
  ],
  "searchPhrases": ["google photos can't find", "google photos search", "lost photo google photos"],
  "maxItems": 2000,
  "proxy": {"useApifyProxy": true}
}
```

**Key steps**:
1. Initialise `ApifyClient(token=settings.apify_api_token)`
2. Call `client.actor("trudax/reddit-scraper").call(run_input=run_input)`
3. Iterate `client.dataset(run["defaultDatasetId"]).iterate_items()`
4. For each item: map `title`, `selftext`, `url`, `score` (upvotes), `created_utc` to raw schema
5. Save comments with `score >= 5` as separate records
6. Apify handles all rate-limiting and proxy rotation automatically

---

### 1.4 — `src/ingestion/youtube.py`

**Source**: YouTube comments on Google Photos tutorial/review videos  
**Tool**: YouTube Data API v3

**Target videos**:
- Search for: `"google photos" search tips`, `google photos how to find old photos`
- Collect top 20 videos by view count
- For each video: fetch up to 200 top-level comments

**Key steps**:
1. `youtube.search().list()` → get video IDs
2. `youtube.commentThreads().list(videoId=..., maxResults=100)` → get comments
3. Map to raw schema (no rating field; use `likeCount` as proxy)

---

### 1.5 — `src/ingestion/community.py`

**Source**: Google Photos Help Community (`support.google.com/photos/community`)  
**Tool**: `requests` + `BeautifulSoup`

**Key steps**:
1. Paginate through search results for `"can't find"`, `"search"`, `"remember"`
2. For each thread: scrape question body + all answers
3. Respect `robots.txt`; add 2s delay between requests
4. Store thread title + body + each answer as separate records

---

### 1.6 — `src/ingestion/social.py`

**Source**: Twitter/X  
**Tool**: `tweepy` (requires Basic or Academic access) or `snscrape` (no API key)

**Query**: `"google photos" (can't find OR lost photo OR remember photo) -is:retweet lang:en`  
**Date range**: Last 3 years, up to 2,000 tweets

---

### 1.7 — Forums (`src/ingestion/forums.py`)

**Sources**: XDA Developers, Stack Overflow/Exchange  
**Tool**: `requests` + `BeautifulSoup` / Stack Exchange API

| Source | Query |
|---|---|
| XDA Forums | `"google photos" search photo` |
| Stack Overflow | `[google-photos] search` |
| Android Enthusiasts SE | `[google-photos] retrieve` |

---

### Phase 1 Exit Criteria
- [x] `data/raw/` contains JSON files for all 7 sources (Note: Reddit/Community blocked by anti-bot, Twitter skipped by design)
- [x] Total records ≥ 5,000 across all sources (Achieved ~11,333)
- [x] No records below 30 char threshold remain
- [x] Deduplication log shows < 5% collision rate

---

## Phase 2: Processing Pipeline

**Objective**: Transform raw text into clean, relevant, embedded chunks stored in ChromaDB.

**Architecture layer**: Layer 2 — Processing Pipeline

**Entry point**: `src/processing/` modules, run sequentially per source.

---

### 2.1 — `src/processing/cleaner.py`

**Input**: Raw JSON records  
**Output**: Cleaned records (Parquet)

Steps per record:
1. Strip HTML tags (`BeautifulSoup`)
2. Normalize Unicode (NFKC)
3. Collapse whitespace; remove boilerplate (e.g., "Read more", "Show less")
4. Language detection with `langdetect`; keep only `en`
5. **MinHash deduplication**: compute MinHash signature → query LSH index → skip if Jaccard similarity > 0.85 with any existing record

---

### 2.2 — `src/processing/relevance_filter.py`

**Input**: Cleaned records  
**Output**: Relevant records only

Two-stage filter:
1. **Keyword gate** (fast): must contain ≥1 retrieval signal word AND ≥1 photo context word
2. **Embedding similarity** (slower, only on keyword-passing records):
   - Embed content with `sentence-transformers/all-MiniLM-L6-v2`
   - Compute cosine similarity to 5 seed sentences
   - Keep if max similarity ≥ 0.45

**Seed sentences**:
```
"I can't find an old photo I remember taking"
"I remember the photo exists but can't locate it in Google Photos"
"searched Google Photos but couldn't find the picture"
"I remember something about the photo but the search doesn't work"
"Google Photos search failed to return the photo I was looking for"
```

---

### 2.3 — `src/processing/chunker.py`

**Input**: Relevant records  
**Output**: Sentence-level chunks (Parquet)

1. Load `en_core_web_sm` spaCy model
2. Sentencize each record's content
3. Create sliding windows of **max 5 sentences, min 2 sentences** with 1-sentence overlap
4. Assign `chunk_id = sha256(record_id + chunk_index)`
5. Attach metadata from parent record: `source`, `date`, `url`, `rating`

---

### 2.4 — `src/processing/embedder.py`

**Input**: Chunks (Parquet)  
**Output**: Chunks + embeddings stored in ChromaDB

1. Batch chunks in groups of 100
2. Generate embeddings: `all-MiniLM-L6-v2` via `sentence-transformers` (local; Groq does not provide an embeddings API)
3. Upsert into ChromaDB collection `photo_retrieval_chunks`:
   - Document: chunk text
   - Embedding: vector
   - Metadata: `source`, `date`, `url`, `rating`, `chunk_id`, `record_id`
4. Log: total chunks embedded, ChromaDB collection size

### Phase 2 Exit Criteria
- [x] ChromaDB collection contains ≥ 3,000 chunks — **4,070 achieved**
- [x] All chunks have metadata attached (source, date, url, rating, chunk_index)
- [x] Relevance filter pass rate logged — **33.6% overall** (photo-gate: 64.8% → embedding@0.30: 51.8%)
- [x] `data/processed/chunks.parquet` saved as backup — **4,070 chunks across 5 sources**

---

## Phase 3: AI Analysis Engine

**Objective**: Run LLM-based extraction and ML clustering over all chunks to produce structured insight data.

**Architecture layer**: Layer 3 — AI Analysis Engine

> [!NOTE]
> All LLM calls use `tenacity` retry with exponential backoff (max 5 retries). Batch size: 50 chunks per run. Results cached to Parquet to avoid re-processing.

---

### 3.1 — `src/analysis/extractor.py` — Retrieval Problem Extractor

**Input**: Each chunk from ChromaDB  
**Output**: `data/processed/extractions.parquet`

**Per-chunk LLM call** (structured output / JSON mode):
```
System: You are a UX researcher analyzing user complaints about photo retrieval.
        Always respond in valid JSON.

User: Analyze this user post about Google Photos:
      """<chunk_text>"""

      Extract:
      {
        "is_retrieval_related": true/false,
        "retrieval_problem": "<1-2 sentence summary>",
        "memory_cues": ["<cue1>", "<cue2>"],
        "forgotten_info": ["<item1>", "<item2>"],
        "search_attempts": ["<attempt1>"],
        "outcome": "found" | "not_found" | "unknown",
        "frustration_level": 1-5
      }
```

- Skip chunks where `is_retrieval_related = false`
- Cache all results to `extractions.parquet`

---

### 3.2 — `src/analysis/memory_classifier.py` — Memory Cue Classifier

**Input**: `memory_cues` list from extractions  
**Output**: `memory_cue_type` column added to extractions Parquet

Use a **zero-shot classifier** (Groq `llama-3.3-70b-versatile` or local `bart-large-mnli`) to label each cue:

| Label | Description |
|---|---|
| `temporal` | Time-based memory ("last year", "in 2019") |
| `spatial` | Location-based ("Goa trip", "old house") |
| `people` | Person-based ("with mom", "my friends") |
| `object_subject` | Subject/object in photo ("medicine bottle", "dog") |
| `event_context` | Event-based ("birthday", "graduation") |
| `visual_aesthetic` | Visual property ("blurry", "red dress") |
| `emotion` | Emotional tag ("happy", "sad") |

Output: frequency table of cue types → feeds directly into Part 2 decomposition.

---

### 3.3 — `src/analysis/gap_classifier.py` — Gap Classifier

**Input**: `forgotten_info` list from extractions  
**Output**: `gap_type` column added

Label each forgotten item:

| Gap Label | Description |
|---|---|
| `missing_timestamp` | No date/time info |
| `missing_location` | No location info |
| `missing_person_name` | Knows a person was in it, not their name |
| `missing_object_name` | Can't name the subject |
| `missing_album` | Doesn't know which album/folder |
| `no_keywords` | Completely keyword-free memory |

---

### 3.4 — `src/analysis/behavior_tagger.py` — Search Behavior Tagger

**Input**: `search_attempts` list from extractions  
**Output**: `behavior_tags` column

Label each attempt:

| Tag | Description |
|---|---|
| `keyword_guess` | Tried typing descriptive words |
| `date_browse` | Used date slider / scrolled by month |
| `album_browse` | Browsed albums manually |
| `face_search` | Used "people" / face search |
| `location_search` | Used "places" / map search |
| `timeline_scroll` | Manually scrolled through all photos |
| `asked_someone` | Asked another person for help |
| `gave_up` | Explicitly abandoned the search |

---

### 3.5 — `src/analysis/clusterer.py` — Opportunity Clusterer

**Input**: All `retrieval_problem` strings from `extractions.parquet`  
**Output**: `data/processed/clusters.parquet`

Steps:
1. Embed all `retrieval_problem` strings (batch call)
2. Run `HDBSCAN(min_cluster_size=10, metric='cosine')` on the embedding matrix
3. For each cluster:
   - Identify the 5 most central chunks (lowest distance to centroid)
   - Call LLM to generate a 1-sentence cluster label from those 5 chunks
   - Compute `cluster_score = chunk_count × mean_frustration_level`
4. Save cluster assignments back to `extractions.parquet` (`cluster_id`, `cluster_label`)
5. Output ranked cluster list in `clusters.parquet`

---

### 3.6 — `src/analysis/evidence_ranker.py` — Evidence Ranker

**Input**: `extractions.parquet` with cluster assignments  
**Output**: `data/processed/evidence.parquet`

For each cluster, select top 5 verbatim source quotes ranked by:

```
score = (frustration_level × 2) + specificity_score + (log(upvotes + 1))
```

Where `specificity_score` = number of distinct memory cue types mentioned (0–7 scale).

### Phase 3 Exit Criteria
- [x] `extractions.parquet` contains structured extraction for ≥ 80% of chunks
- [x] `clusters.parquet` contains ≥ 5 clusters with ≥ 10 members each
- [x] Top cluster has a coherent, meaningful label
- [x] `evidence.parquet` has ≥ 5 quotes per top-10 clusters

---

## Phase 4: Metric Decomposition Engine

**Objective**: Map evidence clusters to the 4 diagnostic gaps and produce the opportunity scorecard.

**Architecture layer**: Layer 4 — Metric Decomposition Engine

---

### 4.1 — `src/decomposition/gap_mapper.py`

**Input**: `clusters.parquet`, `extractions.parquet`  
**Output**: `data/processed/gap_mapping.parquet`

**Mapping logic**: For each cluster, use LLM to classify it into one (or more) of the 4 gaps:

```
Given this cluster of user complaints labeled "<cluster_label>",
classify which retrieval gap(s) it primarily represents:

1. Expression Gap     – User cannot articulate what they remember
2. Intent Gap         – Google Photos misunderstands the query
3. Presentation Gap   – Results are hard to evaluate
4. Recovery Gap       – User cannot refine after failure

Respond with: { "primary_gap": "...", "secondary_gap": "..." (optional) }
```

Aggregate per gap:
- `volume`: total chunks mapped to this gap
- `avg_frustration`: mean frustration level across mapped chunks
- `behavior_profile`: most common `behavior_tags` for this gap
- `top_memory_cues`: most common memory cue types for this gap

---

### 4.2 — `src/decomposition/scorecard.py`

**Input**: `gap_mapping.parquet`  
**Output**: `outputs/opportunity_scorecard.json`

Compute per gap:

| Dimension | Formula |
|---|---|
| **Volume Score** (1–5) | `min(5, chunk_count / 200)` |
| **Severity Score** (1–5) | `avg_frustration_level` |
| **Uniqueness Score** (1–5) | Manual input via config |
| **Feasibility Score** (1–5) | Manual input via config |
| **Composite** | `(Volume × Severity) / (1 + (5 - Feasibility))` |

Output JSON:
```json
{
  "gaps": [
    {
      "gap": "Expression Gap",
      "volume_score": 4.2,
      "severity_score": 3.8,
      "uniqueness_score": 4,
      "feasibility_score": 3,
      "composite_score": 5.32,
      "top_clusters": ["...", "..."],
      "top_evidence_quotes": ["..."],
      "recommended_metric_proxy": "% sessions with blank search abandonment"
    }
  ]
}
```

### Phase 4 Exit Criteria
- [x] All 4 gaps have at least 1 cluster mapped to them
- [x] Composite scores are ranked and interpretable
- [x] Gap with highest composite score is supported by ≥ 3 distinct evidence quotes

---

## Phase 5: Output & Reporting

**Objective**: Generate human-readable deliverables for the PM team.

**Architecture layer**: Output Layer

---

### 5.1 — `src/output/report_generator.py`

Generates `outputs/insight_report.md` using a Jinja2 template:

**Report sections**:
1. **Executive Summary** — Top 3 findings in 3 sentences
2. **Data Collection Summary** — Records per source, pass rates, total chunks
3. **Key Retrieval Problems** — Top 10 clusters with labels, volume, and top quote
4. **Memory Cue Analysis** — Bar chart data of cue type frequencies
5. **Gap Distribution** — Which of the 4 gaps has the most evidence
6. **Opportunity Scorecard** — Ranked gaps with scores and recommended metric proxies
7. **Recommended Next Step** — Highest-scoring gap and its top evidence

---

### 5.2 — `src/output/report_generator.py` (continued)

Generates `outputs/opportunity_map.json`:
```json
{
  "clusters": [
    {
      "rank": 1,
      "label": "...",
      "volume": 142,
      "gap": "Expression Gap",
      "top_quotes": ["...", "..."],
      "memory_cues": ["temporal", "spatial"],
      "behavior_tags": ["keyword_guess", "gave_up"]
    }
  ]
}
```

---

### 5.3 — `src/output/dashboard.py` (Optional)

Streamlit dashboard with:
- Cluster explorer (filterable by gap, source, frustration level)
- Memory cue frequency chart
- Evidence quote browser with source links
- Opportunity scorecard heatmap

Run: `streamlit run src/output/dashboard.py`

---

### 5.4 — `src/output/answer_generator.py` (LLM Answer Generator)

**Objective**: Use Groq to explicitly answer the 4 core questions from the Problem Statement using our synthesized data.
**Input**: `outputs/opportunity_map.json`, `data/processed/extractions.parquet`
**Output**: `outputs/problem_statement_answers.md`

Generates an LLM prompt injecting the top memory cues, gaps, and behaviors we extracted, asking it to answer:
1. What kinds of old photos do users struggle to retrieve?
2. What information do people actually remember about a photo?
3. What information have they forgotten?
4. How do users formulate searches when their memory is incomplete?

### Phase 5 Exit Criteria
- [x] `outputs/insight_report.md` generated and readable
- [x] `outputs/opportunity_map.json` valid JSON, parseable
- [x] `outputs/opportunity_scorecard.json` contains ranked gaps
- [ ] Dashboard loads without errors (if implemented)

---

## End-to-End Run Order

```bash
# Phase 0: Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # fill in API keys

# Phase 1: Ingestion (run each scraper)
python -m src.ingestion.play_store
python -m src.ingestion.app_store
python -m src.ingestion.reddit
python -m src.ingestion.youtube
python -m src.ingestion.community
python -m src.ingestion.social
python -m src.ingestion.forums

# Phase 2: Processing
python -m src.processing.cleaner
python -m src.processing.relevance_filter
python -m src.processing.chunker
python -m src.processing.embedder

# Phase 3: Analysis
python -m src.analysis.extractor
python -m src.analysis.memory_classifier
python -m src.analysis.gap_classifier
python -m src.analysis.behavior_tagger
python -m src.analysis.clusterer
python -m src.analysis.evidence_ranker

# Phase 4: Decomposition
python -m src.decomposition.gap_mapper
python -m src.decomposition.scorecard

# Phase 5: Output
python -m src.output.report_generator
streamlit run src/output/dashboard.py  # optional
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Apify Reddit actor fails / quota exceeded | Low | High | Check Apify dashboard for run errors; upgrade plan if free credits exhausted; Apify $5/month free tier covers ~2,000–5,000 posts |
| Twitter/X API access restricted | High | Medium | Fall back to `snscrape`; deprioritize this source |
| LLM costs exceed budget (extraction at scale) | Low | Medium | Groq's free tier is generous; cache all results to Parquet to avoid re-calling; use `llama-3.1-8b-instant` for classification, `llama-3.3-70b-versatile` only for cluster labeling |
| Low relevance filter pass rate (< 10%) | Medium | High | Expand seed keywords; lower cosine threshold to 0.35 |
| HDBSCAN produces too few clusters | Medium | Medium | Tune `min_cluster_size` down to 5; try `min_samples=3` |
| Google Photos Community blocks scraper | Low | Medium | Add `User-Agent` rotation; fallback to manual export |

---

## Milestone Summary

```
Week 1  │  Phase 0 + Phase 1  │  All raw data collected
Week 2  │  Phase 2            │  ChromaDB populated
Week 3  │  Phase 3            │  All extractions + clusters done
Week 4  │  Phase 4 + Phase 5  │  Final report + scorecard delivered
```
