# Edge Cases & Corner Scenarios: Google Photos Discovery Engine

This document catalogs known and anticipated edge cases across all pipeline phases, with their root causes, detection strategies, and handling approaches.

---

## Phase 1: Data Ingestion

### 1.1 Scraper / Network

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Rate limit hit** | Platform returns HTTP 429 | Response status code | Exponential backoff: `wait = min(2^attempt × 1s, 60s)`; retry up to 5× |
| **IP ban / soft block** | Scraper returns 403 or CAPTCHA page | Response body contains "captcha" or "access denied" | Log + skip source for this run; alert operator |
| **robots.txt disallows scraping** | Target path blocked | Parse `robots.txt` before first request | Skip disallowed paths entirely; log which paths were skipped |
| **Network timeout** | Request hangs > 30s | `requests` timeout param | Set `timeout=30`; retry once; skip record on second failure |
| **SSL / TLS error** | Certificate verification fails | `SSLError` exception | Log + skip; do not disable SSL verification globally |
| **Redirect loop** | URL redirects infinitely | `requests` `max_redirects` | Cap at 5 redirects; raise exception and skip |
| **API key expired / invalid** | 401 Unauthorized from Apify | HTTP 401 response | Raise `ConfigurationError` immediately; halt ingestion for that source |
| **API quota exhausted** | Apify free-tier credits exhausted / YouTube daily quota hit | HTTP 403 or Apify `ActorJobStatus == FAILED` | Save checkpoint; resume next run; check Apify dashboard for credit balance |

---

### 1.2 Content Quality

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Empty content field** | Review / comment body is `null` or `""` | `len(content.strip()) == 0` | Skip record; increment `skipped_empty` counter |
| **Content below minimum length** | Post is 1–29 characters ("OK", "👍") | `len(content) < MIN_CHUNK_LENGTH` | Skip record |
| **Content is only emoji / symbols** | `"😤😤😤"` | Regex: `^\W+$` after strip | Skip record |
| **Duplicate record same source** | Same review scraped twice on re-run | `sha256(source + content)` exists in seen set | Skip; increment `skipped_duplicate` counter |
| **Near-duplicate across sources** | Same complaint posted on Reddit and Play Store | MinHash Jaccard similarity ≥ 0.85 | Keep the one with higher engagement signal (upvotes / rating); mark the other as `near_dup` |
| **Non-English content** | Review in Hindi, Spanish, etc. | `langdetect` returns non-`en` label | Skip in v1; tag as `lang=xx` and store separately for v2 multilingual support |
| **langdetect fails / throws** | Very short or mixed-language text | `langdetect.LangDetectException` | Catch exception; default to `lang=unknown`; include in pipeline (conservative approach) |
| **HTML entities unescaped** | `&amp;`, `&lt;`, `&#39;` in scraped text | Presence of `&` followed by `\w+;` | Run `html.unescape()` before any processing |
| **Markdown / code in content** | Stack Overflow answer with code blocks | Regex match for ` ``` ` or indented blocks | Strip code blocks before NLP; preserve surrounding prose |
| **Bot-generated content** | Automated review spam | Repetitive phrasing pattern; author post frequency > 50/day | Flag `is_suspected_bot=True`; exclude from analysis; keep in raw store |

---

### 1.3 Platform-Specific

| Edge Case | Scenario | Handling |
|---|---|---|
| **Play Store review edited** | User updates a review — old and new both fetched | Dedup by `reviewId`; keep latest by `at` timestamp |
| **Reddit post deleted** | `[deleted]` or `[removed]` content | Check `content in ("[deleted]", "[removed]")`; skip |
| **Reddit comment is top-level but reply-chain only** | Conversation without an original post body | Treat each top-level comment as a standalone record |
| **Apify actor run times out** | Large dataset takes > 30 min | Check `run["status"] == "TIMED-OUT"` after `.call()` | Reduce `maxItems`; split across multiple actor runs |
| **Apify dataset returns partial results** | Actor interrupted mid-run | `itemCount < expected` in dataset info | Detect partial run; merge with prior partial results; re-run |
| **YouTube video is private/deleted** | `commentThreads` API returns 404 | Catch `HttpError 404`; skip video; log |
| **YouTube comments disabled** | API returns empty `items` array | Check `items == []`; skip video silently |
| **Twitter/X API access revoked** | `tweepy` auth fails with 403 | Catch `tweepy.Forbidden`; fall back to `snscrape` |
| **snscrape returns 0 results** | Query too narrow or API change | Log zero-result warning; skip source for this run |
| **Google Community pagination breaks** | BeautifulSoup finds no "next page" link | Check for `None` before following; stop pagination loop |
| **App Store country mismatch** | App ID valid but not available in `us` store | Catch scraper `AppNotFound`; try `gb` or `in` as fallback country |

---

## Phase 2: Processing Pipeline

### 2.1 Text Cleaning

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Content is pure HTML** | Entire string is markup with no visible text | `BeautifulSoup.get_text()` returns < 10 chars | Discard record post-cleaning |
| **Unicode control characters** | Zero-width spaces, BOM, RTL marks embedded | `unicodedata.category(c) == 'Cc'` | Strip via `re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)` |
| **Extremely long record** | Single Reddit post > 50,000 characters | `len(content) > 50_000` | Truncate to first 10,000 characters; log truncation |
| **Repeated phrases / spam pattern** | "Great app! Great app! Great app!" x 20 | Regex for repeated n-gram (3+ repeats) | Collapse to single occurrence; flag `is_repetitive=True` |
| **URL-only content** | Comment is just a link | `re.match(r'^https?://\S+$', content.strip())` | Skip; no useful text signal |
| **After cleaning, content becomes empty** | All content was HTML boilerplate | `len(cleaned) == 0` | Skip record; log as `cleaned_to_empty` |

---

### 2.2 Relevance Filter

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Zero chunks pass keyword gate** | Source has no retrieval-related content | `keyword_pass_count == 0` after full source | Log warning; do not crash pipeline; continue with other sources |
| **Keyword match but off-topic** | "I'm looking for a photo editing app" matches `looking for` + `photo` | Embedding similarity < 0.45 | Correctly rejected at stage 2; expected behavior |
| **Embedding model unavailable** | `sentence-transformers` model fails to load | `OSError` on model load | Fall back to keyword-only filter; log degraded mode |
| **Embedding returns NaN / zero vector** | Tokenizer issue on unusual characters | `np.isnan(embedding).any()` | Re-embed after stripping non-ASCII; skip if still fails |
| **All chunks score below threshold** | Domain shift (very different phrasing) | `max_similarity < 0.35` across all chunks | Lower threshold by 0.05 and retry; log threshold adjustment |
| **Cosine similarity is exactly threshold** | Edge of decision boundary | `similarity == RELEVANCE_THRESHOLD` | Include (use `>=`); document this boundary decision |

---

### 2.3 Chunking

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Record is a single sentence** | One-line review: "Can't find my old photos!" | `len(sentences) == 1` | Treat entire record as a single chunk; do not skip |
| **spaCy fails to sentencize** | Garbled text with no punctuation | `len(sentences) == 1` and `len(text) > 500` | Force-split on `\n` line breaks as fallback |
| **Chunk overlaps produce duplicate extractions** | Sliding window overlap creates near-identical chunks | MinHash at chunk level | De-duplicate chunks before sending to analysis (Jaccard ≥ 0.9) |
| **Record splits into > 50 chunks** | Very long thread response | `chunk_count > 50` | Cap at 50 chunks per record; drop trailing ones; log |
| **Sentence boundary in mid-URL** | spaCy splits on `.com` | Detect `http` in split segment | Merge back with next sentence |

---

### 2.4 Embedding & Vector Store

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **ChromaDB collection already exists** | Re-run without clearing state | `chromadb.errors.UniqueConstraintError` on collection create | Use `get_or_create_collection()` always |
| **Duplicate chunk ID on upsert** | Same chunk re-processed | ChromaDB upsert is idempotent | Use `upsert()` not `add()`; no special handling needed |
| **Vector store disk full** | Large dataset fills local disk | `OSError: No space left on device` | Pre-check available disk (`shutil.disk_usage`); warn if < 2 GB free |
| **Batch embedding fails mid-batch** | Network error during batch of 100 | Exception on batch N | Log failed batch; retry that batch only; do not re-embed successful batches |
| **Embedding dimensionality mismatch** | Model changed between runs | `chromadb.errors.DimensionError` | Detect mismatch on first upsert; prompt operator to clear and rebuild collection |

---

## Phase 3: AI Analysis Engine

### 3.1 LLM Call (Groq API)

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Groq API rate limit (RPM/TPM)** | `429 Too Many Requests` | HTTP 429 response | `tenacity` retry with 60s wait; reduce batch concurrency to 1 |
| **Groq API server error** | `500 Internal Server Error` | HTTP 5xx response | Retry up to 5× with backoff; skip chunk after 5 failures; log |
| **Response is not valid JSON** | Model outputs prose instead of JSON | `json.JSONDecodeError` | Retry with stricter prompt ("Respond ONLY with valid JSON, no prose"); if still fails, mark chunk as `extraction_failed` |
| **JSON schema mismatch** | Model returns unexpected keys | `KeyError` on required field | Use `.get(field, default)` for all fields; log schema deviation |
| **`is_retrieval_related: false` for all chunks** | Model is too conservative | `retrieval_related_count == 0` after 50 chunks | Review prompt; adjust system message to be less restrictive |
| **`frustration_level` out of range** | Model returns `6` or `"high"` | Value not in `1..5` | Clamp to `[1, 5]`; map string values (`"high"` → 4, `"low"` → 2) |
| **`outcome` field hallucinated** | Model returns `"partial"` not in schema | Value not in `{"found", "not_found", "unknown"}` | Map unknown values to `"unknown"` |
| **Empty `memory_cues` list** | Post describes frustration but no specific memory | `memory_cues == []` | Acceptable; record as `cue_type=none`; include in `Expression Gap` evidence |
| **Chunk is too long for context window** | Chunk > 8,192 tokens | `groq.BadRequestError` with token count | Truncate chunk to 2,000 tokens before LLM call; log truncation |
| **Groq API key missing at runtime** | `.env` not loaded | `AuthenticationError` | Raise `ConfigurationError` at startup before processing begins |

---

### 3.2 Classifier (Memory Cue / Gap / Behavior)

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Multi-label ambiguity** | Cue is both temporal and event ("Christmas 2020") | Classifier returns top-2 labels | Store both; count both in frequency tables |
| **Classifier returns `None` label** | Cue doesn't fit any category | Output label not in taxonomy | Map to `other`; log for taxonomy review |
| **Empty input list** | `memory_cues` was `[]` for this chunk | `len(cues) == 0` | Skip classification; leave column as empty list |
| **Batch classification timeout** | LLM call for 50 cues times out | `tenacity` max wait exceeded | Fall back to keyword-based rule classifier for that batch |

---

### 3.3 HDBSCAN Clustering

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Too few data points to cluster** | < 20 retrieval-related extractions | `n_samples < 20` | Skip clustering; treat all extractions as a single cluster; log warning |
| **All points labeled noise (-1)** | `min_cluster_size` too large for data distribution | All `labels == -1` | Reduce `min_cluster_size` by 50%; retry up to 3 times |
| **Single mega-cluster** | 95%+ of points in one cluster | `cluster_sizes[0] / total > 0.95` | Sub-cluster using `min_cluster_size=5` on that cluster only |
| **Cluster count explodes** | Hundreds of tiny clusters | `n_clusters > 50` | Increase `min_cluster_size`; re-run; cap displayed clusters at top 20 by score |
| **NaN in embedding matrix** | Corrupted embedding from failed retry | `np.isnan(matrix).any()` | Drop rows with NaN before clustering; log count of dropped rows |
| **Duplicate embeddings** | Two identical chunks produce zero-distance | HDBSCAN assigns to same cluster normally | No special handling; expected behavior |

---

### 3.4 Evidence Ranker

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **Cluster has < 5 members** | Small cluster selected for top-10 | `cluster_size < 5` | Return all available quotes (< 5); do not pad with unrelated content |
| **All quotes in a cluster are from same source** | Reddit dominates a cluster | `unique_sources == 1` | Flag as `single_source_cluster=True`; note in report for analyst awareness |
| **Quote contains PII** | Username, email, phone in scraped text | Regex for email/phone patterns | Redact: `re.sub(r'\b[\w.]+@[\w.]+\.\w+\b', '[EMAIL]', text)` |
| **Quote is in a foreign language** | Slipped through language filter | `langdetect` on quote text | Skip quote; pick next ranked quote |
| **Upvote count is `None`** | Platform doesn't expose engagement | `upvotes is None` | Default to `0` in ranking formula; weight other signals more |

---

## Phase 4: Metric Decomposition

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **LLM maps cluster to no gap** | Cluster is ambiguous or off-topic | `primary_gap == null` in response | Map to `"Unclassified"`; exclude from scorecard; review manually |
| **All clusters map to same gap** | Model is biased toward one gap type | All `primary_gap == "Expression Gap"` | Review LLM prompt; add few-shot examples for each gap to calibrate |
| **Manual score fields missing** | `uniqueness_score` / `feasibility_score` not set in config | `None` value in scorecard formula | Default to `3` (neutral); warn operator to fill manually |
| **Composite score formula produces division by zero** | `feasibility_cost == -1` (impossible by design) | `ZeroDivisionError` | Clamp denominator to `max(1, ...)` |
| **Gap has zero volume** | No clusters map to a gap | `volume == 0` | Include gap in scorecard with score `0`; mark as `no_evidence_found` |

---

## Phase 5: Output & Reporting

| Edge Case | Scenario | Detection | Handling |
|---|---|---|---|
| **`outputs/` directory doesn't exist** | Fresh run with no prior output | `FileNotFoundError` on write | `os.makedirs("outputs/", exist_ok=True)` at startup |
| **Report template missing** | Jinja2 template file deleted | `TemplateNotFound` exception | Bundle template in `src/output/templates/`; raise clear error if missing |
| **`opportunity_map.json` is empty** | All phases produced zero results | `clusters == []` | Write JSON with `{"clusters": [], "status": "no_results", "reason": "..."}` |
| **Streamlit dashboard port already in use** | Port 8501 taken | `OSError: address already in use` | Specify alternate port: `streamlit run ... --server.port 8502` |
| **Parquet file corrupted** | Disk write interrupted mid-run | `pyarrow.lib.ArrowIOError` on read | Keep `.tmp` write pattern; rename on success; detect `.tmp` files on startup as incomplete |
| **Insight report too large** | Thousands of clusters bloat the report | Report > 5 MB | Cap to top 20 clusters in report; full data stays in JSON |

---

## Cross-Cutting Concerns

### Checkpointing & Resumability

| Scenario | Risk | Mitigation |
|---|---|---|
| Pipeline crashes mid-run | Re-processing all prior work wastes time and API budget | Write a `checkpoint.json` after each phase; skip completed phases on re-run |
| LLM extraction half-done | Some chunks extracted, others not | Track `extracted_chunk_ids` set; only call LLM on unextracted chunks |
| Scraper interrupted mid-pagination | Partial data saved | Save `continuation_token` / `last_page` after each page; resume from there |

### Concurrency

| Scenario | Risk | Mitigation |
|---|---|---|
| Parallel LLM calls exceed Groq RPM | All threads hit rate limit simultaneously | Use a semaphore to cap concurrent LLM calls to 5 |
| Parallel scrapers write to same file | File corruption from concurrent writes | One file per source per scraper; no shared file handles |

### Data Integrity

| Scenario | Risk | Mitigation |
|---|---|---|
| Schema drift between pipeline runs | Older Parquet columns missing in new code | Use `pandas` `reindex()` with defaults when reading Parquet |
| ChromaDB version upgrade changes storage format | Collection unreadable after upgrade | Pin `chromadb` version in `requirements.txt`; document migration steps |
| Encoding mismatch | `UnicodeDecodeError` on Parquet read | Always write Parquet with `utf-8` encoding; validate on read |

### Logging Standards (all modules)

Every module must log:
- `INFO`: records processed, records skipped (with reason), phase start/end
- `WARNING`: degraded mode (fallback used), near-duplicate detected, threshold adjusted
- `ERROR`: API failure (with status code), file write failure, schema mismatch

Use structured logging format:
```python
import logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
```

All logs written to both `stdout` and `logs/<phase>_<YYYY-MM-DD>.log`.
