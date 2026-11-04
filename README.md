# ChronoFact

### Memory that knows *when* something was true, and *why* it believes it.

ChronoFact is a time-aware, verified memory layer for AI assistants, built entirely on Meta models.

Assistants that remember their users fail in two ways: they **store things the user never said**, and they **keep
answering with values that changed months ago**. ChronoFact turns a conversation stream into **verified, time-stamped
facts with provenance**, keeps their history when they change, and answers every question from a small, fixed context.

**+13.6 points over RAG** on LongMemEval · **2.5× on temporal reasoning** · **2.0× across sessions** · **3.9× fewer
false "I don't know"** · **$0 API cost on one GPU**

> 🎬 Demo video: _(link)_

![Live memory: every message streams through the write path; Berlin is superseded by Munich and kept as history](docs/live.png)

## Why ChronoFact

Measured on LongMemEval-S (500 users, ~115k-token histories each, 10,000 judged answers per system), against RAG over
the raw transcripts with the same answering model (Llama-3.1-8B) and the same token budgets. Every number is the
effect of the complete ChronoFact system working together: verified temporal memory, time-scoped retrieval and
packing, and the answer step.

| advantage | evidence |
|---|---|
| **More accurate memory** | **+13.6 points** overall [95% CI +10.2, +16.8], significant at every budget from 256 to 2,048 tokens |
| **Gets time right** | **2.5×** accuracy on temporal reasoning (52.1% vs 20.6%); knowledge updates 75.8% vs 62.6% |
| **Connects sessions** | **2.0×** accuracy on questions that span several sessions (45.3% vs 22.7%) |
| **Does not give up wrongly** | **3.9× fewer** false "not in memory" answers (8.5% vs 32.6% of answerable questions) |
| **Never stores what was not said** | a fact must quote the user verbatim **and** pass an NLI entailment check: **43 unsupported facts blocked per history**. Asking for "tips for a flat near Friedrichshain" does not become "lives near Friedrichshain" |
| **Keeps the history** | a changed value supersedes the old one, which keeps its end date instead of being overwritten, so "now", "in March" and "what changed" all have answers; "two weeks ago" is resolved to a date when written |
| **Bounded as memory grows** | a fixed 256-2,048-token context per question; the lead holds at 12.8-14.8 points from the evidence-only history to the full history with every later session |
| **Cheap to write** | 59 of ~245 user turns per history skip the LLM (salience filter); 359 calls of a local 8B model per ~115k-token history; **$0 API cost** |
| **Auditable and deletable** | every fact links to the user's own words; every write decision is logged; `Store.forget` deletes a fact, its sibling facts, the message and the assistant's reply to it, from memory and from search |
| **Trustworthy numbers** | chronological replay at 5 checkpoints × 4 budgets, paired bootstrap CIs, cross-fitted tuning, byte-identical re-runs (160/160), the LLM judge audited by hand (1 of 50 wrong), 0 errors in the whole run |

What the fact memory adds that retrieval over transcripts cannot provide at all:

| capability | ChronoFact | RAG over transcripts |
|---|---|---|
| current vs past values, with validity dates | ✅ | ❌ |
| facts verified against the user's words before storing | ✅ | ❌ |
| questions scoped in time (now / as of a date / history) | ✅ | ❌ |
| provenance for every remembered fact | ✅ | ❌ |
| forgetting one fact and the messages it came from | ✅ | ❌ |

![Ask across time: "How often did I run in March?" ChronoFact answers from a fact marked "no longer current"; RAG with the same 256-token budget finds nothing](docs/ask.png)

## How it works

```
 every message                                                         every question
 ─────────────                                                         ──────────────
 salience filter ─► extract (Llama 3.1, JSON schema, exact quote)      time scope: now / as of a date / history
        │                         │                                              │
   no claim: skip     verify: quote verbatim + NLI entailment          hybrid search: BM25 + Contriever (RRF)
   (no LLM call)                  │  (BART-MNLI) ─► blocked              │
                      label: kind, slot, richer statement (Llama)      fact gate: facts valid in that scope
                                  │                                              │
                      temporal store: add · deduplicate · merge        pack facts with validity dates + evidence
                      entities · supersede a changed state (old        turns into a fixed token budget
                      value kept with its end date) · close on                   │
                      negation · provenance · forget                   answer (Llama 3.1): FINAL line or NOT IN MEMORY
```

| Meta model | role |
|---|---|
| **Llama-3.1-8B-Instruct** (open weights, local vLLM) | extraction, labelling, negation control, answers, judge, all with schema-constrained JSON |
| **Contriever** (Meta FAIR) | dense retrieval, entity and slot matching |
| **BART-large-MNLI** | write-time verification of every fact and every rich statement; contradiction checks for supersession |
| **Muse Spark** (Meta Model API) | labelled the pairs that calibrated the embedding thresholds; a profile runs every role on it |

## Results

Every history is replayed in time order and evaluated at 5 checkpoints (from the evidence-only history to the full
history) × 4 context budgets: 10,000 answers per system, judged by Llama-3.1-8B (50 sampled verdicts audited by hand:
1 wrong). 95% CIs from 1,000 paired bootstrap resamples of questions. Both systems answer with Llama-3.1-8B from
contexts of the same token budget. RAG over transcripts is the standard baseline: the top transcript chunks (BM25 +
Contriever) and a direct answer. ChronoFact is the full system: the question is scoped in time, verified facts with
validity dates and the evidence turns they came from are packed into the budget, and the model answers from that
memory. The lead below is the combined effect of these parts.

| system | 256 | 512 | 1024 | 2048 | all budgets |
|---|---|---|---|---|---|
| **ChronoFact** | 49.8 | **60.1** | 61.9 | 61.2 | **58.2** [55.0, 61.5] |
| RAG over transcripts | 37.8 | 43.6 | 49.1 | 47.9 | 44.6 [40.8, 48.6] |
| **ChronoFact lead** | **+11.9** | **+16.4** | **+12.8** | **+13.3** | **+13.6** [+10.2, +16.8] |

| question type (n) | ChronoFact | RAG over transcripts |
|---|---|---|
| temporal reasoning (127) | **52.1** | 20.6 |
| multi-session (121) | **45.3** | 22.7 |
| knowledge update (72) | **75.8** | 62.6 |
| single-session user (64) | **88.3** | 85.3 |
| single-session preference (30) | **16.2** | 5.2 |
| single-session assistant (56) | 66.1 | **78.2** |
| abstention (30) | 57.5 | **81.7** |

* ChronoFact leads on five of seven question types. RAG over transcripts wins the 30 unanswerable questions and those
  about what the assistant itself said: it says "not in memory" far more often, which is right on unanswerable
  questions but wrong on 32.6% of the answerable ones (ChronoFact: 8.5%).
* The lead is significant at every budget (every 95% CI above zero), and it holds as the history grows: 59.0% vs
  44.2% at the evidence-only checkpoint, 57.6% vs 44.8% with the full history.
* Integrity audit: 0 problems; 0 write, answer or judge errors; 160/160 re-run retrieval cells byte-identical.

![Results](docs/results.png)

A run writes the full report to `results/final/report.md` (tables, plots, per-type and per-checkpoint numbers, cost).

## Run it

### Requirements

| | minimum | the reported run |
|---|---|---|
| OS | Linux x86_64 | Ubuntu, Linux x86_64 |
| GPU | 1 NVIDIA GPU, compute capability ≥ 8.0 (Ampere or newer, for bf16), ≥ 24 GB memory | RTX PRO 6000 Blackwell, ~52 GB of it (shared with another job) |
| NVIDIA driver | supports CUDA 13 (driver ≥ 580; the pinned PyTorch is built for CUDA 13.0) | 610.43 |
| CPU / RAM | ≥ 16 cores (≥ 32 recommended: embeddings and NLI run on CPU), ≥ 96 GB RAM with the default worker counts (each CPU worker uses ~2 GB; on smaller machines lower `resources.ingest_workers` and `retrieve_workers` in `config.yaml`) | 64 cores, 503 GB RAM |
| disk | ~30 GB free (models ~18 GB, dataset 0.3 GB, one run ~6 GB) | |
| software | Python 3.12, `git`; internet access during setup (pip, Hugging Face, the dataset) | |

Only the LLM server uses the GPU. Time for the full run on the reported setup: ingest ~3.7 h, answers ~1 h, judge
~0.5 h, everything else minutes; a smaller GPU or fewer cores take proportionally longer.

### Steps

```bash
git clone <this repo> chronofact && cd chronofact
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                          # pinned versions (PyTorch, vLLM, ...)
source scripts/env.sh                                     # environment for every command (run it in each new shell)
HF_HUB_OFFLINE=0 python scripts/download_models.py        # Contriever, BART-MNLI, Llama-3.1-8B at pinned commits
nohup bash scripts/serve_llama.sh > /dev/null 2>&1 &      # vLLM on 127.0.0.1:8017; logs in logs/vllm.log
until curl -sf 127.0.0.1:8017/v1/models > /dev/null; do sleep 10; done   # wait until the model is loaded
python -m pytest -q tests                                 # 22 tests
python scripts/chronofact.py all                          # downloads LongMemEval-S, then every step in order
```

`serve_llama.sh` options: `CF_GPU` (GPU index, default 0), `CF_GPU_MEM` (share of GPU memory vLLM may use, default
0.85), `CF_PORT` (default 8017; keep `llm.providers.local` in `config.yaml` in step).

`all` is resumable: every unit writes atomically and every LLM call, embedding and NLI result is cached, so a crash
or a rerun only redoes the units in flight. Three steps stop for a human review (exit code 3) and print the command
that continues; after it, run `python scripts/chronofact.py all` again:

| step | review | then |
|---|---|---|
| serve_check | `results/serve_check.json` (20 extractions, 20 answers) | `python scripts/chronofact.py serve_ok "<notes>"` |
| pilot | `results/pilot_review.json` (30 histories) | `python scripts/chronofact.py pilot_ok "<notes>"` |
| judge_pilot | `results/judge_pilot_sample.json` (50 verdicts; count the wrong ones) | `python scripts/chronofact.py handcheck <wrong_of_50> "<notes>"` |

Outputs go to `results/final/` (report, tables, plots, `runs.parquet`, `integrity_audit.json`, which must report 0
problems); `python scripts/bundle.py` makes a flat copy in `results/bundle/`. Rerunning on the same machine replays
every call from the caches and reproduces the results byte for byte. On other hardware, vLLM's batched greedy decoding
can change a few generated tokens, so expect small differences in the numbers; the confidence intervals show their
scale.

| problem | fix |
|---|---|
| `[llm] ...: APIConnectionError; retrying` | the LLM server is not up yet or crashed: `tail logs/vllm.log` (the pipeline waits up to 2 h) |
| vLLM runs out of GPU memory | lower `CF_GPU_MEM` or free the GPU; a 24 GB GPU needs most of its memory |
| a step stops with exit code 3 | a review gate: run the printed command, then `all` again |
| `unknown step` | the valid steps and commands are printed |

### Profiles and Muse Spark

`config.yaml` maps every role to a model; a profile switches them (`CF_PROFILE=...`, results in `results_<profile>/`):

| profile | write path + answers | judge | notes |
|---|---|---|---|
| `local` (default, the reported run) | Llama-3.1-8B | Llama-3.1-8B | no API key needed |
| `hybrid` | Llama-3.1-8B | Muse Spark 1.3 | only short question/answer texts reach the API |
| `spark` | Muse Spark 1.3 contributor tier | Muse Spark 1.3 | batched extraction and labeling for the 150 req/min tier (the `batch_check` step A/B-tests that batching loses no facts), Spark-calibrated thresholds, a cost gate |

Set `META_API_KEY` for `hybrid` and `spark`. The API layer includes a **policy guard**: a request the API refuses
(LongMemEval contains conversations that can trip a content filter) is recorded and never sent again, nothing is ever
re-sent in another form, a 401/403 stops every process at once, and after 3 refusals nothing more reaches the API
until `policy_stop.json` in the results directory is reviewed and deleted. A spend cap stops all paid calls.
`scripts/calibration/` holds the calibrations behind the thresholds in `config.yaml`.

### Demo app

```bash
bash scripts/serve_demo.sh     # http://127.0.0.1:8090 (needs the LLM server and a finished run in results/)
```

Live memory (chat; every message streams through the real write path), ask across time, a benchmark explorer over all
500 histories with each system's context, answer and verdict, results, and how it works. `demo/tests/` holds a
180-check end-to-end UI test (Chrome + Node 22+).

## Repository

```
chronofact/   config, LLM layer (cache, rate limits, policy guard), models, store (snapshots, forget),
              write path, W2 reconciliation, read path, answers + judge, pipeline units
scripts/      chronofact.py (the runner), report, audit, bundle, download_models, serve_llama, serve_demo, env,
              calibration/
demo/         demo server, scripted story, web app, UI tests
tests/        unit tests
data/         splits.json (the fixed halves for cross-fitted tuning); the dataset downloads to data/raw/
docs/         screenshots
model_pins.json   every model at an exact commit
```

Code comments such as `README 5.4` cite sections of the design spec the pipeline was built from; every rule they
refer to is implemented and documented in the code itself.

## Built during the hackathon (Oct 26 – Nov 4, 2026)

The ChronoFact write path (salience filter, extraction with quote and NLI verification, labelling, temporal update
and supersession, negation control), the temporal fact store with snapshots, provenance and forgetting, W2
reconciliation, the time-aware read path, the evaluation pipeline (chronological checkpoints, budgets, cross-fitted
tuning, bootstrap CIs, integrity audit, judge audit), the threshold calibrations, the Meta API policy guard, and the
demo app.

## Used as dependencies

**Models:** Llama-3.1-8B-Instruct (Llama 3.1 Community License; weights from the ungated mirror
`unsloth/Llama-3.1-8B-Instruct`; Built with Llama), Contriever (`facebook/contriever-msmarco`), BART-large-MNLI
(`facebook/bart-large-mnli`), Muse Spark (Meta Model API). **Data:** LongMemEval-S cleaned
(`xiaowu0162/longmemeval-cleaned`). **Libraries:** see `requirements.txt` (vLLM, PyTorch, Transformers, FAISS,
rank-bm25, dateparser, pandas, FastAPI).
