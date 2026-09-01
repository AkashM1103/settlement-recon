# Settlement Reconciliation + Q&A Agent

Finance teams reconcile payment-gateway settlements against merchant orders by hand: open two
spreadsheets, match payment IDs, then chase the handful of rows that don't line up. This agent
ingests a batch of Razorpay-style settlement records and merchant order records, auto-matches them,
explains every exception in plain language, and answers natural-language questions about the batch
(fees, tax, net payout, refunds, orphans) with a record-level audit trail.

Not-100%-matching is the point: the sample batch deliberately contains orphan settlements, duplicate
settlements, partial-refund amount mismatches, dummy transactions and delayed payouts, and the
report separates "broken" from "merely delayed".

## Quickstart (under 2 minutes)

```bash
git clone <repo-url> && cd settlement-recon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py      # UI: preview data -> run reconciliation -> report -> chat
# or
python main.py --demo-questions   # CLI: full report + sample Q&A, writes out/*.csv
```

Sample data is committed (`data/orders.csv`, `data/settlements.csv`, `data/ground_truth.csv`), so a
fresh clone runs end-to-end with no setup and no API key.

**Optional LLM layer:** copy `.env.example` to `.env` and set `GROQ_API_KEY` to have the LLM
(`openai/gpt-oss-120b` on Groq by default) do the fuzzy-match reasoning and phrase the Q&A answers. Without a key the same interfaces are served
by a deterministic scorer, so nothing in the demo breaks — the report shows which reasoner ran.

Regenerate the batch with different noise: `python scripts/generate_data.py --seed 42`.

## Architecture

```
 orders.csv ┐
            ├─> Ingestion ─> Normalizer ─> Matcher ──────────────> Reasoner ─┐
settlements ┘   (load +      (types,       pass 1: exact          (LLM:      │
                 schema       dates,        payment_id             match /   │
                 checks)      currency,     pass 2: candidate      no_match / │
                              derived       retrieval by           uncertain  │
                              net check)    amount ±5% & date ±3d  + reason   │
                                                                   + conf.)   │
                                                                              v
                                                        Reporter / Q&A ── Streamlit UI
                                                        (match rate, false-match    + CLI
                                                         rate, exception table,
                                                         grounded chat w/ citations)
```

| Module | File | Job |
| --- | --- | --- |
| Ingestion | `src/recon/ingest.py` | Load CSVs, verify required columns, sanity counts |
| Normalizer | `src/recon/normalize.py` | Column aliases, `₹`/comma stripping, date parsing, `net = gross - fees - tax` check |
| Matcher | `src/recon/match.py` | Pass 1 exact `payment_id`; pass 2 candidate retrieval (amount ±5%, date window, top-3); pass 3 reasoner verdict; exception classification |
| Reasoner | `src/recon/reasoner.py` | LLM verdict (`match`/`no_match`/`uncertain` + confidence + one-line reason), deterministic fallback |
| Reporter | `src/recon/report.py` | Match rate, false-match rate vs ground truth, exception breakdown, money totals, latency |
| Q&A | `src/recon/qa.py` | Retrieve relevant reconciled records -> pandas aggregates -> LLM phrases the answer, always citing record IDs |
| UI | `app.py`, `main.py` | Streamlit app and CLI |

Retrieval note: exact `payment_id` matches need no similarity search at all, so the retrieval layer
runs only over settlements that lost their link — a numeric amount/date window producing the top-3
candidate orders, which are then judged by the reasoner. Matched orders are consumed, so one order
can't be claimed by two settlements.

## Metrics (bundled 75-settlement / 74-order batch, deterministic reasoner)

| Metric | Value |
| --- | --- |
| Match rate (confidence ≥ 0.80) | 94.7% |
| Exact `payment_id` matches | 93.3% |
| Recovered by retrieval + reasoning | 3 settlements |
| False-match rate vs ground truth | 0.00% (0 wrong, 0 missed) |
| Exceptions flagged | 13 |
| Latency / throughput | 0.05 s for the batch (~1.6k settlements/s, no LLM) |

Exception breakdown:

| Reason | Count | Meaning |
| --- | --- | --- |
| `orphan_settlement` | 3 | No `payment_id`; recovered via amount/date retrieval |
| `amount_mismatch` | 3 | Partial refund not reflected in the order amount |
| `date_drift` | 3 | Settled >7 days after the order — delayed, not broken |
| `unsettled_order` | 3 | Order with no settlement in this batch |
| `duplicate_settlement` | 2 | Same `payment_id` settled twice (held as uncertain, not auto-matched) |
| `no_order` | 2 | Settlement references a payment absent from orders (test/dummy) |

Ground truth is emitted by the generator (`data/ground_truth.csv`), so the false-match rate is a real
measurement, not an estimate. Run `pytest` to re-verify these properties.

## Q&A audit trail

```
$ python main.py --ask "Which settlements have no matching order?"
A: 2 record(s) match this question: STL5571, STL5572.
   cited: STL5571, STL5572
```

Questions are answered by querying the reconciled dataset, not by free-form retrieval over raw text:
the retriever picks the relevant slice (by record ID, exception type, or status), pandas computes the
figures, and the LLM only phrases what it is given. The evidence blob behind each answer is viewable
in the UI.

## Known limitations

- Candidate retrieval is a numeric amount/date window, not embeddings — with these fields, an
  embedding index adds latency without adding signal. It would matter with free-text narration fields
  (bank remarks, merchant descriptors).
- Greedy one-pass assignment: an order is consumed by the first settlement that claims it, rather
  than solving a global optimal assignment.
- The `date_drift` and `amount_mismatch` thresholds (7 days, 1%) are Razorpay-flavoured defaults, not
  learned; they are exposed as sliders/CLI flags for tuning.
- Ground truth exists only because the data is synthetic; on real batches the false-match rate must
  come from sampled manual review.
- Multi-currency is normalized but not converted — a non-INR batch would need FX handling.
- The LLM reasoner is called per unlinked settlement with no batching or caching; on a large batch
  that dominates latency.

## Tests

```bash
pytest        # 8 tests: pipeline invariants, exception coverage, normalizer, Q&A grounding
```
