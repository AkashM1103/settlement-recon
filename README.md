# Settlement Reconciliation and Q&A

This project compares payment settlement rows with merchant orders. It marks exact matches, tries a small amount-and-date search for missing payment links, and lists rows that need a person to review.

The included CSVs are generated sample data. This is a learning demo, not a live payment system, and its results do not prove accuracy on real merchant data.

Live Demo Link: https://settlement-recon-jtch7w7iyykh3g6pffhxnv.streamlit.app/

## What happens when it runs

1. Read the orders and settlements CSV files and check their columns and IDs.
2. Clean column names, currency labels, amounts, and dates. Reject invalid or ambiguous input. Warn when `gross - fees - tax` does not equal `net` within five paise.
3. Match exact payment IDs first. For settlements without a usable link, look for orders with a nearby amount and date, then ask either the simple rules or optional Groq model to judge the shortlist.
4. Calculate totals and exception counts, save CSV reports, and optionally answer questions from the result rows.

A payment ID may appear on multiple settlements; that is how the demo represents a duplicate payout. An order ID and a non-empty order payment ID must each be unique.

## Run it on Windows

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --demo-questions --no-llm
```

To open the web app:

```powershell
streamlit run app.py
```

Upload both `orders.csv` and `settlements.csv`, or leave both upload fields empty to use the sample batch. Uploading only one file shows an error instead of silently using the sample data.

## Share a demo on the same Wi-Fi network (Windows)

From PowerShell, open the project folder and run:

```powershell
.\run-network-demo.ps1
```

The script prints an address such as `http://192.168.1.25:8501`. Open that address on another laptop connected to the same Wi-Fi. Keep the PowerShell window open while the demo is in use. If Windows Firewall asks, allow Python on your private network. Do not expose this demo to public networks; it has no login protection. For a public link, deploy the repository to a hosting service such as Streamlit Community Cloud.
## Run it on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --demo-questions --no-llm
```

Then run `streamlit run app.py` to open the web app.

## Run options

The CLI prints the reconciliation summary and writes `reconciliation.csv`, `exceptions.csv`, and `unsettled_orders.csv` into `out/` by default. Use `--help` to see all options. For example:

```bash
python main.py --no-llm --amount-tolerance 0.05 --date-window 3 \
  --delayed-settlement-days 7 --amount-mismatch-tolerance 0.01
```

The default candidate search allows a settlement one day before an order through ten days after it (three base days plus seven delayed days). The candidate amount window starts near the settlement gross amount and allows a wider upper range for possible partial refunds. Amount mismatch and delayed-payout thresholds can be set from the CLI or Streamlit sidebar.

Regenerate the sample data with:

```bash
python scripts/generate_data.py --seed 42
```

## Optional Groq matching

Without an API key, the project uses its deterministic matching rules. To use Groq for fuzzy-match decisions, copy `.env.example` to `.env` and set a real `GROQ_API_KEY`. The LLM is used only for those matching decisions. Q&A stays rule-based so totals and listed records come directly from the reconciled rows. Never commit `.env` or share its key.

## Project files

- `main.py`: command-line entry point and report export.
- `app.py`: Streamlit screens for input, results, and Q&A.
- `src/recon/ingest.py`: CSV loading, column cleanup, and ID checks.
- `src/recon/normalize.py`: amount/date cleanup and validation.
- `src/recon/match.py`: exact matching, candidate search, and exception labels.
- `src/recon/reasoner.py`: deterministic or optional Groq decision for fuzzy matches.
- `src/recon/report.py`: counts, rates, financial totals, and text report.
- `src/recon/qa.py`: simple question routing and answers from report data.
- `src/recon/pipeline.py`: connects the project steps.
- `scripts/generate_data.py`: creates the sample orders, settlements, and ground truth.
- `tests/`: checks matching, input handling, and Q&A behavior.

## Tests

Run the tests from this folder:

```bash
pytest
```

## Limits to keep in mind

- Candidate search uses amount and date rules, not a learned similarity model.
- Assignment is one settlement at a time; it does not calculate a globally optimal set of matches.
- This demo reports INR only. It does not convert other currencies.
- Ground truth is available only for generated examples. Real batches need a reviewed sample to estimate match quality.
- The LLM can still make a poor fuzzy-match decision. Keep low-confidence and exceptional rows for human review.

