# BillSense AI — Women TechQuest 2026 Utility Bill Decoder

BillSense AI reads KE/K-Electric, LESCO and IESCO electricity bill images and produces the required Level 1 JSON and Level 2 answers. It also includes an optional Streamlit demonstration UI.

## Requirements

- Python 3.11
- An allowed Gemini model and valid API key (default: `gemini-3.1-flash-lite`)
- Test templates and bill images supplied by the organizers

## Setup

1. Copy `.env.example` to `.env` and replace the placeholder with your own API key. Never submit `.env`.
2. Run `python -m pip install -r requirements.txt`.
3. Place training images under `train/bills/`.
4. Place test images under `test/bills/` and official CSV templates under `test/csv/`.

## Run

- `python main.py --train-check` — run training extraction (uses cached results when available).
- `python main.py` — create `output/level1.csv` and `output/level2.csv` using the official test templates.
- `python package_submission.py` — validate rows/columns and create `BillSense_AI_submission.zip` with `output/` and `source/`.
- `python -m streamlit run app.py` — run the optional interactive demo.

## Reliability and known limitations

- Batch requests include retry/backoff and disk checkpoints under `.cache/billsense`.
- Level 2 groups questions per bill to reduce image requests. If the model cannot produce all answers, no misleading complete submission is generated.
- Extraction checks JSON shape but does **not** guarantee every printed field is correct. Review visually, especially net-metering bills, credits, and usage-history questions.
- A bill can show multiple meter registers with no unique single consumption field. In this case the extraction prompt directs the model to return `null` rather than arbitrarily choose a register.
- If you change the prompt/model and want fresh results, delete only the appropriate `.cache/billsense/level1/` and/or `level2/` files; existing cached results otherwise remain in use.
- This software does not automatically verify that every question answer is factually correct. Spot-check against bill images before submitting.

## AI assistance disclosure

Development used ChatGPT (OpenAI) for architecture, coding, review, prompt engineering and documentation. Runtime bill extraction and answering use Google's Gemini 3.1 Flash-Lite model via the google-genai SDK. Human review and testing were performed on training examples. No secrets are included in the submission.

## Submission layout

```
output/level1.csv
output/level2.csv
source/README.md
source/.env.example
source/requirements.txt
source/extract.py
source/answer_questions.py
source/main.py
source/app.py
source/package_submission.py
```
