
import os
import json
import time
import random
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
MODEL = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")

if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You are a precise electricity bill analyst for Pakistan.
Supported providers: KE, LESCO and IESCO.

Use ONLY the original bill image and supplied extracted data.
The original image is authoritative if the JSON is wrong.

ACCURACY:
- Never invent or assume bill values.
- Read printed values rather than handwritten annotations.
- Do not disclose names, addresses, account numbers or IDs.
- Use printed dates, charges, taxes and payment amounts.
- Keep distinct late-payment windows separate.
- Do not confuse current bill with final payable amount.
- Never use external tariff rates.
- If data is missing or ambiguous, say so.

HISTORICAL DATA:
- Read every month in the bill history or usage chart.
- Transcribe month, year and units carefully.
- Pay special attention to values near thresholds.
- Compare numerical values, not chart bar heights alone.
- For "above", use strictly greater than.
- For "below", use strictly less than.
- Verify that counts match the months you list.
- Do not omit months such as 203 units when checking
  whether usage exceeds 200 units.
- Distinguish historical usage from current consumption.
- Preserve negative historical values.
- Do not invent missing months.

NET METERING:
- Multiple meter rows may represent different registers.
- Never add unidentified meter rows together and call
  their sum total consumption.
- Identify import, export and net units only when supported
  by the original bill.
- Explain uncertainty rather than guessing.

PAYMENTS AND CREDITS:
- CR indicates a credit balance.
- Negative arrears may represent an existing credit.
- "NOT TO BE PAID" means no payment is currently required.
- Never treat a credit balance as an outstanding payment.
- Distinguish current charges from remaining credit.

CALCULATIONS:
- Verify arithmetic before answering.
- Show short calculations when relevant.
- Do not combine unrelated measurements.
- Distinguish exact values from estimates.
- For hypothetical questions, state assumptions.
- Do not invent future rates or guaranteed savings.

OUTPUT:
- Answer the actual question directly.
- Use concise, understandable English.
- Return only the answer, with no internal reasoning.
"""


def _image_part(image_path):
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime = mimetypes.guess_type(path.name)[0]

    if mime not in ("image/jpeg", "image/png"):
        raise ValueError(f"Unsupported image type: {path}")

    return types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type=mime
    )


def _generate(prompt, image_path, json_mode=False):
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type=(
            "application/json" if json_mode else "text/plain"
        )
    )

    last_error = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt, _image_part(image_path)],
                config=config
            )

            if not response.text or not response.text.strip():
                raise RuntimeError("Empty Gemini response")

            return response.text.strip()

        except Exception as exc:
            last_error = exc
            message = str(exc).lower()

            retryable = any(
                term in message
                for term in (
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "unavailable",
                    "resource_exhausted",
                    "deadline",
                    "timeout"
                )
            )

            if not retryable or attempt == 4:
                raise

            delay = min(2 ** (attempt + 1), 20)
            delay += random.uniform(0, 1)

            print(
                f"Gemini temporary error. "
                f"Retry {attempt + 1}/5 in {delay:.1f}s"
            )

            time.sleep(delay)

    raise RuntimeError(str(last_error))


def answer_question(image_path, bill_data, question):
    if not isinstance(bill_data, dict):
        raise TypeError("bill_data must be a dictionary")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    prompt = f"""
{SYSTEM_PROMPT}

EXTRACTED BILL JSON:
{json.dumps(bill_data, ensure_ascii=False)}

QUESTION:
{question}

Check the original image, especially historical tables,
before providing your final answer.
"""

    return _generate(prompt, image_path)


def answer_questions_batch(image_path, bill_data, questions):
    """
    questions: list of dictionaries with
    question_id and question.

    Returns:
    dictionary mapping question_id -> answer.
    """

    if not questions:
        return {}

    normalized = []

    for item in questions:
        question_id = str(item["question_id"])
        question = str(item["question"])

        normalized.append({
            "question_id": question_id,
            "question": question
        })

    prompt = f"""
{SYSTEM_PROMPT}

You must answer every supplied question independently.

IMPORTANT:
1. Examine the original bill image thoroughly.
2. Read all 12 or 13 historical months where available.
3. Verify threshold counts and numerical comparisons.
4. Distinguish bill totals from individual components.
5. Do not invent data.
6. Return valid JSON only.

EXTRACTED BILL:
{json.dumps(bill_data, ensure_ascii=False)}

QUESTIONS:
{json.dumps(normalized, ensure_ascii=False)}

REQUIRED JSON FORMAT:
{{
  "answers": [
    {{
      "question_id": "Q1",
      "answer": "Complete customer-facing answer"
    }}
  ]
}}

Include exactly one answer for each supplied question.
Do not include any personal identifying information.
"""

    raw = _generate(
        prompt,
        image_path,
        json_mode=True
    )

    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")

    answers = data.get("answers")

    if not isinstance(answers, list):
        raise ValueError("Missing answers list")

    result = {}

    for item in answers:
        if not isinstance(item, dict):
            continue

        question_id = str(item.get("question_id", ""))
        answer = item.get("answer", "")

        if isinstance(answer, str) and answer.strip():
            result[question_id] = answer.strip()

    expected_ids = {
        item["question_id"] for item in normalized
    }

    if set(result) != expected_ids:
        missing = expected_ids - set(result)

        raise ValueError(
            f"Missing or unexpected batch answers: {missing}"
        )

    return result


if __name__ == "__main__":
    image = Path("train/bills/KESC_0008.png")
    data_path = Path("training_results/KESC_0008.json")

    with data_path.open(encoding="utf-8") as file:
        bill = json.load(file)

    questions = [
        "How much of my bill is taxes?",
        "How many months in my usage history went above 200 units?"
    ]

    for question in questions:
        print("\nQUESTION:", question)
        print(
            "ANSWER:",
            answer_question(image, bill, question)
        )
        print("-" * 60)
