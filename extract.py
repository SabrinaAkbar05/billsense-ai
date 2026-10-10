
"""Image-grounded electricity bill extraction."""

from __future__ import annotations

import json
import mimetypes
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv(
    "MODEL_NAME",
    "gemini-3.1-flash-lite"
)

KEYS = [
    "provider",
    "tariff",
    "sanctioned_load_kw",
    "bill_month",
    "reading_date",
    "issue_date",
    "due_date",
    "previous_reading",
    "current_reading",
    "units_consumed",
    "charges",
    "total_charges",
    "taxes",
    "total_taxes",
    "current_bill",
    "arrears",
    "payable_within_due_date",
    "payable_after_due_date",
]

PROMPT = """
Extract the ATTACHED Pakistani electricity bill image
into exactly ONE JSON object.

REQUIRED KEYS:
provider, tariff, sanctioned_load_kw, bill_month,
reading_date, issue_date, due_date, previous_reading,
current_reading, units_consumed, charges, total_charges,
taxes, total_taxes, current_bill, arrears,
payable_within_due_date, payable_after_due_date.

GENERAL RULES:
- Use only values visible in the original bill image.
- Never invent values or use external tariff knowledge.
- Never include customer names, addresses, account IDs,
  reference numbers or other personal identifiers.
- Unreadable or ambiguous scalar values must be null.
- Dates must use YYYY-MM-DD.
- bill_month must use YYYY-MM.
- Numeric values must be JSON numbers, not strings.
- Return all required keys, with no additional keys.
- Missing charges or taxes must be empty arrays.

CHARGES:
Each item must be:
{"type": "allowed_type", "amount": number}

Allowed charge types:
energy, fixed, fpa, quarterly_adjustment, surcharge,
meter_rent, subsidy, other.

TAXES:
Allowed tax types:
gst, electricity_duty, income_tax, municipal_tax,
other_tax.

ACCOUNTING RULES:
- Identify gross electricity charges, subsidies,
  net electricity charges, FPA, taxes and current bill
  as DIFFERENT printed concepts.
- Do not confuse a subtotal with an individual charge.
- Do not count both a subtotal and its components.
- Do not force charge components to add up to a subtotal
  if the bill displays adjustments separately.
- total_charges should reflect the printed electricity
  charges total or clearly identified charges subtotal.
- total_taxes should reflect the printed taxes total.
- current_bill should reflect the printed current bill.
- Preserve printed values even if totals differ because
  of billing conventions or rounding.
- Do not calculate missing totals.

SUBSIDIES:
- If the bill prints gross electricity charges of 7711,
  a subsidy of 1002 and net electricity charges of 6709,
  represent the subsidy as -1002.
- Do not label 6709 as gross energy charges if it is
  explicitly the net amount.
- Do not subtract the same subsidy twice.
- These numbers are illustrative, not default values;
  extract the actual numbers from each image.

FPA:
- Fuel Price Adjustment may be displayed separately
  from current electricity charges.
- Include an FPA entry only if its amount is printed.
- Do not automatically add FPA to total_charges.
- Do not automatically add FPA to current_bill.
- Do not infer FPA from payable differences.

TAXES:
- Extract individual printed tax lines when available.
- If only aggregate taxes are printed, use one
  other_tax entry with the printed amount.
- Never count aggregate taxes and their detail twice.

READINGS:
- Read previous reading, current reading and consumed
  units from the correct meter or consumption section.
- For net metering, multiple rows may represent import,
  export and other registers.
- Never sum arbitrary meter rows.
- If the single requested scalar cannot be determined
  unambiguously, return null.

PAYMENT AMOUNTS:
- Distinguish current bill, arrears, amount within due
  date and amount after due date.
- Use PRINTED values only.
- Ignore handwritten notes or annotations.
- If multiple late-payment amounts are printed,
  use the final/latest applicable late-payment amount
  for payable_after_due_date.
- Do not infer a late-payment amount from arithmetic.
- Preserve negative arrears and credits.
- If the bill explicitly prints a credit with CR,
  represent the amount as negative where appropriate.
- If it only says NOT TO BE PAID without a numeric
  amount, use null.

PROVIDER:
- Read the printed company name.
- Keep the provider value concise and consistent,
  such as KE, IESCO or LESCO, when clearly supported
  by the printed bill.

FINAL CHECK:
- Verify every numeric value against the image.
- Do not invent charges to reconcile arithmetic.
- Return strict JSON only.
"""


def extract_bill(image_path: str | Path) -> dict:
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Bill image not found: {path}"
        )

    mime = mimetypes.guess_type(path.name)[0]

    if mime not in ("image/jpeg", "image/png"):
        raise ValueError(
            f"Unsupported image format: {path}"
        )

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY in .env"
        )

    client = genai.Client(api_key=api_key)

    image_part = types.Part.from_bytes(
        data=path.read_bytes(),
        mime_type=mime,
    )

    last_error = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[PROMPT, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )

            if not response.text:
                raise RuntimeError(
                    "Empty extraction response"
                )

            result = json.loads(response.text)

            if not isinstance(result, dict):
                raise ValueError(
                    "Expected a JSON object"
                )

            for key in KEYS:
                default = (
                    [] if key in ("charges", "taxes")
                    else None
                )
                result.setdefault(key, default)

            return {
                key: result[key]
                for key in KEYS
            }

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
                    "timeout",
                    "deadline",
                )
            )

            if not retryable or attempt == 4:
                raise

            delay = min(2 ** (attempt + 1), 20)
            delay += random.uniform(0, 1)

            print(
                f"Temporary Gemini error. "
                f"Retrying in {delay:.1f}s..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Extraction failed: {last_error}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=Path,
    )

    args = parser.parse_args()

    result = extract_bill(args.image)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
