
import csv
import json
import time
import argparse
from pathlib import Path

from extract import extract_bill
from answer_questions import (
    answer_question,
    answer_questions_batch,
)

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "test"
OUTPUT = ROOT / "output"
CACHE = ROOT / ".cache" / "competition"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")

    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def find_image(bill_id):
    for extension in IMAGE_EXTENSIONS:
        candidate = TEST / "bills" / (bill_id + extension)

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Missing bill image: {bill_id}")


def extract_with_cache(bill_id, image):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{bill_id}.json"

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    result = extract_bill(image)

    if not isinstance(result, dict):
        raise ValueError(f"Invalid extraction for {bill_id}")

    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result


def answer_group(image, bill, questions):
    if len(questions) == 1:
        q = questions[0]
        return {
            str(q["question_id"]): answer_question(
                image, bill, q["question"]
            )
        }

    try:
        return answer_questions_batch(image, bill, questions)

    except Exception as error:
        print(f"Batch failed; splitting: {error}")

        midpoint = len(questions) // 2

        left = answer_group(
            image, bill, questions[:midpoint]
        )

        right = answer_group(
            image, bill, questions[midpoint:]
        )

        return {**left, **right}


def run():
    level1_template = read_csv(TEST / "csv" / "level1.csv")
    level2_template = read_csv(TEST / "csv" / "level2.csv")

    if len(level1_template) != 10:
        raise ValueError("Expected 10 Level 1 rows")

    if len(level2_template) != 120:
        raise ValueError("Expected 120 Level 2 rows")

    bill_ids = [row["bill_id"] for row in level1_template]

    if len(set(bill_ids)) != 10:
        raise ValueError("Duplicate bill IDs")

    questions_by_bill = {}

    for row in level2_template:
        questions_by_bill.setdefault(row["bill_id"], []).append(row)

    if set(questions_by_bill) != set(bill_ids):
        raise ValueError("Question bill IDs do not match Level 1")

    OUTPUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    # LEVEL 1
    extracted = {}
    level1_rows = []

    for index, bill_id in enumerate(bill_ids, start=1):
        print(f"[Level 1] {index}/10: {bill_id}", flush=True)

        image = find_image(bill_id)
        data = extract_with_cache(bill_id, image)

        extracted[bill_id] = data

        level1_rows.append({
            "bill_id": bill_id,
            "json": json.dumps(
                data, ensure_ascii=False, separators=(",", ":")
            ),
        })

        write_csv(
            OUTPUT / "level1.csv",
            ["bill_id", "json"],
            level1_rows,
        )

    # LEVEL 2
    answers_path = CACHE / "answers.json"

    if answers_path.exists():
        answers = json.loads(
            answers_path.read_text(encoding="utf-8")
        )
    else:
        answers = {}

    for index, bill_id in enumerate(bill_ids, start=1):
        print(f"[Level 2] {index}/10: {bill_id}", flush=True)

        image = find_image(bill_id)
        bill = extracted[bill_id]

        questions = questions_by_bill[bill_id]

        pending = [
            q for q in questions
            if not str(
                answers.get(
                    f"{bill_id}:{q['question_id']}", ""
                )
            ).strip()
        ]

        # Small batches reduce missing or confused answers.
        for start in range(0, len(pending), 3):
            group = pending[start:start + 3]

            result = answer_group(image, bill, group)

            for question_id, answer in result.items():
                answers[f"{bill_id}:{question_id}"] = answer

            answers_path.write_text(
                json.dumps(
                    answers, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )

            completed = sum(
                bool(str(value).strip())
                for value in answers.values()
            )

            print(f"  Saved answers: {completed}/120", flush=True)

            # Save the CSV after each batch.
            save_level2(level2_template, answers)

    save_level2(level2_template, answers)

    final_level2 = read_csv(OUTPUT / "level2.csv")

    empty = [
        row for row in final_level2
        if not row["answer"].strip()
    ]

    if empty:
        raise RuntimeError(
            f"{len(empty)} answers are still missing"
        )

    print("\nSUCCESS: 10 extractions and 120 answers generated.")
    print("Files:")
    print(OUTPUT / "level1.csv")
    print(OUTPUT / "level2.csv")


def save_level2(template, answers):
    rows = []

    for row in template:
        bill_id = row["bill_id"]
        question_id = row["question_id"]

        rows.append({
            "bill_id": bill_id,
            "question_id": question_id,
            "question": row["question"],
            "answer": answers.get(
                f"{bill_id}:{question_id}", ""
            ),
        })

    write_csv(
        OUTPUT / "level2.csv",
        ["bill_id", "question_id", "question", "answer"],
        rows,
    )


def train_check():
    images = sorted(
        p for p in (ROOT / "train" / "bills").iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image in images:
        print(f"Checking {image.name}", flush=True)
        result = extract_bill(image)

        print(
            result.get("provider"),
            result.get("units_consumed"),
        )

    print(f"Training check complete: {len(images)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-check",
        action="store_true",
    )

    args = parser.parse_args()

    if args.train_check:
        train_check()
    else:
        run()
