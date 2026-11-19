"""Durable progress for a single blinded reviewer, shared by the CLI and UI."""

import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock


def read_grades(path: Path, sample_ids: set[str]) -> dict[str, int]:
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("judge") != "human" or not isinstance(document.get("labels"), list):
        raise ValueError("Expected a human review file with a list of labels")
    grades = {}
    for row in document["labels"]:
        key, grade = row["sample_id"], row["grade"]
        if key not in sample_ids or key in grades:
            raise ValueError(f"Unknown or duplicate sample ID: {key}")
        if type(grade) is not int or grade not in range(4):
            raise ValueError(f"Expected an integer grade from 0 to 3 for {key}")
        grades[key] = grade
    return grades


def save_grade(path: Path, sample_ids: set[str], key: str, grade: int) -> None:
    if key not in sample_ids or type(grade) is not int or grade not in range(4):
        raise ValueError("Expected a sampled pair and an integer grade from 0 to 3")
    with FileLock(str(path) + ".lock", timeout=10):
        grades = read_grades(path, sample_ids)
        grades[key] = grade
        document = {"judge": "human", "labels": [
            {"sample_id": k, "grade": grades[k]} for k in sorted(grades)
        ]}
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            temp_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
