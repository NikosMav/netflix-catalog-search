from pathlib import Path

import pytest

from retrieval import review

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_blind_review_resumes_in_a_new_session(tmp_path, monkeypatch):
    progress = tmp_path / "human.json"
    read, save = review.read_grades, review.save_grade
    monkeypatch.setattr(review, "read_grades", lambda path, keys: read(progress, keys))
    monkeypatch.setattr(review, "save_grade", lambda path, keys, key, grade: save(progress, keys, key, grade))

    def open_review():
        app = AppTest.from_file(str(APP))
        app.query_params["page"] = "review"
        app.run()
        assert not app.exception
        return app

    def sample_id(app):
        return next(c.value for c in app.caption if c.value.startswith("sample_id:"))

    first = open_review()
    first_id = sample_id(first)
    next(b for b in first.button if b.label == "Grade 2").click().run()
    assert not first.exception
    assert sample_id(first) != first_id
    assert sample_id(open_review()) == sample_id(first)
