from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_previous_review_link_displays_completed_audit():
    app = AppTest.from_file(str(APP))
    app.query_params["page"] = "review"
    app.run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Judge audit"
    assert [metric.value for metric in app.metric] == ["83.2%", "98.0%"]
    assert len(app.dataframe[0].value) == 51


def test_evaluation_ranking_tracks_the_selected_method_and_query():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Evaluation").run()
    app.button[0].click().run()

    def ranking_visible():
        assert not app.exception
        return any("labeled relevant titles appear" in item.value for item in app.caption)

    assert ranking_visible()
    method = app.selectbox(key="eval_method")
    method.set_value("TF-IDF").run()
    assert not ranking_visible()
    app.button[0].click().run()
    assert ranking_visible()
    app.selectbox(key="eval_query_test").set_value("adult_cartoon_hasbeen").run()
    assert not ranking_visible()
