from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from retrieval.dense import DenseRetriever


def test_concurrent_index_builds_reuse_one_complete_cache(tmp_path, monkeypatch):
    catalog = pd.DataFrame({"text": ["first", "second"]})
    calls = []
    expected = np.eye(2, dtype=np.float32)

    def encode(self, texts):
        calls.append(texts)
        return expected

    monkeypatch.setattr(DenseRetriever, "_encode", encode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        builds = [pool.submit(DenseRetriever, catalog, cache_dir=tmp_path) for _ in range(2)]
        for build in builds:
            np.testing.assert_array_equal(build.result().embeddings, expected)
    assert calls == [["first", "second"]]
    path, = tmp_path.glob("*.npy")
    path.write_bytes(path.read_bytes()[:12])
    np.testing.assert_array_equal(DenseRetriever(catalog, cache_dir=tmp_path).embeddings, expected)
    assert len(calls) == 2


def test_interrupted_cache_write_can_be_retried(tmp_path, monkeypatch):
    catalog = pd.DataFrame({"text": ["first"]})
    monkeypatch.setattr(DenseRetriever, "_encode", lambda self, texts: np.ones((1, 2)))

    def interrupt(file, *args, **kwargs):
        file.write(b"partial")
        raise OSError("write interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(np, "save", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            DenseRetriever(catalog, cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.npy*"))
    assert DenseRetriever(catalog, cache_dir=tmp_path).embeddings.shape == (1, 2)
