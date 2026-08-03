"""Phase 5 deliverable 1 tests: embed_aligned's batch behavior, memmap
output, and BGR/shape assertion. Uses the same fake-model pattern as
test_embed.py (Phase 1) -- no real ONNX model needed.
"""

from __future__ import annotations

import numpy as np
import pytest

import pipeline.embed as embed_module
from pipeline.embed import EMBEDDING_DIM, EmbedModel, embed_aligned
from pipeline.params import PipelineParams


class _FakeRawModel:
    """Returns a deterministic, NOT-pre-normalised vector per crop (so the
    test can tell embed_aligned's own L2-normalisation is actually
    happening, same reasoning as test_embed.py's fake model). Also records
    every batch size it was called with, so tests can assert batching
    actually splits the input into embed_batch_size-sized chunks.
    """

    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def get_feat(self, aligned) -> np.ndarray:
        n = len(aligned)
        self.call_sizes.append(n)
        rng = np.random.default_rng(seed=hash(n) % (2**31))
        return rng.normal(loc=0.0, scale=10.0, size=(n, EMBEDDING_DIM)).astype(np.float32)


@pytest.fixture
def fake_model(monkeypatch):
    raw = _FakeRawModel()
    model = EmbedModel(raw)
    monkeypatch.setattr(embed_module, "load_model", lambda model_dir: model)
    return raw


def test_embed_aligned_writes_correctly_shaped_normalised_memmap(tmp_path, fake_model):
    aligned = np.zeros((10, 112, 112, 3), dtype=np.uint8)
    params = PipelineParams(embed_batch_size=64)

    result = embed_aligned(aligned, tmp_path, params, model_dir=tmp_path)

    assert result.count == 10
    assert result.dim == EMBEDDING_DIM
    embeddings = np.load(result.embeddings_npy_path, mmap_mode="r")
    assert embeddings.shape == (10, EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(np.asarray(embeddings), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_aligned_splits_into_batches(tmp_path, fake_model):
    aligned = np.zeros((10, 112, 112, 3), dtype=np.uint8)
    params = PipelineParams(embed_batch_size=4)

    embed_aligned(aligned, tmp_path, params, model_dir=tmp_path)

    # 10 crops at batch_size=4 -> batches of 4, 4, 2.
    assert fake_model.call_sizes == [4, 4, 2]


def test_embed_aligned_empty_input_writes_valid_empty_array(tmp_path, fake_model):
    aligned = np.zeros((0, 112, 112, 3), dtype=np.uint8)
    params = PipelineParams()

    result = embed_aligned(aligned, tmp_path, params, model_dir=tmp_path)

    assert result.count == 0
    embeddings = np.load(result.embeddings_npy_path)
    assert embeddings.shape == (0, EMBEDDING_DIM)


def test_embed_aligned_rejects_wrong_shape(tmp_path, fake_model):
    wrong_shape = np.zeros((5, 100, 100, 3), dtype=np.uint8)
    params = PipelineParams()

    with pytest.raises(ValueError, match="112, 112, 3"):
        embed_aligned(wrong_shape, tmp_path, params, model_dir=tmp_path)


def test_embed_aligned_rejects_non_uint8_dtype(tmp_path, fake_model):
    wrong_dtype = np.zeros((5, 112, 112, 3), dtype=np.float32)
    params = PipelineParams()

    with pytest.raises(ValueError, match="uint8"):
        embed_aligned(wrong_dtype, tmp_path, params, model_dir=tmp_path)
