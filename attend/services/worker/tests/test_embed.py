"""Phase 1 deliverable 7: "embed_batch output shape and normalisation."

Uses a fake model object rather than the real ArcFace ONNX model, so this
suite runs without ~250MB of downloaded weights or onnxruntime actually
doing inference -- it tests embed.py's OWN contract (shape validation,
L2-normalisation, BGR-shaped input enforcement), not insightface's model
correctness, which can't be unit-tested anyway.
"""

import numpy as np
import pytest

from pipeline.embed import EMBEDDING_DIM, embed_batch, l2_normalize


class _FakeRawModel:
    """Stands in for insightface's recognition model. `get_feat` returns
    deterministic, NOT-pre-normalised vectors, specifically so the test can
    tell the difference between "embed_batch normalises its output" and
    "the fake model happened to already return normalised vectors."
    """

    def get_feat(self, aligned: np.ndarray) -> np.ndarray:
        n = aligned.shape[0]
        rng = np.random.default_rng(seed=42)
        return rng.normal(loc=0.0, scale=10.0, size=(n, EMBEDDING_DIM)).astype(np.float32)


class _FakeEmbedModel:
    def __init__(self) -> None:
        self.raw = _FakeRawModel()


def test_l2_normalize_unit_norm():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    normalised = l2_normalize(vectors)
    norms = np.linalg.norm(normalised, axis=1)
    # The zero vector can't be made unit-norm; l2_normalize must not raise or
    # produce NaN for it (see the epsilon guard in l2_normalize).
    assert norms[0] == pytest.approx(1.0, abs=1e-5)
    assert norms[1] == pytest.approx(1.0, abs=1e-5)
    assert not np.isnan(normalised[2]).any()


def test_embed_batch_shape_and_normalisation():
    model = _FakeEmbedModel()
    aligned = np.zeros((5, 112, 112, 3), dtype=np.uint8)

    embeddings = embed_batch(model, aligned)

    assert embeddings.shape == (5, EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_batch_rejects_wrong_shape():
    model = _FakeEmbedModel()
    wrong_shape = np.zeros((5, 100, 100, 3), dtype=np.uint8)  # not 112x112

    with pytest.raises(ValueError, match="112, 112, 3"):
        embed_batch(model, wrong_shape)
