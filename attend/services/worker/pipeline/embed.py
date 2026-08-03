"""Face embedding: ArcFace r100 (buffalo_l), via onnxruntime.

ASSUMPTION / THING TO VERIFY ON YOUR MAC: insightface's `buffalo_l` model
pack ships a recognition model file that in some insightface releases is
named `w600k_r50.onnx` (a ResNet50-based ArcFace variant) rather than an
"r100" file -- the roadmap PDF says "ArcFace r100 (buffalo_l)" but those two
details (r100 vs buffalo_l's actual recognition backbone) don't necessarily
match in every insightface version, and I have no network access in this
sandbox to check the installed package's actual model manifest. The FIRST
time you run this on your Mac, insightface will auto-download the buffalo_l
pack to ~/.insightface/models/buffalo_l/ -- look at what .onnx files land
there and update RECOGNITION_MODEL_FILENAME below to match if it's not
w600k_r50.onnx. Nothing else in this file depends on which exact file it is;
this is a one-line fix if I guessed wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RECOGNITION_MODEL_FILENAME = "w600k_r50.onnx"
EMBEDDING_DIM = 512


class EmbedModel:
    """Thin wrapper around an insightface model_zoo recognition model, for
    the same reason as detect.py's DetectorModel: callers depend on this
    narrow interface, not insightface's internals.
    """

    def __init__(self, raw_model) -> None:
        self.raw = raw_model


_embed_singleton: EmbedModel | None = None


def load_model(model_dir: Path | None = None, ctx_id: int = -1) -> EmbedModel:
    global _embed_singleton
    if _embed_singleton is not None:
        return _embed_singleton

    from insightface.model_zoo import model_zoo

    if model_dir is None or not (model_dir / RECOGNITION_MODEL_FILENAME).exists():
        default_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
        if (default_dir / RECOGNITION_MODEL_FILENAME).exists():
            model_dir = default_dir
        elif model_dir is not None:
            raise FileNotFoundError(f"{RECOGNITION_MODEL_FILENAME} not found in {model_dir} or {default_dir}")
        else:
            raise FileNotFoundError(f"{RECOGNITION_MODEL_FILENAME} not found in {default_dir}")

    raw_model = model_zoo.get_model(str(model_dir / RECOGNITION_MODEL_FILENAME))
    raw_model.prepare(ctx_id=ctx_id)

    _embed_singleton = EmbedModel(raw_model)
    return _embed_singleton


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Pulled out as its own pure function (no
    model, no I/O) specifically so it can be unit-tested without needing the
    actual ONNX model loaded -- see tests/test_embed.py.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)  # avoid div-by-zero on an all-zero row
    return vectors / norms


def embed_batch(model: EmbedModel, aligned: np.ndarray) -> np.ndarray:
    """`aligned`: (N, 112, 112, 3) BGR uint8 -- ArcFace expects BGR, not RGB;
    getting this backwards is a silent bug (the model still runs, it just
    embeds a colour-channel-swapped face, and every similarity score becomes
    quietly wrong instead of erroring).

    Returns (N, 512) float32, L2-normalised.
    """
    if aligned.ndim != 4 or aligned.shape[1:] != (112, 112, 3):
        raise ValueError(f"expected (N, 112, 112, 3) uint8 BGR, got shape {aligned.shape}")

    try:
        crops = [aligned[i] for i in range(aligned.shape[0])]
        raw_embeddings = model.raw.get_feat(crops)
    except AttributeError:
        raw_embeddings = model.raw.get_feat(aligned)
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"expected {EMBEDDING_DIM}-dim embeddings, model returned shape {embeddings.shape}")

    return l2_normalize(embeddings)
