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

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.params import PipelineParams

logger = logging.getLogger("attend.worker.embed")

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


# --------------------------------------------------------------------------
# Phase 5: batch embedding for classroom video (embeds every crop
# align_crops wrote in Phase 4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingSet:
    embeddings_npy_path: Path
    count: int
    dim: int


def embed_aligned(
    aligned_memmap: np.ndarray, out_dir: Path, params: PipelineParams, model_dir: Path
) -> EmbeddingSet:
    """Phase 5 deliverable 1: embeds every crop in `aligned_memmap` (the
    (N, 112, 112, 3) uint8 BGR array align_crops wrote in Phase 4),
    `params.embed_batch_size` crops at a time through the singleton ArcFace
    model, and writes the result to `out_dir/embeddings.npy` as a single
    (N, 512) float32 L2-normalised memmap -- same self-describing
    np.lib.format.open_memmap convention align_crops used for aligned.npy,
    so later phases can `np.load(path, mmap_mode="r")` without needing
    N/dtype passed out of band.

    `model_dir` is not in the roadmap's literal `embed_aligned(aligned_memmap,
    out_dir, params)` signature -- added for the same reason detect.py's
    `detect_all_frames` needed an explicit model_dir: load_model() has to be
    told where the ONNX weights live, and there's no other way to plumb that
    through without a global.

    ASSERTS BGR uint8 (112, 112, 3): insightface's ArcFace expects BGR, not
    RGB -- getting this backwards is the single most common silent bug here
    (the model still runs, it just embeds a colour-channel-swapped face, and
    every similarity score becomes quietly wrong instead of erroring). Since
    `aligned_memmap` is exactly what align_crops (Phase 4) wrote, and
    align_face/cv2 never swap channels, this should always hold -- the check
    below is a cheap, fail-loud tripwire in case anything upstream ever hands
    this the wrong array.
    """
    n = aligned_memmap.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_npy_path = out_dir / "embeddings.npy"

    if n == 0:
        # An all-rejected video (Phase 4 can produce a 0-row aligned.npy) --
        # write a valid, empty, still-self-describing embeddings.npy rather
        # than crashing or skipping the file entirely.
        np.save(embeddings_npy_path, np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
        logger.warning("embed stage: 0 aligned crops -- nothing to embed")
        return EmbeddingSet(embeddings_npy_path=embeddings_npy_path, count=0, dim=EMBEDDING_DIM)

    if aligned_memmap.ndim != 4 or tuple(aligned_memmap.shape[1:]) != (112, 112, 3) or aligned_memmap.dtype != np.uint8:
        raise ValueError(
            f"embed_aligned expects (N, 112, 112, 3) uint8 BGR, got shape="
            f"{aligned_memmap.shape} dtype={aligned_memmap.dtype}"
        )

    model = load_model(model_dir)
    batch_size = params.embed_batch_size

    out_memmap = np.lib.format.open_memmap(
        embeddings_npy_path, mode="w+", dtype=np.float32, shape=(n, EMBEDDING_DIM)
    )

    start_time = time.monotonic()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.ascontiguousarray(aligned_memmap[start:end])  # copy out of the memmap slice
        out_memmap[start:end] = embed_batch(model, batch)
    elapsed = time.monotonic() - start_time

    out_memmap.flush()
    del out_memmap  # release the file handle before returning

    throughput = n / elapsed if elapsed > 0 else float("inf")
    logger.info("embed stage: %d crops in %.2fs (%.1f crops/sec)", n, elapsed, throughput)

    return EmbeddingSet(embeddings_npy_path=embeddings_npy_path, count=n, dim=EMBEDDING_DIM)


def run_embed_stage(aligned_npy_path: Path, out_dir: Path, params: PipelineParams, model_dir: Path) -> EmbeddingSet:
    """The I/O wrapper run.py's orchestrator calls: loads aligned.npy as a
    memmap (never the whole array copied into RAM at once -- embed_aligned
    only pulls out one batch at a time) and hands it to embed_aligned.
    """
    aligned_memmap = np.load(aligned_npy_path, mmap_mode="r")
    return embed_aligned(aligned_memmap, out_dir, params, model_dir)
