"""Stub for Phase 0.

The real content of this file -- a single frozen dataclass `PipelineParams`
holding every tunable number in the project (non-negotiable rule #3) -- is a
Phase 1 deliverable, because the first tunables (detector score, blur
threshold, cluster eps, match threshold, etc.) don't have meaning until the
stages that use them exist. Introducing them now would just be guessing at
numbers with nothing to validate them against.
"""
