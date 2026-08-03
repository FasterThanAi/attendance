"""Phase 2 deliverable 7: "chunk idempotency" and "resume after simulated
failure" (the server-side half of resumability -- get_upload_status
correctly reporting partial progress is what the frontend's resume logic in
apps/web/lib/uploadClient.ts relies on; that TypeScript logic itself isn't
covered by this Python suite).
"""

import pytest

from app.services import upload as upload_service


@pytest.fixture(autouse=True)
def _use_tmp_job_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_service.settings, "job_data_dir", str(tmp_path))
    yield


def test_create_upload_session_computes_chunk_count():
    result = upload_service.create_upload_session(
        class_session_id=1, filename="session.mp4", total_size_bytes=12_000_000
    )
    assert result["chunk_size_bytes"] == upload_service.CHUNK_SIZE_BYTES
    # ceil(12_000_000 / 5_242_880) == 3
    assert result["total_chunks"] == 3


def test_writing_the_same_chunk_twice_is_idempotent():
    created = upload_service.create_upload_session(1, "session.mp4", total_size_bytes=1000)
    upload_id = created["upload_id"]

    upload_service.write_chunk(upload_id, 0, b"hello")
    status_1 = upload_service.get_upload_status(upload_id)

    upload_service.write_chunk(upload_id, 0, b"hello")  # same chunk again, e.g. a client retry
    status_2 = upload_service.get_upload_status(upload_id)

    assert status_1 == status_2
    assert status_1["received_chunks"] == [0]


def test_rejects_out_of_range_chunk_index():
    created = upload_service.create_upload_session(1, "session.mp4", total_size_bytes=1000)
    upload_id = created["upload_id"]  # total_chunks == 1 for a 1000-byte file

    with pytest.raises(upload_service.UploadValidationError) as exc_info:
        upload_service.write_chunk(upload_id, 5, b"data")
    assert exc_info.value.code == "invalid_chunk_index"


def test_status_reports_partial_progress_for_resume(tmp_path):
    """This is the "resume after simulated failure" scenario from the
    frontend's point of view: a client uploads chunks 0 and 1 out of 3, the
    connection drops (simulated by just... not uploading chunk 2 yet), and
    a later call to GET /uploads/{id} (here, get_upload_status directly)
    must report exactly what's missing so the client knows to send only
    chunk 2, not start over.
    """
    created = upload_service.create_upload_session(1, "session.mp4", total_size_bytes=15_000_000)  # 3 chunks
    upload_id = created["upload_id"]

    upload_service.write_chunk(upload_id, 0, b"a" * 100)
    upload_service.write_chunk(upload_id, 1, b"b" * 100)
    # chunk 2 "fails to arrive" -- simulated network failure, nothing written

    status = upload_service.get_upload_status(upload_id)
    assert status["received_chunks"] == [0, 1]
    assert status["is_complete"] is False

    # "Resume": the client sends only the missing chunk.
    upload_service.write_chunk(upload_id, 2, b"c" * 100)
    status_after_resume = upload_service.get_upload_status(upload_id)
    assert status_after_resume["received_chunks"] == [0, 1, 2]
    assert status_after_resume["is_complete"] is True


def test_assemble_rejects_when_chunks_missing():
    created = upload_service.create_upload_session(1, "session.mp4", total_size_bytes=15_000_000)
    upload_id = created["upload_id"]
    upload_service.write_chunk(upload_id, 0, b"a" * 100)
    # chunks 1 and 2 never arrive

    with pytest.raises(upload_service.UploadValidationError) as exc_info:
        upload_service.assemble_and_validate(upload_id)
    assert exc_info.value.code == "chunks_missing"


def test_get_status_for_unknown_upload_raises():
    with pytest.raises(upload_service.UploadValidationError) as exc_info:
        upload_service.get_upload_status("does-not-exist")
    assert exc_info.value.code == "upload_not_found"
