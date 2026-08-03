#!/usr/bin/env python3
"""Helper script to seed a test session draft, detected clusters, and matches for Phase 8 UI testing.

Usage:
    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend python3 attend/eval/scripts/seed_draft.py
"""

from datetime import datetime, timezone
import json
import os
import psycopg2

def main():
    db_url = os.environ.get("DATABASE_URL", "postgresql://attend:attend@localhost:5432/attend")
    conn = psycopg2.connect(db_url)
    now = datetime.now(timezone.utc)
    try:
        with conn.cursor() as cur:
            # 1. Ensure VideoUpload exists for ClassSession 1
            cur.execute(
                "INSERT INTO video_upload (id, class_session_id, storage_uri, duration_seconds, width, height, fps, bytes, uploaded_at) "
                "VALUES (1, 1, '/data/jobs/uploads/upload_1.mp4', 30.0, 1920, 1080, 30.0, 1000000, %s) "
                "ON CONFLICT (id) DO NOTHING RETURNING id",
                (now,)
            )

            # 2. Ensure ProcessingJob exists with state='succeeded'
            cur.execute(
                "INSERT INTO processing_job (id, class_session_id, video_upload_id, state, params_json, started_at, finished_at) "
                "VALUES (1, 1, 1, 'succeeded', '{}', %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET state = 'succeeded' RETURNING id",
                (now, now)
            )

            # 3. Ensure students exist
            cur.execute("INSERT INTO student (id, department_id, roll_number, full_name, admission_year, is_active) "
                        "VALUES (1, 1, 'CS001', 'Priyanshu Kumar', 2026, TRUE) ON CONFLICT (id) DO NOTHING")
            cur.execute("INSERT INTO student (id, department_id, roll_number, full_name, admission_year, is_active) "
                        "VALUES (2, 1, 'CS002', 'Rahul Sharma', 2026, TRUE) ON CONFLICT (id) DO NOTHING")
            cur.execute("INSERT INTO student (id, department_id, roll_number, full_name, admission_year, is_active) "
                        "VALUES (3, 1, 'CS003', 'Anita Roy', 2026, TRUE) ON CONFLICT (id) DO NOTHING")
            cur.execute("INSERT INTO enrollment (student_id, course_id) VALUES (1, 1), (2, 1), (3, 1) ON CONFLICT DO NOTHING")

            # 4. Insert GalleryPhoto for Priyanshu Kumar if missing
            cur.execute(
                "INSERT INTO gallery_photo (id, student_id, storage_uri, captured_at, quality_score, pose_bucket) "
                "VALUES (101, 1, '/data/jobs/enrollment/1/crops/crop_0.jpg', %s, 0.95, 'frontal') "
                "ON CONFLICT (id) DO NOTHING",
                (now,)
            )

            # 5. Insert DetectedClusters for Job 1
            cur.execute(
                "INSERT INTO detected_cluster (id, processing_job_id, representative_vector, crop_count, mean_quality, best_crop_uri, created_at, retention_expires_at) "
                "VALUES (101, 1, '\\x00', 10, 0.9, '/data/jobs/enrollment/1/crops/crop_0.jpg', %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET best_crop_uri = EXCLUDED.best_crop_uri",
                (now, now)
            )

            cur.execute(
                "INSERT INTO detected_cluster (id, processing_job_id, representative_vector, crop_count, mean_quality, best_crop_uri, created_at, retention_expires_at) "
                "VALUES (102, 1, '\\x00', 8, 0.7, '/data/jobs/enrollment/1/crops/crop_0.jpg', %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET best_crop_uri = EXCLUDED.best_crop_uri",
                (now, now)
            )

            # 6. Insert ClusterMatches for Job 1
            # Put Priyanshu Kumar (Student ID 1) in 'uncertain' (Needs Review) so he appears in the Needs Your Check stepper with his name and photo!
            cur.execute(
                "INSERT INTO cluster_match (id, cluster_id, student_id, decision, similarity, runner_up_similarity) "
                "VALUES (1, 101, 1, 'uncertain', 0.58, 0.42) ON CONFLICT (id) DO UPDATE SET decision = 'uncertain', student_id = 1",
            )

            cur.execute(
                "INSERT INTO cluster_match (id, cluster_id, student_id, decision, similarity, runner_up_similarity) "
                "VALUES (2, 102, 2, 'confident', 0.88, 0.30) ON CONFLICT (id) DO UPDATE SET decision = 'confident', student_id = 2",
            )

            # 7. Draft Summary JSON
            draft_summary = {
                "total_enrolled": 3,
                "proposed_present": 1,
                "needs_review": 1,
                "proposed_absent": 1,
                "unrecognised_clusters": 0,
                "coverage_percent": 66.7,
                "mean_confident_similarity": 0.88,
                "session_health": "good",
            }

            # Update class_session 1 with status='awaiting_review' and draft_summary_json
            cur.execute(
                "UPDATE class_session SET status = 'awaiting_review', draft_summary_json = %s WHERE id = 1",
                (json.dumps(draft_summary),)
            )

        conn.commit()
        print("Successfully updated seed draft: Priyanshu Kumar is now in 'Needs your check' stepper!")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
