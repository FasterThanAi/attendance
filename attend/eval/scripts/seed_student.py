#!/usr/bin/env python3
"""Helper script to seed a test student, teacher, course, class session, and grant consent.

Usage:
    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend python3 attend/eval/scripts/seed_student.py
"""

from datetime import datetime, timezone
import os
import sys
import psycopg2

def main():
    db_url = os.environ.get("DATABASE_URL", "postgresql://attend:attend@localhost:5432/attend")
    conn = psycopg2.connect(db_url)
    now = datetime.now(timezone.utc)
    try:
        with conn.cursor() as cur:
            # 1. Institution
            cur.execute(
                "INSERT INTO institution (id, name, created_at) VALUES (1, 'Test Institute', %s) "
                "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                (now,)
            )
            inst_id = cur.fetchone()[0]

            # 2. Department
            cur.execute(
                "INSERT INTO department (id, institution_id, name, code) VALUES (1, %s, 'Computer Science', 'CS') "
                "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                (inst_id,)
            )
            dept_id = cur.fetchone()[0]

            # 3. Student (id=1)
            cur.execute(
                "INSERT INTO student (id, department_id, roll_number, full_name, admission_year, is_active) "
                "VALUES (1, %s, 'CS001', 'Priyanshu Kumar', 2026, TRUE) "
                "ON CONFLICT (id) DO UPDATE SET full_name = EXCLUDED.full_name "
                "RETURNING id",
                (dept_id,)
            )
            student_id = cur.fetchone()[0]

            # 4. Consent for Student (id=1)
            cur.execute(
                "INSERT INTO consent (student_id, granted_at, consent_version, scope, evidence_uri) "
                "VALUES (%s, %s, 'v1.0', 'classroom_attendance', 'form_signed') "
                "ON CONFLICT DO NOTHING",
                (student_id, now)
            )

            # 5. Teacher (id=1)
            cur.execute(
                "INSERT INTO teacher (id, department_id, full_name, email, password_hash) "
                "VALUES (1, %s, 'Dr. Alan Turing', 'turing@test.edu', 'pbkdf2:dummy') "
                "ON CONFLICT (id) DO UPDATE SET full_name = EXCLUDED.full_name RETURNING id",
                (dept_id,)
            )
            teacher_id = cur.fetchone()[0]

            # 6. Course (id=1)
            cur.execute(
                "INSERT INTO course (id, department_id, code, title, semester) "
                "VALUES (1, %s, 'CS101', 'Computer Science 101', 'Fall 2026') "
                "ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title RETURNING id",
                (dept_id,)
            )
            course_id = cur.fetchone()[0]

            # 7. Class Session (id=1)
            cur.execute(
                "INSERT INTO class_session (id, course_id, teacher_id, scheduled_at, room, status) "
                "VALUES (1, %s, %s, %s, 'Room 101', 'recording') "
                "ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status RETURNING id",
                (course_id, teacher_id, now)
            )
            session_id = cur.fetchone()[0]

            # 8. Enrollment (Student 1 in Course 1)
            cur.execute(
                "INSERT INTO enrollment (student_id, course_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (student_id, course_id)
            )

        conn.commit()
        print(f"Successfully seeded database:\n"
              f"  - Student ID=1 ('Priyanshu Kumar') with active consent\n"
              f"  - Teacher ID=1 ('Dr. Alan Turing')\n"
              f"  - Course ID=1 ('CS101')\n"
              f"  - Class Session ID=1 ('Room 101', status='recording')")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
