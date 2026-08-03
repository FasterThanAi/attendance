#!/usr/bin/env python3
"""Phase 7 deliverable 1: ground-truth labelling tool.

Shows you, one at a time, each student enrolled in a session's course while
you scrub through the session's video, and lets you record whether they were
actually present -- writing eval/datasets/{class_session_id}/truth.csv in
the exact format pipeline.match/eval_lib expect. Labelling ~90 students x 8
sessions by hand is this phase's own stated bottleneck ("make the tool
fast"), so this is entirely keyboard-driven, no mouse, and every keystroke
that records a decision auto-saves immediately -- a labelling session that
gets interrupted after student 60 of 90 never loses students 1-60.

CONVENTION (see eval_lib.py's module docstring): the output directory name
IS the class_session_id, as a string -- eval/datasets/{class_session_id}/truth.csv.

Controls (shown on-screen too):
  SPACE       play / pause the video
  j / k       step back / forward 1 frame (while paused)
  J / K       jump back / forward 5 seconds
  y           mark current student PRESENT, save, advance to next student
  n           mark current student ABSENT, save, advance to next student
  1 / 2 / 3   set seat_position to left / centre / right
  g           toggle wears_glasses for the current student
  [ / ]       decrement / increment row_number for the current student
  e           edit free-text notes for the current student (terminal prompt)
  b           go back to the previous student (to fix a mistake)
  p           jump to a specific student by roll_number (terminal prompt)
  s           save truth.csv now (also happens automatically on y/n)
  q           save and quit

Usage:
    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \\
        python eval/scripts/label_session.py --class-session-id 12 \\
        --video-path /data/videos/session12.mp4 --datasets-dir eval/datasets

`--video-path` is optional -- if omitted, the tool looks up the session's
most recent video_upload.storage_uri from the DB. Pass it explicitly if
you're labelling from a copy of the video on your own machine instead of
the path the server recorded (a very likely case: the server's storage_uri
is a path inside a Docker volume, not necessarily reachable from wherever
you're running this tool).

Standalone on purpose, same as gallery_sanity.py/match_report.py: plain
psycopg2 + opencv, no api/worker package dependency.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

TRUTH_FIELDNAMES = ["roll_number", "actually_present", "row_number", "seat_position", "wears_glasses", "notes"]
SEAT_POSITIONS = {ord("1"): "left", ord("2"): "centre", ord("3"): "right"}
SEEK_STEP_SECONDS = 5.0


@dataclass
class StudentRecord:
    student_id: int
    roll_number: str
    full_name: str
    actually_present: bool | None = None
    row_number: int = 1
    seat_position: str = "centre"
    wears_glasses: bool = False
    notes: str = ""


@dataclass
class LabellingState:
    records: list[StudentRecord]
    current_index: int = 0


def fetch_roster(database_url: str, class_session_id: int) -> list[StudentRecord]:
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT course_id FROM class_session WHERE id = %s", (class_session_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"No class_session with id={class_session_id}")
            course_id = row[0]

            cur.execute(
                "SELECT s.id, s.roll_number, s.full_name "
                "FROM student s JOIN enrollment e ON e.student_id = s.id "
                "WHERE e.course_id = %s ORDER BY s.roll_number",
                (course_id,),
            )
            student_rows = cur.fetchall()
    finally:
        conn.close()

    if not student_rows:
        raise ValueError(f"No students enrolled in the course for class_session_id={class_session_id}")

    return [StudentRecord(student_id=r[0], roll_number=r[1], full_name=r[2]) for r in student_rows]


def fetch_video_path(database_url: str, class_session_id: int) -> str:
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT storage_uri FROM video_upload WHERE class_session_id = %s "
                "ORDER BY uploaded_at DESC LIMIT 1",
                (class_session_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(
            f"No video_upload found for class_session_id={class_session_id}. "
            "Pass --video-path explicitly if the video lives somewhere this DB row doesn't point to."
        )
    return row[0]


def load_existing_truth(truth_path: Path, records: list[StudentRecord]) -> None:
    """Resumes a prior labelling pass -- if truth.csv already has an entry
    for a roll_number, load it back into that student's record instead of
    starting blank, so re-opening the tool after an interruption picks up
    exactly where you left off.
    """
    if not truth_path.exists():
        return
    by_roll = {r.roll_number: r for r in records}
    with open(truth_path, newline="") as f:
        for row in csv.DictReader(f):
            rec = by_roll.get(row["roll_number"])
            if rec is None:
                continue
            rec.actually_present = row["actually_present"].strip().lower() in ("true", "1", "yes")
            rec.row_number = int(row["row_number"])
            rec.seat_position = row["seat_position"]
            rec.wears_glasses = row["wears_glasses"].strip().lower() in ("true", "1", "yes")
            rec.notes = row["notes"]


def save_truth_csv(truth_path: Path, records: list[StudentRecord]) -> None:
    """Only writes rows for students who've actually been labelled
    (actually_present is not None) -- a half-finished session's truth.csv
    should reflect exactly what's been decided so far, not a false "absent"
    default for the other 40 students you haven't gotten to yet.
    """
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = truth_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRUTH_FIELDNAMES)
        writer.writeheader()
        for r in records:
            if r.actually_present is None:
                continue
            writer.writerow({
                "roll_number": r.roll_number,
                "actually_present": r.actually_present,
                "row_number": r.row_number,
                "seat_position": r.seat_position,
                "wears_glasses": r.wears_glasses,
                "notes": r.notes,
            })
    tmp_path.replace(truth_path)  # atomic on POSIX -- never leaves a half-written truth.csv


def _draw_overlay(frame, state: LabellingState, video_paused: bool):
    rec = state.records[state.current_index]
    done_count = sum(1 for r in state.records if r.actually_present is not None)
    total = len(state.records)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    def put(text, y, scale=0.7, color=(255, 255, 255)):
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    present_str = (
        "PRESENT" if rec.actually_present is True else "ABSENT" if rec.actually_present is False else "(not marked)"
    )
    present_color = (0, 200, 0) if rec.actually_present is True else (0, 0, 220) if rec.actually_present is False else (200, 200, 200)

    put(f"[{done_count}/{total}] {rec.roll_number}  {rec.full_name}", 30, 0.8)
    put(f"present: {present_str}", 58, 0.65, present_color)
    put(
        f"row={rec.row_number}  seat={rec.seat_position}  glasses={rec.wears_glasses}  "
        f"notes={rec.notes[:30]!r}",
        84, 0.55,
    )
    put(
        "y=present  n=absent  b=back  p=jump  1/2/3=seat  g=glasses  [ ]=row  e=notes  "
        "SPACE=play/pause  j/k=step  J/K=+/-5s  s=save  q=quit",
        112, 0.45, (180, 220, 255),
    )
    put("PAUSED" if video_paused else "playing", 136, 0.5, (0, 165, 255) if video_paused else (0, 255, 0))
    return frame


def run(database_url: str, class_session_id: int, video_path: str | None, datasets_dir: Path) -> int:
    records = fetch_roster(database_url, class_session_id)
    if video_path is None:
        video_path = fetch_video_path(database_url, class_session_id)

    truth_path = datasets_dir / str(class_session_id) / "truth.csv"
    load_existing_truth(truth_path, records)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video at {video_path!r}.")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    state = LabellingState(records=records)
    window_name = f"label_session -- class_session_id={class_session_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = True
    ok, frame = cap.read()
    if not ok:
        print("Could not read the first frame of the video.")
        return 1

    print(f"Labelling {len(records)} student(s). Writing to {truth_path}.")
    print("Focus the video window and use the keyboard -- see the on-screen legend.")

    while True:
        if not paused:
            ok, next_frame = cap.read()
            if ok:
                frame = next_frame
            else:
                paused = True  # end of video -- stay on the last frame, keep labelling

        display = _draw_overlay(frame, state, paused)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        rec = state.records[state.current_index]

        if key == 255:  # no key pressed this tick
            continue
        elif key == ord(" "):
            paused = not paused
        elif key == ord("j"):
            paused = True
            pos = max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
        elif key == ord("k"):
            paused = True
            ok, next_frame = cap.read()
            if ok:
                frame = next_frame
        elif key == ord("J"):
            paused = True
            new_pos_ms = max(0.0, cap.get(cv2.CAP_PROP_POS_MSEC) - SEEK_STEP_SECONDS * 1000)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_pos_ms)
            ok, frame = cap.read()
        elif key == ord("K"):
            paused = True
            new_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC) + SEEK_STEP_SECONDS * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, new_pos_ms)
            ok, frame = cap.read()
        elif key in (ord("y"), ord("n")):
            rec.actually_present = key == ord("y")
            save_truth_csv(truth_path, state.records)
            if state.current_index < len(state.records) - 1:
                state.current_index += 1
        elif key == ord("b"):
            state.current_index = max(0, state.current_index - 1)
        elif key == ord("p"):
            target = input("\nJump to roll_number: ").strip()
            matches = [i for i, r in enumerate(state.records) if r.roll_number == target]
            if matches:
                state.current_index = matches[0]
            else:
                print(f"No student with roll_number={target!r}.")
        elif key in SEAT_POSITIONS:
            rec.seat_position = SEAT_POSITIONS[key]
        elif key == ord("g"):
            rec.wears_glasses = not rec.wears_glasses
        elif key == ord("["):
            rec.row_number = max(1, rec.row_number - 1)
        elif key == ord("]"):
            rec.row_number += 1
        elif key == ord("e"):
            rec.notes = input(f"\nNotes for {rec.roll_number} ({rec.full_name}): ").strip()
        elif key == ord("s"):
            save_truth_csv(truth_path, state.records)
            print(f"Saved to {truth_path}.")
        elif key == ord("q"):
            save_truth_csv(truth_path, state.records)
            break

    cap.release()
    cv2.destroyAllWindows()

    done_count = sum(1 for r in state.records if r.actually_present is not None)
    print(f"\nSaved {done_count}/{len(state.records)} labelled student(s) to {truth_path}.")
    if done_count < len(state.records):
        remaining = [r.roll_number for r in state.records if r.actually_present is None]
        print(f"{len(remaining)} student(s) not yet labelled: {', '.join(remaining[:10])}"
              f"{' ...' if len(remaining) > 10 else ''}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-session-id", required=True, type=int)
    parser.add_argument("--video-path", default=None, help="Overrides the DB's video_upload.storage_uri")
    parser.add_argument("--datasets-dir", type=Path, default=Path("eval/datasets"))
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("Set DATABASE_URL, e.g.:\n"
              "  DATABASE_URL=postgresql://attend:attend@localhost:5432/attend "
              "python eval/scripts/label_session.py --class-session-id 12")
        return 1
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    return run(database_url, args.class_session_id, args.video_path, args.datasets_dir)


if __name__ == "__main__":
    sys.exit(main())
