# Attend — consent to record and use your face for attendance

**Please read this before recording your enrollment video or appearing in any
classroom attendance video.**

## What this project is

This is a student engineering project called Attend. It automatically takes
attendance in a classroom by recording a short video that pans across the
room, finding faces in that video, and comparing them to a small set of
reference images you provide when you enroll.

## What is recorded

- A short (about 5 second) video of your face, which you record yourself
  while turning your head slowly from left to right. This is used only to
  build your reference set (called your "gallery").
- During each class, a video of the whole classroom is recorded by the
  teacher. You will appear in this video along with your classmates.

## What is derived from that recording

From your enrollment video, the system computes a small number
(5 to 8) of **mathematical face descriptors** — a list of 512 numbers per
photo, produced by a face-recognition model. These numbers are not a photo
and cannot be viewed as an image. However, they are still considered
biometric data because they are derived directly from your face, and are
handled with that in mind.

A small number of your best-quality enrollment photos are also kept (not
just the numbers), so that a human — your teacher, during review — can
visually confirm a match. Beyond this small enrollment set, individual video
frames from classroom sessions are not kept as photographs once a session's
processing is complete; only the crops needed to explain the day's decision
are retained for the retention period below.

## How long it is kept

- Your face descriptors and enrollment photos are kept for **180 days** by
  default (see `BIOMETRIC_RETENTION_DAYS` in the project configuration),
  measured from creation, after which they are automatically and permanently
  deleted.
- Your attendance record itself (present/absent, per class, per date) is an
  ordinary academic record, not biometric data, and is kept for as long as
  your institution normally keeps attendance records. Deleting your
  biometric data does not delete your attendance history.

## Who can see this data

Only the people running this project (the project owner listed below) and
your teacher, for the purpose of taking and reviewing attendance. It is not
shared outside the project, not sold, not used for any purpose other than
attendance, and not used to train any other system.

## How to revoke consent

You can withdraw consent at any time, for any reason, with no need to
explain why. Contact the project owner (contact details below) and ask for
your data to be deleted. Once processed:

- Your enrollment photos and face descriptors will be permanently deleted.
- You will be excluded from future automated attendance and instead marked
  by the teacher's normal manual process.
- Your past attendance records are not affected (they are academic records,
  not biometric data), unless you specifically also request those be
  corrected through your institution's normal process.

## This does not affect your grade or official attendance

During the entire study, your institution's official attendance is whatever
the teacher records through the normal method. This system runs in parallel
for research purposes, and its output does not by itself change anyone's
official attendance or grade.

## If you are under 18

If you are under 18, a parent or legal guardian must also provide verifiable
consent before you can be enrolled. If that consent cannot be obtained, you
will be excluded from the system entirely and marked by the normal manual
process, with no disadvantage to you.

## Contact

Project owner: Priyanshu Kumar
Email: rishucoder64@gmail.com

This is an independent project without a formal academic supervisor. If you
are running this at a college and can loop in a faculty member, department
head, or ethics contact before recording real students, do that — an extra
set of eyes on a project that collects classmates' biometric data is worth
having even when it isn't a formal requirement.
