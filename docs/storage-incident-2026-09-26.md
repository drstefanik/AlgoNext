# Storage incident — 26 September 2026

## Confirmed cause

The production API returned HTTP 500 for the reported analysis and HTTP 503
from `/ready`. Redis reported `MISCONF` after failed RDB persistence. VPS logs
confirmed `No space left on device` for both Redis and PostgreSQL.

At 07:00 UTC the 150 GiB filesystem had 1.7 GiB available (99% used).
The worker had restarted at 06:57:59 UTC. The interrupted analysis still showed
108 of 120 tracking windows, last updated at 06:53:57 UTC. A running status in
the database was therefore insufficient evidence that processing was continuing.

The storage diagnostic also measured approximately 99 GiB in the AlgoNext
object-storage volume and 13 GiB of worker temporary workspaces. These files
were preserved. No source videos, result assets, database volumes, queue data,
or saved player selections were deleted.

## Recovery performed

- Executed the bounded storage workflow on `ops/storage-incident-20260926`.
- Reclaimed 12.54 GB of unused Docker build cache older than one hour, retaining
  a 2 GB cache floor. Docker reported approximately 14 GiB available afterward.
- A project-filtered dangling-image cleanup reclaimed zero additional bytes.
- Confirmed PostgreSQL accepts `SELECT 1`, Redis reports successful persistence,
  and the API, Redis, database, and worker are ready on revision
  `057735655f76a6429cceddd6bd23903dc5ecded0`.
- Confirmed the worker had no active, reserved, or scheduled tasks before job
  recovery. The saved input video was readable using a ranged GET (HTTP 206,
  total object length 2,386,411,460 bytes).
- Recovered the existing job through the retry endpoint after 912 seconds of
  inactivity, supplying the interrupted attempt ID in both the header and
  request body. The new attempt is `38f55b60-871a-4bfe-8a56-f4e3c273989d`.
  Verified a fresh worker claim (`RUNNING`, `UPLOADING_INPUT`, 27%) and exact
  preservation of the selected player references. No duplicate job was created.
- At 07:11:16 UTC the production frontend proxy returned HTTP 200 with fresh
  tracking progress: 4 of 120 windows, 36%, on the new attempt. The same proxy
  returned HTTP 200 from `/ready`, with database, Redis, and worker all ready.

## Operational limits

This is a capacity recovery, not a validation of player identity or match
ratings. Existing scoring and identity gates remain unchanged. The object
store continues to occupy most of the disk: unused build-cache reclamation is
not a substitute for a reviewed retention policy or additional storage.

No application deployment or service restart was performed during this
recovery. The analysis restarts computation; the previous 108/120 count is not
a resumable checkpoint. The final analysis outcome remains to be verified.

Evidence: GitHub Actions runs `36225421665`, `36225549887`, `36225650841`, and
`36225761430` in `drstefanik/AlgoNext`.
