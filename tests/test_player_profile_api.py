import copy
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import HTTPException
from starlette.requests import Request

from app.player_profile_api import PlayerProfilePayload, save_player_profile
from app.schemas import JobCreate


@dataclass
class DummyJob:
    id: str
    status: str = "READY_TO_ENQUEUE"
    target: dict = field(default_factory=dict)
    player_ref: dict | None = field(default_factory=dict)
    updated_at: datetime | None = None


class DummySession:
    def __init__(self, job: DummyJob):
        self.job = job
        self.committed = False
        self.refreshed = False

    def get(self, _model, job_id: str, **_kwargs):
        return self.job if job_id == self.job.id else None

    def commit(self):
        self.committed = True

    def refresh(self, _job):
        self.refreshed = True


class PlayerProfileApiTests(unittest.TestCase):
    def setUp(self):
        self.job = DummyJob(
            id="job-profile-1",
            target={
                "player": {"team_name": None},
                "selections": [],
                "analysis_attempt_id": "selection-revision",
            },
            player_ref={"track_id": 9, "selection_source": "preview_frame_pick"},
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        self.session = DummySession(self.job)
        self.request = self.request_for_attempt("selection-revision")

    def request_for_attempt(self, analysis_attempt_id: str | None = None):
        headers = (
            [(b"x-analysis-attempt-id", analysis_attempt_id.encode())]
            if analysis_attempt_id
            else []
        )
        return Request({"type": "http", "headers": headers})

    def test_job_creation_does_not_require_player_profile(self):
        payload = JobCreate(
            video_key="uploads/match.mp4",
            role="Midfielder",
            category="U17",
            full_match_mode=True,
        )

        self.assertIsNone(payload.team_name)
        self.assertIsNone(payload.player_name)
        self.assertIsNone(payload.shirt_number)

    def test_profile_is_attached_to_the_visually_selected_player(self):
        payload = PlayerProfilePayload(
            playerName="  Mario Rossi  ",
            teamName="  AS Roma  ",
            shirtNumber=8,
        )

        response = save_player_profile(
            self.job.id,
            payload,
            self.request,
            self.session,
        )

        self.assertTrue(response["ok"])
        self.assertTrue(self.session.committed)
        self.assertTrue(self.session.refreshed)
        self.assertEqual(
            self.job.target["player"],
            {
                "team_name": "AS Roma",
                "player_name": "Mario Rossi",
                "shirt_number": 8,
            },
        )
        self.assertEqual(self.job.player_ref["profile"], self.job.target["player"])
        self.assertEqual(
            self.job.target["analysis_attempt_id"],
            "selection-revision",
        )

    def test_profile_is_rejected_before_visual_selection(self):
        self.job.player_ref = {}
        payload = PlayerProfilePayload(teamName="AS Roma", shirtNumber=8)

        with self.assertRaises(HTTPException) as raised:
            save_player_profile(
                self.job.id,
                payload,
                self.request,
                self.session,
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(
            raised.exception.detail["code"],
            "PLAYER_SELECTION_REQUIRED",
        )
        self.assertFalse(self.session.committed)

    def test_partial_update_preserves_existing_player_data(self):
        self.job.target["player"] = {
            "player_name": "Mario Rossi",
            "team_name": "AS Roma",
            "shirt_number": 8,
        }
        payload = PlayerProfilePayload(shirtNumber=10)

        save_player_profile(
            self.job.id,
            payload,
            self.request,
            self.session,
        )

        self.assertEqual(self.job.target["player"]["player_name"], "Mario Rossi")
        self.assertEqual(self.job.target["player"]["team_name"], "AS Roma")
        self.assertEqual(self.job.target["player"]["shirt_number"], 10)

    def test_blank_optional_text_is_stored_as_none(self):
        payload = PlayerProfilePayload(playerName="   ", teamName="")

        save_player_profile(
            self.job.id,
            payload,
            self.request,
            self.session,
        )

        self.assertIsNone(self.job.target["player"]["player_name"])
        self.assertIsNone(self.job.target["player"]["team_name"])

    def test_active_analysis_rejects_profile_without_mutation(self):
        payload = PlayerProfilePayload(teamName="AS Roma", shirtNumber=8)

        for status in ("QUEUED", "RUNNING", "PROCESSING"):
            with self.subTest(status=status):
                self.job.status = status
                self.job.target["analysis_attempt_id"] = "attempt-a"
                self.request = self.request_for_attempt("attempt-a")
                self.session.committed = False
                self.session.refreshed = False
                before = copy.deepcopy(self.job.__dict__)

                with self.assertRaises(HTTPException) as raised:
                    save_player_profile(
                        self.job.id,
                        payload,
                        self.request,
                        self.session,
                    )

                self.assertEqual(raised.exception.status_code, 409)
                self.assertEqual(
                    raised.exception.detail,
                    {
                        "code": "ANALYSIS_IN_PROGRESS",
                        "message": (
                            "Player details cannot change during an active analysis."
                        ),
                        "details": {
                            "status": status,
                            "analysis_attempt_id": "attempt-a",
                        },
                    },
                )
                self.assertEqual(self.job.__dict__, before)
                self.assertFalse(self.session.committed)
                self.assertFalse(self.session.refreshed)

    def test_stale_attempt_cannot_overwrite_terminal_profile(self):
        self.job.status = "COMPLETED"
        self.job.target["analysis_attempt_id"] = "attempt-b"
        before = copy.deepcopy(self.job.__dict__)

        with self.assertRaises(HTTPException) as raised:
            save_player_profile(
                self.job.id,
                PlayerProfilePayload(teamName="Stale Team"),
                self.request_for_attempt("attempt-a"),
                self.session,
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(
            raised.exception.detail["code"],
            "ANALYSIS_ATTEMPT_MISMATCH",
        )
        self.assertEqual(self.job.__dict__, before)
        self.assertFalse(self.session.committed)
        self.assertFalse(self.session.refreshed)

    def test_stale_session_refreshes_under_lock_before_profile_write(self):
        class ScalarResult:
            def __init__(self, job):
                self.job = job

            def scalar_one_or_none(self):
                return self.job

        class StaleSnapshotSession(DummySession):
            def __init__(self, cached_job, authoritative_session):
                super().__init__(cached_job)
                self.authoritative_session = authoritative_session
                self.lock_options = []
                self.locked = []

            def execute(self, statement):
                options = dict(statement.get_execution_options())
                self.lock_options.append(options)
                self.locked.append(
                    getattr(statement, "_for_update_arg", None) is not None
                )
                if options.get("populate_existing"):
                    authoritative_job = self.authoritative_session.job
                    self.job.status = authoritative_job.status
                    self.job.target = copy.deepcopy(authoritative_job.target)
                    self.job.player_ref = copy.deepcopy(authoritative_job.player_ref)
                return ScalarResult(self.job)

        cached = copy.deepcopy(self.job)
        cached.status = "PARTIAL"
        cached.target["analysis_attempt_id"] = "attempt-a"
        authoritative = copy.deepcopy(cached)
        authoritative.status = "RUNNING"
        authoritative.target["analysis_attempt_id"] = "attempt-b"
        authoritative_session = DummySession(authoritative)
        stale_session = StaleSnapshotSession(cached, authoritative_session)
        before_player = copy.deepcopy(authoritative.target["player"])

        with self.assertRaises(HTTPException) as raised:
            save_player_profile(
                cached.id,
                PlayerProfilePayload(teamName="Stale Team"),
                self.request_for_attempt("attempt-a"),
                stale_session,
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(
            raised.exception.detail["details"]["current_analysis_attempt_id"],
            "attempt-b",
        )
        self.assertTrue(stale_session.lock_options)
        self.assertTrue(
            all(
                options.get("populate_existing")
                for options in stale_session.lock_options
            )
        )
        self.assertTrue(all(stale_session.locked))
        self.assertEqual(cached.target["player"], before_player)
        self.assertFalse(stale_session.committed)


if __name__ == "__main__":
    unittest.main()
