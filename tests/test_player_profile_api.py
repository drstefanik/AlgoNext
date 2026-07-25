import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone

from starlette.requests import Request

from app.player_profile_api import PlayerProfilePayload, save_player_profile


@dataclass
class DummyJob:
    id: str
    target: dict = field(default_factory=dict)
    player_ref: dict | None = field(default_factory=dict)
    updated_at: datetime | None = None


class DummySession:
    def __init__(self, job: DummyJob):
        self.job = job
        self.committed = False
        self.refreshed = False

    def get(self, _model, job_id: str):
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
                "player": {"team_name": "Da associare"},
                "selections": [],
            },
            player_ref={"track_id": 9, "selection_source": "preview_frame_pick"},
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        self.session = DummySession(self.job)
        self.request = Request({"type": "http", "headers": []})

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


if __name__ == "__main__":
    unittest.main()
