import os
import unittest
from unittest.mock import patch

from app.integrations.lgi_readonly import (
    LgiMatch,
    LgiPlaybackUnavailable,
    _safe_vimeo_embed_url,
    build_source_uri,
    parse_source_uri,
)
from app.schemas import JobCreate


MATCH_ID = "9ff2c0eb-17ab-4ed0-a0b8-b13613f1f6aa"


def vimeo_match(embed_url: str | None) -> LgiMatch:
    return LgiMatch(
        id=MATCH_ID,
        title="Fiorentina-Inter",
        slug="fiorentina-inter",
        home_team="Fiorentina",
        away_team="Inter",
        competition="Campionato U18",
        season="2025/26",
        duration_seconds=7200,
        provider="vimeo",
        mux_playback_id=None,
        vimeo_video_id="1121079583",
        vimeo_player_embed_url=embed_url,
    )


class LgiReadOnlySourceTests(unittest.TestCase):
    def test_source_uri_round_trip_is_uuid_scoped(self):
        source = build_source_uri(MATCH_ID)
        self.assertEqual(source, f"lgi://match/{MATCH_ID}")
        self.assertEqual(parse_source_uri(source), MATCH_ID)
        self.assertIsNone(parse_source_uri("lgi://match/not-a-uuid"))
        self.assertIsNone(parse_source_uri(f"https://example.com/{MATCH_ID}"))

    def test_job_create_requires_exactly_one_source(self):
        valid = JobCreate(
            lgi_match_id=MATCH_ID,
            role="Midfielder",
            category="U18",
        )
        self.assertEqual(valid.lgi_match_id, MATCH_ID)

        with self.assertRaises(ValueError):
            JobCreate(role="Midfielder", category="U18")
        with self.assertRaises(ValueError):
            JobCreate(
                lgi_match_id=MATCH_ID,
                video_url="https://example.com/match.mp4",
                role="Midfielder",
                category="U18",
            )

    def test_vimeo_embed_url_is_host_and_video_id_allowlisted(self):
        expected = "https://player.vimeo.com/video/1121079583?h=fc3000c816"
        self.assertEqual(_safe_vimeo_embed_url(vimeo_match(expected)), expected)
        self.assertEqual(
            _safe_vimeo_embed_url(
                vimeo_match("https://attacker.example/video/1121079583")
            ),
            "https://player.vimeo.com/video/1121079583",
        )
        invalid = vimeo_match(expected)
        object.__setattr__(invalid, "vimeo_video_id", "../../etc/passwd")
        with self.assertRaises(LgiPlaybackUnavailable):
            _safe_vimeo_embed_url(invalid)

    @patch.dict(os.environ, {}, clear=True)
    def test_database_url_is_required_and_never_falls_back_to_primary(self):
        from app.integrations.lgi_readonly import _database_url

        with self.assertRaisesRegex(
            RuntimeError, "LGI_READONLY_DATABASE_URL is not configured"
        ):
            _database_url()


if __name__ == "__main__":
    unittest.main()
