from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

LGI_SOURCE_SCHEME = "lgi"
LGI_SOURCE_HOST = "match"
LGI_DEFAULT_REFERER = "https://www.lgichannel.net/"
LGI_PILOT_MATCH_IDS = (
    "9ff2c0eb-17ab-4ed0-a0b8-b13613f1f6aa",
    "5edc5b35-2f92-449b-b266-a4948e319f4a",
    "a20a9560-aa1b-4ac3-a185-25a33c09bdad",
    "f39d18e1-c615-4e5a-8059-dc6b01d0d622",
    "9a23c90b-7073-484f-9d89-381181a8696f",
)

_MUX_ID_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
_VIMEO_ID_PATTERN = re.compile(r"^\d+$")
_engine: Engine | None = None


class LgiSourceError(RuntimeError):
    code = "LGI_SOURCE_ERROR"


class LgiSourceUnavailable(LgiSourceError):
    code = "LGI_SOURCE_UNAVAILABLE"


class LgiMatchNotFound(LgiSourceError):
    code = "LGI_MATCH_NOT_FOUND"


class LgiPlaybackUnavailable(LgiSourceError):
    code = "LGI_PLAYBACK_UNAVAILABLE"


@dataclass(frozen=True)
class LgiLineupPlayer:
    id: str
    name: str
    side: str
    shirt_number: str | None
    role: str | None
    starter: bool
    captain: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "side": self.side,
            "shirt_number": self.shirt_number,
            "role": self.role,
            "starter": self.starter,
            "captain": self.captain,
        }


@dataclass(frozen=True)
class LgiMatch:
    id: str
    title: str
    slug: str
    home_team: str
    away_team: str
    competition: str | None
    season: str | None
    duration_seconds: int | None
    provider: str
    mux_playback_id: str | None
    vimeo_video_id: str | None
    vimeo_player_embed_url: str | None
    lineup: tuple[LgiLineupPlayer, ...] = ()
    lineup_count: int = 0

    def as_dict(self, *, include_lineup: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "competition": self.competition,
            "season": self.season,
            "duration_seconds": self.duration_seconds,
            "provider": self.provider,
            "lineup_count": max(len(self.lineup), self.lineup_count),
            "pilot": self.id in LGI_PILOT_MATCH_IDS,
        }
        if include_lineup:
            payload["lineup"] = [player.as_dict() for player in self.lineup]
        return payload


def _database_url() -> str:
    value = (os.environ.get("LGI_READONLY_DATABASE_URL") or "").strip()
    if not value:
        raise LgiSourceUnavailable("LGI_READONLY_DATABASE_URL is not configured.")
    return value


def _normalized_database_url() -> str:
    value = _database_url()
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://") :]
    if value.startswith("postgresql://"):
        value = "postgresql+psycopg2://" + value[len("postgresql://") :]
    parsed = make_url(value)
    if parsed.get_backend_name() != "postgresql":
        raise LgiSourceUnavailable("LGI read-only source must be PostgreSQL.")
    return value


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            _normalized_database_url(),
            poolclass=QueuePool,
            pool_size=2,
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={
                "options": (
                    "-c default_transaction_read_only=on "
                    "-c statement_timeout=15000 "
                    "-c idle_in_transaction_session_timeout=15000"
                )
            },
        )
    return _engine


def reset_engine_for_tests() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def assert_read_only_connection(engine: Engine | None = None) -> dict[str, object]:
    source_engine = engine or get_engine()
    with source_engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                  current_user AS role_name,
                  current_setting('transaction_read_only') AS transaction_read_only,
                  has_table_privilege(current_user, 'public.matches', 'SELECT') AS can_select_matches,
                  has_table_privilege(current_user, 'public.matches', 'INSERT') AS can_insert_matches,
                  has_table_privilege(current_user, 'public.matches', 'UPDATE') AS can_update_matches,
                  has_table_privilege(current_user, 'public.matches', 'DELETE') AS can_delete_matches
                """
            )
        ).mappings().one()
    if row["transaction_read_only"] != "on":
        raise LgiSourceUnavailable("LGI connection is not transaction read-only.")
    if not row["can_select_matches"]:
        raise LgiSourceUnavailable("LGI role cannot read public.matches.")
    if any(
        row[key]
        for key in (
            "can_insert_matches",
            "can_update_matches",
            "can_delete_matches",
        )
    ):
        raise LgiSourceUnavailable("LGI role has forbidden write privileges.")
    return {
        "ok": True,
        "role": row["role_name"],
        "transaction_read_only": True,
        "write_privileges": False,
    }


def build_source_uri(match_id: str) -> str:
    normalized = str(UUID(str(match_id)))
    return f"{LGI_SOURCE_SCHEME}://{LGI_SOURCE_HOST}/{normalized}"


def parse_source_uri(value: str) -> str | None:
    parsed = urlsplit(str(value or ""))
    if parsed.scheme != LGI_SOURCE_SCHEME or parsed.netloc != LGI_SOURCE_HOST:
        return None
    raw_id = parsed.path.strip("/")
    try:
        return str(UUID(raw_id))
    except (ValueError, TypeError):
        return None


def _row_to_match(row: dict[str, object], lineup: tuple[LgiLineupPlayer, ...] = ()) -> LgiMatch:
    provider = str(row.get("provider") or "").strip().lower()
    return LgiMatch(
        id=str(row["id"]),
        title=str(row.get("title") or ""),
        slug=str(row.get("slug") or ""),
        home_team=str(row.get("home_team") or ""),
        away_team=str(row.get("away_team") or ""),
        competition=str(row["competition"]) if row.get("competition") else None,
        season=str(row["season"]) if row.get("season") else None,
        duration_seconds=(
            int(row["duration_seconds"])
            if row.get("duration_seconds") is not None
            else (
                int(row["vimeo_duration"])
                if row.get("vimeo_duration") is not None
                else None
            )
        ),
        provider=provider,
        mux_playback_id=(
            str(row["mux_playback_id"]) if row.get("mux_playback_id") else None
        ),
        vimeo_video_id=(
            str(row["vimeo_video_id"]) if row.get("vimeo_video_id") else None
        ),
        vimeo_player_embed_url=(
            str(row["vimeo_player_embed_url"])
            if row.get("vimeo_player_embed_url")
            else None
        ),
        lineup=lineup,
        lineup_count=(
            int(row["starter_count"])
            if row.get("starter_count") is not None
            else len(lineup)
        ),
    )


def get_match(match_id: str, engine: Engine | None = None) -> LgiMatch:
    normalized_id = str(UUID(str(match_id)))
    source_engine = engine or get_engine()
    with source_engine.connect() as connection:
        match_row = (
            connection.execute(
                text(
                    """
                    SELECT id, title, slug, home_team, away_team, competition, season,
                           duration_seconds, vimeo_duration, provider, mux_playback_id,
                           vimeo_video_id, vimeo_player_embed_url
                    FROM public.matches
                    WHERE id = CAST(:match_id AS uuid)
                      AND published IS TRUE
                      AND lower(provider) IN ('mux', 'vimeo')
                    LIMIT 1
                    """
                ),
                {"match_id": normalized_id},
            )
            .mappings()
            .one_or_none()
        )
        if match_row is None:
            raise LgiMatchNotFound(f"LGI match {normalized_id} was not found.")

        lineup_rows = (
            connection.execute(
                text(
                    """
                    SELECT p.id, p.name, mlp.side::text AS side,
                           mlp.shirt_number, COALESCE(mlp.role, p.position_group) AS role,
                           mlp.starter, mlp.captain
                    FROM public.match_lineup_players mlp
                    JOIN public.players p ON p.id = mlp.player_id
                    WHERE mlp.match_id = CAST(:match_id AS uuid)
                    ORDER BY mlp.side, mlp.starter DESC, mlp.sort_order, p.name
                    """
                ),
                {"match_id": normalized_id},
            )
            .mappings()
            .all()
        )

    lineup = tuple(
        LgiLineupPlayer(
            id=str(row["id"]),
            name=str(row["name"]),
            side=str(row["side"]).upper(),
            shirt_number=(
                str(row["shirt_number"]) if row.get("shirt_number") is not None else None
            ),
            role=str(row["role"]) if row.get("role") else None,
            starter=bool(row["starter"]),
            captain=bool(row["captain"]),
        )
        for row in lineup_rows
    )
    return _row_to_match(dict(match_row), lineup)


def list_matches(
    *,
    query: str = "",
    limit: int = 30,
    pilot_only: bool = False,
    engine: Engine | None = None,
) -> list[LgiMatch]:
    source_engine = engine or get_engine()
    bounded_limit = max(1, min(int(limit), 100))
    normalized_query = str(query or "").strip()
    pilot_ids = list(LGI_PILOT_MATCH_IDS)
    with source_engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT m.id, m.title, m.slug, m.home_team, m.away_team,
                           m.competition, m.season, m.duration_seconds,
                           m.vimeo_duration, m.provider, m.mux_playback_id,
                           m.vimeo_video_id, m.vimeo_player_embed_url,
                           COUNT(*) FILTER (WHERE mlp.starter IS TRUE) AS starter_count,
                           COUNT(*) FILTER (
                             WHERE mlp.starter IS TRUE
                               AND NULLIF(trim(mlp.shirt_number), '') IS NOT NULL
                           ) AS numbered_starter_count
                    FROM public.matches m
                    JOIN public.match_lineup_players mlp ON mlp.match_id = m.id
                    WHERE m.published IS TRUE
                      AND lower(m.provider) IN ('mux', 'vimeo')
                      AND (
                        :query = ''
                        OR m.title ILIKE :pattern
                        OR m.home_team ILIKE :pattern
                        OR m.away_team ILIKE :pattern
                        OR m.competition ILIKE :pattern
                      )
                      AND (
                        :pilot_only IS FALSE
                        OR m.id::text = ANY(CAST(:pilot_ids AS text[]))
                      )
                    GROUP BY m.id
                    HAVING COUNT(*) FILTER (WHERE mlp.starter IS TRUE) >= 22
                       AND COUNT(*) FILTER (
                         WHERE mlp.starter IS TRUE
                           AND NULLIF(trim(mlp.shirt_number), '') IS NOT NULL
                       ) >= 22
                    ORDER BY
                      CASE WHEN m.id::text = ANY(CAST(:pilot_ids AS text[])) THEN 0 ELSE 1 END,
                      m.starts_at DESC NULLS LAST,
                      m.title
                    LIMIT :limit
                    """
                ),
                {
                    "query": normalized_query,
                    "pattern": f"%{normalized_query}%",
                    "pilot_only": bool(pilot_only),
                    "pilot_ids": pilot_ids,
                    "limit": bounded_limit,
                },
            )
            .mappings()
            .all()
        )
    return [_row_to_match(dict(row)) for row in rows]


def _notify_progress(callback: Callable[[], None] | None, last_tick: list[float]) -> None:
    if callback is None:
        return
    now = time.monotonic()
    if now - last_tick[0] >= 5:
        last_tick[0] = now
        callback()


def _download_mux(
    match: LgiMatch,
    destination: Path,
    progress_callback: Callable[[], None] | None,
) -> None:
    playback_id = str(match.mux_playback_id or "").strip()
    if not _MUX_ID_PATTERN.fullmatch(playback_id):
        raise LgiPlaybackUnavailable("LGI Mux playback ID is missing or invalid.")
    hls_url = f"https://stream.mux.com/{playback_id}.m3u8"
    headers = f"Referer: {LGI_DEFAULT_REFERER}\r\n"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-headers",
        headers,
        "-i",
        hls_url,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    last_tick = [time.monotonic()]
    while process.poll() is None:
        _notify_progress(progress_callback, last_tick)
        time.sleep(1)
    stderr = process.stderr.read() if process.stderr else ""
    if process.returncode != 0:
        raise LgiPlaybackUnavailable(
            f"Mux download failed for LGI match {match.id}: {stderr[-400:]}"
        )


def _safe_vimeo_embed_url(match: LgiMatch) -> str:
    video_id = str(match.vimeo_video_id or "").strip()
    if not _VIMEO_ID_PATTERN.fullmatch(video_id):
        raise LgiPlaybackUnavailable("LGI Vimeo video ID is missing or invalid.")
    configured = str(match.vimeo_player_embed_url or "").strip()
    if configured:
        parsed = urlsplit(configured)
        if parsed.scheme == "https" and parsed.hostname == "player.vimeo.com":
            expected_path = f"/video/{video_id}"
            if parsed.path.rstrip("/") == expected_path:
                return configured
    return f"https://player.vimeo.com/video/{video_id}"


def _download_vimeo(
    match: LgiMatch,
    destination: Path,
    progress_callback: Callable[[], None] | None,
) -> None:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise LgiPlaybackUnavailable("yt-dlp is not installed in the worker.") from exc

    referer = (os.environ.get("LGI_VIMEO_REFERER") or LGI_DEFAULT_REFERER).strip()
    last_tick = [time.monotonic()]

    def hook(_status: dict[str, object]) -> None:
        _notify_progress(progress_callback, last_tick)

    options = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "outtmpl": str(destination),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 4,
        "http_headers": {
            "Referer": referer,
            "Origin": referer.rstrip("/"),
            "User-Agent": "AlgoNextWorker/1.0",
        },
        "progress_hooks": [hook],
    }
    try:
        with YoutubeDL(options) as downloader:
            result = downloader.extract_info(
                _safe_vimeo_embed_url(match),
                download=True,
            )
            requested = Path(downloader.prepare_filename(result))
    except Exception as exc:
        raise LgiPlaybackUnavailable(
            f"Vimeo download failed for LGI match {match.id}."
        ) from exc

    if not destination.exists() and requested.exists():
        requested.replace(destination)
    if not destination.exists() or destination.stat().st_size <= 0:
        raise LgiPlaybackUnavailable(
            f"Vimeo download produced no file for LGI match {match.id}."
        )


def download_match(
    match_id: str,
    destination: Path,
    progress_callback: Callable[[], None] | None = None,
) -> LgiMatch:
    destination.parent.mkdir(parents=True, exist_ok=True)
    match = get_match(match_id)
    logger.info(
        "LGI_READONLY_DOWNLOAD_START match_id=%s provider=%s",
        match.id,
        match.provider,
    )
    if match.provider == "mux":
        _download_mux(match, destination, progress_callback)
    elif match.provider == "vimeo":
        _download_vimeo(match, destination, progress_callback)
    else:
        raise LgiPlaybackUnavailable(
            f"Unsupported LGI provider for match {match.id}."
        )
    logger.info(
        "LGI_READONLY_DOWNLOAD_COMPLETE match_id=%s provider=%s bytes=%s",
        match.id,
        match.provider,
        destination.stat().st_size if destination.exists() else 0,
    )
    return match
