"""Administration command: ``python -m lainbow.server.admin`` (``lainbow-admin``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from lainbow.core.settings import (
    ServerSettings,
    SettingsError,
    describe,
    format_settings,
    load_settings,
)
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.schema import ActiveProfile, Run, Track
from lainbow.db.session import make_engine, make_session_factory
from lainbow.db.work import progress
from lainbow.server import engines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lainbow-admin", description="Lainbow administration.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add-engine", help="register an engine and print its token")
    add.add_argument("name")
    sub.add_parser("list-engines", help="list registered engines")
    rm = sub.add_parser("remove-engine", help="remove an engine and release its leases")
    rm.add_argument("name")
    sub.add_parser("status", help="show library, analysis and engine status")
    sub.add_parser("config", help="show every effective setting")
    return parser


def _format_rows(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    all_rows = [header, *rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip() for r in all_rows
    )


def _format_table(rows: list[tuple[str, str, str]]) -> str:
    return _format_rows(("NAME", "LAST SEEN", "COMPLETED"), [tuple(r) for r in rows])


def _relative(moment: datetime | None, now: datetime) -> str:
    if moment is None:
        return "never"
    seconds = max(0, int((now - moment).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def _status_text(session: Session) -> str:
    now = datetime.now(UTC)
    total = session.scalar(select(func.count()).select_from(Track)) or 0
    sections = [f"Tracks: {total}"]

    model_rows: list[tuple[str, ...]] = []
    for active in session.scalars(select(ActiveProfile).order_by(ActiveProfile.model)):
        p = progress(session, model=active.model, profile=active.profile)
        percent = f"{100 * p.done / p.total:.1f}%" if p.total else "-"
        model_rows.append(
            (
                active.model,
                active.profile,
                str(p.done),
                str(p.pending),
                str(p.leased),
                str(p.failed),
                percent,
            )
        )
    sections.append(
        _format_rows(
            ("MODEL", "PROFILE", "DONE", "PENDING", "LEASED", "FAILED", "PERCENT"), model_rows
        )
    )

    engine_rows: list[tuple[str, ...]] = [
        (e.name, _relative(e.last_seen_at, now), str(e.completed))
        for e in engines.list_engines(session)
    ]
    sections.append(_format_rows(("ENGINE", "LAST SEEN", "COMPLETED"), engine_rows))

    run_rows: list[tuple[str, ...]] = []
    for kind in ("scan", "vacuum", "analyze"):
        run = session.scalar(
            select(Run).where(Run.kind == kind).order_by(Run.created_at.desc()).limit(1)
        )
        if run is None:
            run_rows.append((kind, "never", "-"))
        else:
            run_rows.append((kind, run.state, _relative(run.created_at, now)))
    sections.append(_format_rows(("LAST RUN", "STATE", "STARTED"), run_rows))
    return "\n\n".join(sections)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the admin command and return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        settings = load_settings(ServerSettings)
    except SettingsError as e:
        print(e, file=sys.stderr)
        return 2
    if args.command == "config":
        print(format_settings(describe(settings)))
        return 0
    url = settings.database_url.get_secret_value()
    upgrade_to_head(url)
    db_engine = make_engine(url)
    factory = make_session_factory(db_engine)
    try:
        with factory() as session:
            if args.command == "add-engine":
                try:
                    token = engines.create_engine(session, args.name)
                except engines.EngineExistsError as e:
                    print(f"error: {e}", file=sys.stderr)
                    return 1
                session.commit()
                print(token)
            elif args.command == "status":
                print(_status_text(session))
            elif args.command == "list-engines":
                rows = [
                    (
                        e.name,
                        e.last_seen_at.strftime("%Y-%m-%d %H:%M:%S") if e.last_seen_at else "never",
                        str(e.completed),
                    )
                    for e in engines.list_engines(session)
                ]
                print(_format_table(rows))
            else:
                if not engines.remove_engine(session, args.name):
                    print(f"error: engine {args.name!r} not found", file=sys.stderr)
                    return 1
                session.commit()
                print(f"removed engine {args.name!r}")
    finally:
        db_engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
