"""Administration command: ``python -m lainbow.server.admin`` (``lainbow-admin``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from lainbow.core.settings import ServerSettings, SettingsError, load_settings
from lainbow.db.migrate import upgrade_to_head
from lainbow.db.session import make_engine, make_session_factory
from lainbow.server import engines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lainbow-admin", description="Lainbow administration.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add-engine", help="register an engine and print its token")
    add.add_argument("name")
    sub.add_parser("list-engines", help="list registered engines")
    rm = sub.add_parser("remove-engine", help="remove an engine and release its leases")
    rm.add_argument("name")
    return parser


def _format_table(rows: list[tuple[str, str, str]]) -> str:
    header = ("NAME", "LAST SEEN", "COMPLETED")
    all_rows = [header, *rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(3)]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip() for r in all_rows
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the admin command and return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        settings = load_settings(ServerSettings)
    except SettingsError as e:
        print(e, file=sys.stderr)
        return 2
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
