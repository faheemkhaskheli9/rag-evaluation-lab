"""Ingestion CLI: ``python -m ragel.cli ingest FILE...`` / ``list`` / ``text ID``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_settings
from .ingest import CorruptDocumentError, IngestStore, UnsupportedTypeError


def _store(config_path: str | None) -> IngestStore:
    settings = load_settings(config_path)
    return IngestStore(settings.storage_dir, settings.max_upload_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragel", description="RAG eval lab ingestion CLI")
    parser.add_argument("--config", help="path to a JSON config file")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="ingest PDF/text files")
    ing.add_argument("paths", nargs="+", type=Path)
    sub.add_parser("list", help="list ingested documents")
    txt = sub.add_parser("text", help="print extracted text for a document id")
    txt.add_argument("document_id")

    args = parser.parse_args(argv)
    store = _store(args.config)

    if args.command == "list":
        for r in store.list_documents():
            print(f"{r.document_id}  {r.content_type:4}  {r.char_count:>8} chars  {r.filename}")
        return 0

    if args.command == "text":
        text = store.read_text(args.document_id)
        if text is None:
            print(f"error: no document {args.document_id!r}", file=sys.stderr)
            return 1
        print(text)
        return 0

    exit_code = 0
    for path in args.paths:
        try:
            content = path.read_bytes()
        except OSError as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            record = store.ingest(path.name, content)
        except (UnsupportedTypeError, CorruptDocumentError, ValueError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"ingested {path.name} -> {record.document_id} ({record.char_count} chars)")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
