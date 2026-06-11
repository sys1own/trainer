from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import orchestrator


def build_command(args: argparse.Namespace) -> int:
    orchestrator.configure_logging(verbose=args.verbose)
    metadata = orchestrator.run_build(
        workspace_root=str(Path(args.workspace).resolve()),
        cycles=args.cycles,
        telemetry_interval=args.telemetry_interval,
    )
    print()
    print("Build completed successfully.")
    print(f"Manifest: {metadata.get('manifest_path')}")
    for asset in metadata.get("applied_assets", []):
        print(f"Updated asset: {asset}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Builder orchestration CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Run the full builder pipeline")
    build_parser.add_argument("--workspace", default=".", help="Workspace root to build")
    build_parser.add_argument("--cycles", type=int, default=3, help="Number of orchestration cycles")
    build_parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=2.0,
        help="Seconds between telemetry refreshes",
    )
    build_parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    build_parser.set_defaults(handler=build_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())