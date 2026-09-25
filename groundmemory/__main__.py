"""
groundmemory CLI entry point.

Usage
-----
Re-index all workspace files:

    groundmemory --sync

Create a manual backup:

    groundmemory --backup

List available backups:

    groundmemory --list-backups

Restore a workspace backup:

    groundmemory --restore -1                     # most recent backup
    groundmemory --restore 2026-04-08             # exact date (error if ambiguous)
    groundmemory --restore 2026-04-08_165530      # exact timestamp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _get_workspace_path() -> Path:
    """Resolve the active workspace path from the environment / defaults."""
    from groundmemory.config import groundmemoryConfig
    cfg = groundmemoryConfig.auto()
    return cfg.root_dir / cfg.workspace


def cmd_sync(workspace_path: Path, config=None) -> None:
    from groundmemory.config import groundmemoryConfig
    from groundmemory.core.workspace import Workspace
    from groundmemory.core.index import MemoryIndex
    from groundmemory.core.embeddings import make_provider
    from groundmemory.core.sync import sync_workspace

    cfg = config if config is not None else groundmemoryConfig.auto()
    ws = Workspace(workspace_path, custom_files=cfg.custom_files)
    index = MemoryIndex(ws.db_path)
    provider = make_provider(cfg.embedding)
    result = sync_workspace(ws, index, provider, cfg.chunking, force=True)
    index.close()
    indexed = result.get("added", 0) + result.get("updated", 0)
    print(f"Sync complete for workspace '{workspace_path.name}':")
    print(f"  indexed:  {indexed}")
    print(f"  skipped:  {result.get('skipped', 0)}")
    print(f"  deleted:  {result.get('deleted', 0)}")


def cmd_backup(workspace_path: Path) -> None:
    from groundmemory.core.backup import create_backup
    archive = create_backup(workspace_path)
    print(f"Backup created: {archive.stem}")
    print(f"  → {archive}")


def cmd_list_backups(workspace_path: Path) -> None:
    from groundmemory.core.backup import list_backups
    backups = list_backups(workspace_path)
    if not backups:
        print(f"No backups found for workspace: {workspace_path}")
        return
    print(f"Backups for workspace '{workspace_path.name}':")
    for i, b in enumerate(reversed(backups), start=1):
        print(f"  -{i}  {b.stem}")


def cmd_restore(spec: str, workspace_path: Path) -> None:
    from groundmemory.core.backup import list_backups, parse_spec, restore_backup

    backups = list_backups(workspace_path)
    if not backups:
        print(f"No backups found for workspace: {workspace_path}", file=sys.stderr)
        sys.exit(1)

    target = parse_spec(spec, backups)

    if target is None:
        # Check for ambiguous date match
        matches = [b for b in backups if b.stem.startswith(spec)]
        if matches:
            print(
                f"Ambiguous spec '{spec}' matches {len(matches)} backups. "
                "Specify an exact timestamp:\n",
                file=sys.stderr,
            )
            for m in matches:
                print(f"  {m.stem}", file=sys.stderr)
        else:
            print(f"No backup found matching '{spec}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Restoring backup: {target.stem}")
    print(f"  → workspace: {workspace_path}")
    print()
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)

    restore_backup(target, workspace_path)
    print(f"\nRestored successfully from '{target.stem}'.")
    print("If an MCP server is running against this workspace, restart it.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="groundmemory",
        description="GroundMemory workspace management CLI.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Re-index all workspace files (useful after manual edits outside the agent).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a manual backup of the current workspace.",
    )
    parser.add_argument(
        "--restore",
        metavar="SPEC",
        help=(
            "Restore a workspace backup. SPEC can be: "
            "'-1' (most recent), '-2' (second-most-recent), "
            "'YYYY-MM-DD' (exact date), or 'YYYY-MM-DD_HHmmss' (exact timestamp)."
        ),
    )
    parser.add_argument(
        "--list-backups",
        action="store_true",
        help="List all available backups for the current workspace.",
    )
    args = parser.parse_args()

    workspace_path = _get_workspace_path()

    if args.sync:
        cmd_sync(workspace_path)
    elif args.backup:
        cmd_backup(workspace_path)
    elif args.list_backups:
        cmd_list_backups(workspace_path)
    elif args.restore:
        cmd_restore(args.restore, workspace_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()