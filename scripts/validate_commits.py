#!/usr/bin/env python3
"""Validate Git commit messages with Commitizen conventional commits."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def commit_hashes(revision_range: str | None) -> list[str]:
    command = ["git", "rev-list", "--reverse"]
    command.append(revision_range or "HEAD")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [commit for commit in result.stdout.splitlines() if commit]


def commit_message(commit: str) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%B", commit],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_message(message: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("cz") or "cz"
    return subprocess.run(
        [executable, "check", "--message", message],
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate commit messages against Conventional Commits 1.0.0."
    )
    parser.add_argument(
        "--range",
        dest="revision_range",
        help="Git revision range to validate, for example origin/main..HEAD.",
    )
    parser.add_argument(
        "--message",
        help="Validate one explicit commit message instead of reading Git history.",
    )
    args = parser.parse_args()

    failures: list[tuple[str, str]] = []

    if args.message is not None:
        result = validate_message(args.message)
        if result.returncode != 0:
            failures.append(("provided message", result.stderr or result.stdout))
    else:
        for commit in commit_hashes(args.revision_range):
            message = commit_message(commit)
            result = validate_message(message)
            if result.returncode != 0:
                subject = message.splitlines()[0] if message else "<empty>"
                failures.append((f"{commit[:12]} {subject}", result.stderr or result.stdout))

    if failures:
        print("Invalid conventional commit messages:", file=sys.stderr)
        for commit, error in failures:
            print(f"\n- {commit}", file=sys.stderr)
            print(error.strip(), file=sys.stderr)
        return 1

    print("All commit messages are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
