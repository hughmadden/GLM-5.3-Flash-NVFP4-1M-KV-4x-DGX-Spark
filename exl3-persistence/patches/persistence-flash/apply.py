#!/usr/bin/env python3
"""Hash-pinned overlay of the persistence connector series from its vLLM fork branch.

The series lives as commits on a public vLLM fork (``manifest.json`` ``fork``:
repo, branch, tip commit, and the per-patch commit list). This tool does not
carry or apply diffs: it fetches the eight touched files at the pinned fork
commit, verifies each against the manifest's AFTER sha256 *before* anything is
written, and overlays them onto a pristine upstream tree whose BEFORE hashes
match. ``reverse`` does the opposite with the upstream files at
``upstream_commit``. Nothing is written unless every fetched byte verifies.

Modes:
  fetch DEST        stage the pinned fork files under DEST/vllm/... (hash-verified)
  check TREE        TREE is pristine upstream and the fork files verify (no write)
  apply TREE        overlay the verified fork files onto a pristine TREE
  verify TREE       TREE already carries the series (AFTER hashes + anchors)
  reverse TREE      restore the pristine upstream files onto a ported TREE

Sources: ``--from DIR`` (or ``PERSIST_FORK_SOURCE``) reads the fork files from
a local directory laid out as ``vllm/...`` (the output of ``fetch``, or a fork
checkout) instead of GitHub; ``--upstream-from DIR`` (``PERSIST_UPSTREAM_SOURCE``)
does the same for ``reverse``. Without them the files come from
raw.githubusercontent.com at the pinned commits. Either way the hashes decide.

``TREE`` is either a source root containing ``vllm/`` or the ``vllm`` package
directory itself (a bare overlay tree, as shipped in runtime images). Manifest
keys are always ``vllm/...``. Use a quiescent tree; this is a build operation,
not a live updater.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
PACKAGE = "vllm"
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{name}"


def resolve_root(root: Path) -> tuple[Path, bool]:
    """Return (root, bare) where bare means root IS the vllm package dir."""
    if (root / PACKAGE / "v1" / "kv_offload" / "base.py").is_file():
        return root, False
    if (root / "v1" / "kv_offload" / "base.py").is_file():
        return root, True
    raise ValueError(f"Not a vLLM source root or vllm package: {root}")


def source_path(root: Path, bare: bool, name: str) -> Path:
    """Map a manifest key (always ``vllm/...``) onto the given tree."""
    if bare:
        assert name.startswith(PACKAGE + "/")
        return root / name[len(PACKAGE) + 1 :]
    return root / name


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest() -> dict:
    return json.loads((HERE / "manifest.json").read_text())


def check_tree(root: Path, state: str, spec: dict, bare: bool = False) -> None:
    for name, entry in spec["files"].items():
        path = source_path(root, bare, name)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked source: {name}")
        data = path.read_bytes()
        if digest(data) != entry[state]:
            raise ValueError(f"Unexpected {state} SHA256: {name}")
        for anchor in entry["anchors"]:
            if data.decode().count(anchor) != 1:
                raise ValueError(f"Nonunique/missing source anchor: {name}: {anchor!r}")


def _read_source(source: str, repo: str, commit: str, name: str) -> bytes:
    """Read one manifest file from a local dir, a base URL, or GitHub raw."""
    if source:
        if source.startswith(("http://", "https://")):
            url = source.rstrip("/") + "/" + name
        else:
            root = Path(source).resolve()
            if not root.is_dir():
                raise ValueError(f"Source directory missing: {root}")
            root, bare = resolve_root(root)
            path = source_path(root, bare, name)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or symlinked source file: {path}")
            return path.read_bytes()
    else:
        url = RAW.format(repo=repo, commit=commit, name=name)
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status} fetching {url}")
        return response.read()


def fetch_state(spec: dict, state: str, source: str | None = None) -> dict[str, bytes]:
    """Fetch every manifest file in ``state`` ("after" = fork, "before" =
    upstream) and verify each sha256 before returning. Fails closed."""
    if state == "after":
        repo, commit = spec["fork"]["repo"], spec["fork"]["commit"]
        source = source if source is not None else os.environ.get("PERSIST_FORK_SOURCE", "")
    else:
        repo, commit = spec["upstream_repo"], spec["upstream_commit"]
        source = source if source is not None else os.environ.get("PERSIST_UPSTREAM_SOURCE", "")
    files = {}
    for name, entry in spec["files"].items():
        data = _read_source(source, repo, commit, name)
        if digest(data) != entry[state]:
            raise ValueError(
                f"Fetched {state} SHA256 mismatch for {name} "
                f"(from {source or repo + '@' + commit[:10]})"
            )
        files[name] = data
    return files


def write_files(root: Path, bare: bool, files: dict[str, bytes], spec: dict,
                expect_before: str, expect_after: str) -> None:
    """Atomically replace the manifest files, rolling back on any failure."""
    # Recheck immediately before writing to catch concurrent modification.
    check_tree(root, expect_before, spec, bare)
    originals = {name: source_path(root, bare, name).read_bytes() for name in spec["files"]}
    replaced = []
    try:
        for name in spec["files"]:
            target = source_path(root, bare, name)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                tmp.write(files[name])
                tmp.flush()
                os.fsync(tmp.fileno())
                temporary = Path(tmp.name)
            shutil.copymode(target, temporary)
            os.replace(temporary, target)
            replaced.append(name)
    except BaseException:
        for name in replaced:
            source_path(root, bare, name).write_bytes(originals[name])
        raise
    check_tree(root, expect_after, spec, bare)


def fetch(dest: Path, spec: dict, source: str | None = None) -> None:
    """Stage the verified fork files under dest/vllm/... (build-context staging)."""
    files = fetch_state(spec, "after", source)
    dest = dest.resolve()
    for name, data in files.items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (dest / "FORK.json").write_text(json.dumps(
        {"repo": spec["fork"]["repo"], "branch": spec["fork"]["branch"],
         "commit": spec["fork"]["commit"], "upstream_commit": spec["upstream_commit"]},
        indent=2) + "\n")


def run(root: Path, mode: str, source: str | None = None, upstream_source: str | None = None) -> None:
    spec = manifest()
    if mode == "fetch":
        fetch(root, spec, source)
        return
    root, bare = resolve_root(root)
    if mode == "verify":
        check_tree(root, "after", spec, bare)
        return
    if mode == "reverse":
        check_tree(root, "after", spec, bare)
        files = fetch_state(spec, "before", upstream_source)
        write_files(root, bare, files, spec, "after", "before")
        return
    # check / apply
    check_tree(root, "before", spec, bare)
    files = fetch_state(spec, "after", source)
    if mode == "check":
        return
    write_files(root, bare, files, spec, "before", "after")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["fetch", "check", "apply", "verify", "reverse"])
    parser.add_argument(
        "tree", type=Path,
        help="source root containing vllm/, the vllm package directory itself, "
             "or (for fetch) the staging directory to write",
    )
    parser.add_argument("--from", dest="source", default=None,
                        help="local dir or base URL holding the fork files as vllm/... "
                             "(default: PERSIST_FORK_SOURCE, else GitHub raw at the pinned commit)")
    parser.add_argument("--upstream-from", dest="upstream_source", default=None,
                        help="same for the pristine upstream files used by reverse "
                             "(default: PERSIST_UPSTREAM_SOURCE, else GitHub raw)")
    args = parser.parse_args()
    try:
        run(args.tree.resolve(), args.mode, args.source, args.upstream_source)
    except (ValueError, OSError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        parser.exit(1, f"Native patch refused: {error}\n")
    spec = manifest()
    print(f"Native patch {args.mode}: OK (fork {spec['fork']['repo']}@{spec['fork']['commit'][:10]}"
          f" on upstream {spec['upstream_commit'][:10]})")


if __name__ == "__main__":
    main()
