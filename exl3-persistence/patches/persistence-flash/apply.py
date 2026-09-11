#!/usr/bin/env python3
"""Hash-pinned, preflighted native patch application (no vLLM imports).

Use a quiescent source tree. All changed files are validated and patched in a
private temporary tree before replacing any source. This is not a live updater.

``source_tree`` is either a source root containing ``vllm/`` or the ``vllm``
package directory itself (a bare overlay tree, as shipped in some runtime
images). Manifest keys are always ``vllm/...``; the ``vllm/`` prefix is stripped
when the given path already *is* that package. Staging always reconstructs the
full ``vllm/...`` layout so the patches apply verbatim.
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

HERE = Path(__file__).resolve().parent
PACKAGE = "vllm"


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


def run(root: Path, mode: str) -> None:
    root, bare = resolve_root(root)
    spec = manifest()
    for entry in spec["patches"]:
        if digest((HERE / entry["file"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"Patch digest mismatch: {entry['file']}")
    before, after = ("after", "before") if mode == "reverse" else ("before", "after")
    check_tree(root, "after" if mode == "verify" else before, spec, bare)
    if mode == "verify":
        return
    with tempfile.TemporaryDirectory(prefix="native-patch-") as directory:
        staged = Path(directory)
        originals = {}
        for name in spec["files"]:
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            originals[name] = source_path(root, bare, name).read_bytes()
            target.write_bytes(originals[name])
        patches = list(spec["patches"])
        if mode == "reverse":
            patches.reverse()
        for entry in patches:
            command = ["git", "apply", "--whitespace=error"]
            if mode == "reverse":
                command.append("--reverse")
            command.append(str(HERE / entry["file"]))
            subprocess.run(command[:2] + ["--check"] + command[2:], cwd=staged, check=True)
            subprocess.run(command, cwd=staged, check=True)
        check_tree(staged, after, spec)
        if mode == "check":
            return
        # Recheck after staging to catch concurrent modifications before writes.
        check_tree(root, before, spec, bare)
        replaced = []
        try:
            for name in spec["files"]:
                target = source_path(root, bare, name)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                    tmp.write((staged / name).read_bytes())
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
    check_tree(root, after, spec, bare)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "apply", "verify", "reverse"])
    parser.add_argument(
        "source_tree",
        type=Path,
        help="source root containing vllm/, or the vllm package directory itself",
    )
    args = parser.parse_args()
    try:
        run(args.source_tree.resolve(), args.mode)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Native patch refused: {error}\n")
    print(f"Native patch {args.mode}: OK")


if __name__ == "__main__":
    main()
