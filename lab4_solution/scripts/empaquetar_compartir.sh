#!/usr/bin/env bash
# Build a public sharing folder and ZIP from the repository's exact publication rules.
set -euo pipefail

package_project() {
    local mode=$1 script=$2 label=$3
    shift 3
    local dry=0 data=0 audio=0 named=0 runtime
    while (($#)); do
        case "$1" in
            -h|--help)
                if [[ $mode == drive ]]; then
                    printf '%s\n' 'Usage: bash empaquetar_drive.sh [--dry-run] [--include-data] [--include-private-audio] [folder-name]'
                else
                    printf '%s\n' 'Usage: bash empaquetar_compartir.sh [--dry-run] [folder-name]'
                fi
                return ;;
            --dry-run) dry=1 ;;
            --include-data) [[ $mode == drive ]] || { echo 'Drive option only.' >&2; return 2; }; data=1 ;;
            --include-private-audio) [[ $mode == drive ]] || { echo 'Drive option only.' >&2; return 2; }; audio=1 ;;
            -*) printf 'Unknown option: %s\n' "$1" >&2; return 2 ;;
            *) if ((named)); then echo 'Expected one folder name.' >&2; return 2; fi
               label=$1; named=1 ;;
        esac
        shift
    done
    command -v git >/dev/null || { echo 'Git is required.' >&2; return 1; }
    for runtime in python python3; do
        if "$runtime" -I -B -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; then break; fi
        runtime=''
    done
    [[ -n $runtime ]] || { echo 'Python 3.8 or later is required.' >&2; return 1; }
    "$runtime" -I -B - "$script" "$mode" "$label" "$dry" "$data" "$audio" <<'PY'
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def fail(message):
    raise SystemExit(message)


def checked(path, root):
    """Reject links, including Windows junctions, before accessing descendants."""
    current = root
    for part in (None, *path.relative_to(root).parts):
        if part is not None:
            if part in (".", ".."):
                fail(f"Invalid path: {path}")
            current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            fail(f"Linked path: {current}")
    return path


def main():
    script, mode, label, dry, data, audio = sys.argv[1:]
    script = Path(os.path.abspath(script))
    root = script.parents[2]
    checked(script, root)
    if script.parent != root / "lab4_solution/scripts":
        fail("Expected the project scripts directory.")
    if not checked(root / ".git", root).is_dir():
        fail("A local Git repository or normal clone is required.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,63}", label) or label.endswith((".", " ")):
        fail("Use a single output folder name under dist/.")
    if label.split(".")[0].upper() in {"CON", "NUL", "AUX", "PRN", *["COM"+str(i) for i in range(1,10)], *["LPT"+str(i) for i in range(1,10)]}:
        fail("Reserved output folder name.")
    private = data == "1" or audio == "1"
    if mode not in ("public", "drive") or (mode == "public" and private):
        fail("Private inputs require Drive mode.")
    if private:
        label += "-PRIVATE" + ("-data" if data == "1" else "") + ("-audio" if audio == "1" else "")
    out = checked(root / "dist" / label, root)
    if out.exists():
        fail(f"Destination already exists: {out}")
    policy = checked(root / ".gitignore", root).read_text(encoding="utf-8").splitlines()
    rules = [line for line in policy if line and not line.startswith("#")]
    if not rules or rules[0] != "*":
        fail("Missing default-deny publication policy.")

    allowed = []
    categories = (
        r"(?:README\.md|LEEME\.md|\.gitignore|\.gitattributes)",
        r"docs/(?:[A-Z_]+\.md|assets/[a-z_]+\.svg|results/[a-z_]+\.csv)",
        r"lab4_solution/(?:README\.md|pyproject\.toml|uv\.lock|src/asistente_compras/[a-z_]+\.py|tests/test_[a-z_]+\.py)",
        r"lab4_solution/(?:notebooks_public/0[0-6]_[a-z_]+\.ipynb|scripts/empaquetar_(?:drive|compartir)\.sh)",
    )
    for rule in rules:
        if not rule.startswith("!"):
            continue
        if not re.fullmatch(r"!/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+/?", rule):
            fail(f"Expected an exact allowlist rule: {rule}")
        rel = rule[2:]
        if any(part in (".", "..", ".git") for part in rel.split("/")):
            fail(f"Unsafe allowlist rule: {rule}")
        if rel.endswith("/"):
            continue
        if not any(re.fullmatch(category, rel) for category in categories):
            fail(f"Unsupported publication path: {rel}")
        if not checked(root / rel, root).is_file():
            fail(f"Missing reviewed file: {rel}")
        allowed.append(rel)
    required = {".gitignore", "README.md", "LEEME.md", "lab4_solution/README.md",
                "lab4_solution/pyproject.toml", "lab4_solution/uv.lock",
                "lab4_solution/scripts/empaquetar_compartir.sh", "lab4_solution/scripts/empaquetar_drive.sh"}
    notebooks = [p for p in allowed if "/notebooks_public/" in p]
    if not required.issubset(allowed) or len(notebooks) != 7 or len(set(allowed)) != len(allowed):
        fail("Incomplete or duplicated publication allowlist.")

    # --no-index applies the same exclusion rules even to tracked files.
    git = ["git", "--git-dir=.git", "--work-tree=.", "--no-optional-locks"]
    probe = allowed + ["dist/" + label + "/lab4.zip"]
    raw = subprocess.run(git + ["check-ignore", "--no-index", "-v", "-z", "--stdin"],
                         cwd=root, input=b"".join(p.encode() + b"\0" for p in probe),
                         stdout=subprocess.PIPE, check=True).stdout.split(b"\0")[:-1]
    matches = {os.fsdecode(p): (os.fsdecode(f), os.fsdecode(rule))
               for f, _, rule, p in zip(raw[::4], raw[1::4], raw[2::4], raw[3::4])}
    if any(matches.get(p) != (".gitignore", "!/" + p) for p in allowed):
        fail("Git did not confirm each file-level exception.")
    if probe[-1] not in matches or matches[probe[-1]][1].startswith("!"):
        fail("The output directory must be ignored.")
    if subprocess.check_output(git + ["ls-files", "--", "dist"], cwd=root):
        fail("The output directory contains tracked paths.")

    files = {}
    for rel in allowed:
        if mode == "public":
            files[rel] = rel
        elif rel.startswith("lab4_solution/") and "/scripts/" not in rel:
            files[rel] = rel.replace("/notebooks_public/", "/notebooks/")
    if data == "1":
        inputs = ["manifest.json", "processed/catalog.parquet"]
        for split in ("dev", "test"):
            inputs.extend([f"processed/queries_{split}.parquet", f"eval/qrels_{split}.parquet"])
        inputs.extend(["index/multilingual-e5-base_titulo_marca-w200_229758f91a.npy",
                       "index/multilingual-e5-base_titulo_marca_bullets-w200_06dbc3e034.npy"])
        for rel in inputs:
            files["data/"+rel] = "data/"+rel
    if audio == "1":
        files["data/voz/referencia.wav"] = "data/voz/referencia.wav"
        for case, turns in enumerate((3, 4, 4, 4, 4, 6), 1):
            for turn in range(1, turns+1):
                rel = f"data/voz/escenarios/V{case}/turno_{turn:02}.wav"
                files[rel] = rel
    for rel in files:
        if not checked(root / rel, root).is_file():
            fail(f"Missing package input: {rel}")
    print(f"{'PRIVATE' if private else mode} package: {out} ({len(files)} files)")
    if dry == "1":
        for rel, target in files.items():
            print(f"{rel} -> lab4/{target}")
        return

    # Reserve a fresh destination and expose the completed tree only after copying.
    checked(out.parent, root).mkdir(exist_ok=True)
    checked(out, root).mkdir()
    building = out / ".building"
    dest = building / "lab4"
    dest.mkdir(parents=True)
    if mode == "drive":
        (dest / "data").mkdir()
    for rel, target in files.items():
        source = checked(root / rel, root)
        output = checked(dest / target, root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, output.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        shutil.copymode(source, output)
    if mode == "public":
        archive = building / "lab4.zip"
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
            for target in files.values():
                bundle.write(checked(dest / target, root), "lab4/" + target)
    dest.rename(out / "lab4")
    if mode == "public":
        archive.rename(out / "lab4.zip")
    building.rmdir()


try:
    main()
except (OSError, subprocess.CalledProcessError) as error:
    fail(str(error))
PY
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    package_project public "$0" para_compartir "$@"
fi
