#!/usr/bin/env bash
# Prepare MyDrive/lab4 with reviewed notebooks and explicitly selected local inputs.
set -euo pipefail

# Resolve the shared implementation without crossing a symlink or Windows junction.
for runtime in python python3; do
    if "$runtime" -I -B -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; then break; fi
    runtime=''
done
[[ -n $runtime ]] || { echo 'Python 3.8 or later is required.' >&2; exit 1; }
library=$("$runtime" -I -B - "$0" <<'PY'
import os
import stat
import sys
from pathlib import Path

script = Path(os.path.abspath(sys.argv[1]))
library = script.with_name("empaquetar_compartir.sh")
for path in (script.parents[2], script.parents[1], script.parent, script, library):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        sys.exit("Linked script path.")
sys.stdout.buffer.write(os.fsencode(library.as_posix()))
PY
)
source "$library"
package_project drive "$0" para_drive "$@"
