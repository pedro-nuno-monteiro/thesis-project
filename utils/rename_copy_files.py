from pathlib import Path
from typing import Dict
import re
import shutil
import argparse

USER_RE = re.compile(r'^user(\d{1,3})$', re.IGNORECASE)
POS_PREFIX_RE = re.compile(r'^position([a-ez]\d{2})$', re.IGNORECASE)
POS_BARE_RE = re.compile(r'^[a-ez]\d{2}$', re.IGNORECASE)
ESP_RE = re.compile(r'^esp0*([1-4])$', re.IGNORECASE)
INT_RE = re.compile(r'^\d{1,3}$')

def parse_tokens(tokens: list[str]) -> tuple[int, str, int]:
    userid = None
    location = None
    espid = None

    for raw in tokens:
        t = raw.strip()
        if not t:
            continue

        if (m := USER_RE.match(t)) and userid is None:
            userid = int(m.group(1))
            continue

        if (m := INT_RE.match(t)) and userid is None:
            userid = int(m.group(0))
            continue

        if (m := POS_PREFIX_RE.match(t)) and location is None:
            location = m.group(1).lower()
            continue

        if (m := POS_BARE_RE.match(t)) and location is None:
            location = m.group(0).lower()
            continue

        if (m := ESP_RE.match(t)) and espid is None:
            espid = int(m.group(1))
            continue

    if userid is None:
        userid = 0
    if location is None:
        location = "z00"
    if espid is None:
        raise ValueError("ESP id (esp1..esp4) not found")

    return userid, location, espid

def transform_name(name: str, existing_counts: Dict[str, int]) -> str:
    stem = name.rsplit('.', 1)[0]
    tokens = stem.split('_')
    userid, location, espid = parse_tokens(tokens)
    base_name = f"{userid:02d}-{location}-{espid:02d}"

    # Increment rep number for duplicates
    rep = existing_counts.get(base_name, 0) + 1
    existing_counts[base_name] = rep

    return f"{base_name}-{rep:02d}.csv"

def process(src: Path, dst: Path, dry_run: bool = False, recursive: bool = True) -> int:
    count: int = 0
    existing_counts: Dict[str, int] = {}
    files: list[Path] = list(src.rglob("*.csv")) if recursive else list(src.glob("*.csv"))

    dst.mkdir(parents=True, exist_ok=True)

    for f in files:
        try:
            new_name = transform_name(f.name, existing_counts)
            target = dst / new_name
            if dry_run:
                print(f"[DRY] {f}  ->  {target}")
            else:
                shutil.copy2(f, target)
                print(f"[OK ] {f}  ->  {target}")
            count += 1
        except Exception as e:
            print(f"[SKIP] {f}  (reason: {e})")
    return count

def main():
    ap = argparse.ArgumentParser(
        description="Copy CSV files with new names like 'userid-location-espid-rep.csv'."
    )
    ap.add_argument("source", type=Path, help="Source folder to scan for CSV files")
    ap.add_argument("dest", type=Path, help="Destination folder to copy files to")
    ap.add_argument("--no-recursive", action="store_true", help="Do not scan subfolders")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    args = ap.parse_args()

    try :
        if not args.source.is_dir():
            print(f"Error: Source path '{args.source}' is not a directory.")
            return
    except Exception as e:
        print(f"Error: Cannot access source path '{args.source}': {e}")
        return
    
    recursive = not args.no_recursive
    processed = process(args.source, args.dest, dry_run=args.dry_run, recursive=recursive)
    if args.dry_run:
        print(f"Planned {processed} file(s).")
    else:
        print(f"Copied {processed} file(s).")

if __name__ == "__main__":
    main()
