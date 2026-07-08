import re, shutil
from pathlib import Path

DATA_DIR = Path(r"C:\Users\pedro\OneDrive - Universidade de Coimbra\Ambiente de Trabalho\tese\thesis-project\data")
ESP = 8
DRY_RUN = False   # set False to actually move

PATTERN = re.compile(
    r"^(?P<scenario>\d+)_(?P<location>(?:[A-G]-(?:[1-9]|1[0-4])|Z-0))_"
    r"(?P<user>\d+)_(?P<esp>\d+)_(?P<trial>\d+)_"
    r"(?P<timestamp>\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$", re.IGNORECASE)

target = DATA_DIR.parent / f"esp{ESP}-a-aguardar-instructions"
matched = [f for f in DATA_DIR.glob("*.csv")
           if (m := PATTERN.match(f.name)) and int(m.group("esp")) == ESP]

print(f"esp {ESP}: {len(matched)} files")
if DRY_RUN:
    print("[DRY RUN] nothing moved. Set DRY_RUN=False to move.")
else:
    target.mkdir(parents=True, exist_ok=True)
    for f in matched:
        shutil.move(str(f), str(target / f.name))
    print(f"moved {len(matched)} files to {target}")