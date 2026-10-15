"""
Téléchargement de la MIT-BIH Arrhythmia Database (PhysioNet) via l'API wfdb.

Usage :
    python scripts/download_data.py                  # télécharge dans data/raw/mitdb
    python scripts/download_data.py --out data/raw   # dossier personnalisé
    python scripts/download_data.py --records 100 101 --out data/raw
"""
import argparse
import sys
from pathlib import Path

import wfdb

DB_NAME = "mitdb"
EXPECTED_FS = 360          # Hz
EXPECTED_N_RECORDS = 48    # la base en contient 48 ; 102/104/107/217 (rythme entraîné) seront exclus au prétraitement (norme AAMI)


def download(out_dir: Path, records=None) -> Path:
    target = out_dir / DB_NAME
    target.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = wfdb.get_record_list(DB_NAME)
    print(f"{len(records)} enregistrements à récupérer -> {target}")

    # wfdb ne retélécharge pas les fichiers déjà présents
    wfdb.dl_database(DB_NAME, dl_dir=str(target), records=records)
    return target


def verify(target: Path) -> bool:
    """Vérifie que chaque enregistrement est lisible, à 360 Hz, avec ses annotations."""
    ok = True
    heas = sorted(target.glob("*.hea"))
    print(f"{len(heas)} fichiers .hea trouvés")
    for hea in heas:
        name = hea.stem
        try:
            header = wfdb.rdheader(str(target / name))
            ann = wfdb.rdann(str(target / name), "atr")
        except Exception as e:  # noqa: BLE001
            print(f"  [ERREUR] {name} : {e}")
            ok = False
            continue
        if header.fs != EXPECTED_FS:
            print(f"  [WARN] {name} : fs={header.fs} (attendu {EXPECTED_FS})")
            ok = False
        print(f"  {name}: {header.sig_len} samples, {len(ann.sample)} annotations, leads={header.sig_name}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Télécharge la MIT-BIH Arrhythmia Database")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="dossier de sortie")
    parser.add_argument("--records", nargs="*", default=None, help="sous-ensemble d'enregistrements (ex: 100 101)")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    target = download(args.out, args.records)
    if not args.skip_verify and not verify(target):
        print("Vérification : des problèmes ont été détectés.", file=sys.stderr)
        sys.exit(1)
    print("Terminé.")


if __name__ == "__main__":
    main()