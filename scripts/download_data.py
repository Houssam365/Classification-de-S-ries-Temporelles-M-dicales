"""Download the MIT-BIH Arrhythmia Database from PhysioNet."""

from pathlib import Path
import argparse


def download_mitdb(data_dir: str | Path | None = None) -> Path:
    """Download and decode MIT-BIH into ``data_dir``.

    When ``data_dir`` is omitted, files are saved in the repository's ``data/``
    directory. Decoded NumPy files are written to ``data_dir/decoded/``.
    MIT-BIH is public and does not require PhysioNet credentials.
    """
    if data_dir is None:
        destination = Path(__file__).resolve().parents[1] / "data"
    else:
        destination = Path(data_dir).expanduser().resolve()

    destination.mkdir(parents=True, exist_ok=True)

    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "The 'wfdb' package is required. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    wfdb.dl_database(
        db_dir="mitdb",
        dl_dir=str(destination),
        keep_subdirs=False,
        overwrite=False,
    )

    records_file = destination / "RECORDS"
    if not records_file.is_file():
        raise FileNotFoundError(
            f"PhysioNet download completed without the expected file: {records_file}"
        )

    decode_mitdb(destination)
    return destination


def decode_mitdb(data_dir: str | Path | None = None) -> Path:
    """Decode WFDB signals and annotations into one compressed NumPy file per record."""
    if data_dir is None:
        source = Path(__file__).resolve().parents[1] / "data"
    else:
        source = Path(data_dir).expanduser().resolve()

    records_file = source / "RECORDS"
    if not records_file.is_file():
        raise FileNotFoundError(f"MIT-BIH RECORDS file not found: {records_file}")

    try:
        import numpy as np
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "The 'numpy' and 'wfdb' packages are required. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc

    output_dir = source / "decoded"
    output_dir.mkdir(parents=True, exist_ok=True)

    record_names = [
        line.strip() for line in records_file.read_text().splitlines() if line.strip()
    ]
    for record_name in record_names:
        record_path = str(source / record_name)
        record = wfdb.rdrecord(record_path)
        annotations = wfdb.rdann(record_path, "atr")

        if record.p_signal is None:
            raise ValueError(f"No physical signal data found for record {record_name}.")

        np.savez_compressed(
            output_dir / f"{record_name}.npz",
            signal=np.asarray(record.p_signal, dtype=np.float32),
            sampling_frequency=np.asarray(record.fs, dtype=np.float32),
            lead_names=np.asarray(record.sig_name, dtype=str),
            annotation_samples=np.asarray(annotations.sample, dtype=np.int64),
            annotation_symbols=np.asarray(annotations.symbol, dtype=str),
        )

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and decode MIT-BIH.")
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Decode records already present in data/ without downloading them.",
    )
    arguments = parser.parse_args()

    if arguments.decode_only:
        data_dir = Path(__file__).resolve().parents[1] / "data"
        decoded_dir = decode_mitdb(data_dir)
    else:
        data_dir = download_mitdb()
        decoded_dir = data_dir / "decoded"

    print(f"MIT-BIH data directory: {data_dir}")
    print(f"Decoded records saved to: {decoded_dir}")
