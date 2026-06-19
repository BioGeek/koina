#!/usr/bin/env python3
"""Run InstaNovo Koina predictions for spectra in an MGF file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

np: Any = None
pd: Any = None
mgf: Any = None
grpcclient: Any = None


def import_dependencies() -> None:
    global grpcclient, mgf, np, pd
    try:
        import numpy as _np
        import pandas as _pd
        from pyteomics import mgf as _mgf
        import tritonclient.grpc as _grpcclient
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing dependency. Install the example requirements with "
            '`python3 -m pip install "tritonclient[grpc]" pyteomics pandas numpy`.'
        ) from error

    np = _np
    pd = _pd
    mgf = _mgf
    grpcclient = _grpcclient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mgf", type=Path, help="Input MGF file.")
    parser.add_argument(
        "-m",
        "--model",
        default="InstaNovoWithRefinement",
        choices=("InstaNovo", "InstaNovoPlus", "InstaNovoWithRefinement"),
        help="Koina model endpoint to call.",
    )
    parser.add_argument("-s", "--server", default="localhost:8500", help="Triton gRPC server, for example localhost:8500.")
    parser.add_argument("--ssl", action="store_true", help="Use TLS for the gRPC connection.")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Number of spectra per inference request.")
    parser.add_argument("-o", "--output", type=Path, default=Path("instanovo_predictions.csv"), help="Output CSV path.")
    return parser.parse_args()


def charge_to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return charge_to_int(value[0]) if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+", str(value))
        return int(match.group(0)) if match else 0


def precursor_mz_to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else 0.0
    return float(value)


def read_mgf(path: Path) -> list[dict[str, Any]]:
    rows = []
    with mgf.read(str(path)) as reader:
        for spectrum_index, spectrum in enumerate(reader):
            params = spectrum.get("params", {})
            rows.append(
                {
                    "spectrum_index": spectrum_index,
                    "title": params.get("title", str(spectrum_index)),
                    "precursor_mz": precursor_mz_to_float(params.get("pepmass")),
                    "precursor_charge": charge_to_int(params.get("charge")),
                    "mz_array": np.asarray(spectrum["m/z array"], dtype=np.float32),
                    "intensity_array": np.asarray(spectrum["intensity array"], dtype=np.float32),
                }
            )
    if not rows:
        raise ValueError(f"No spectra found in {path}.")
    return rows


def make_batch(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    max_peaks = max(len(row["mz_array"]) for row in rows)
    mz_array = np.zeros((len(rows), max_peaks), dtype=np.float32)
    intensity_array = np.zeros((len(rows), max_peaks), dtype=np.float32)
    precursor_mz = np.zeros((len(rows), 1), dtype=np.float32)
    precursor_charge = np.zeros((len(rows), 1), dtype=np.int32)
    num_peaks = np.zeros((len(rows), 1), dtype=np.int32)

    for row_idx, row in enumerate(rows):
        n_peaks = len(row["mz_array"])
        mz_array[row_idx, :n_peaks] = row["mz_array"]
        intensity_array[row_idx, :n_peaks] = row["intensity_array"]
        precursor_mz[row_idx, 0] = row["precursor_mz"]
        precursor_charge[row_idx, 0] = row["precursor_charge"]
        num_peaks[row_idx, 0] = n_peaks

    return {
        "mz_array": mz_array,
        "intensity_array": intensity_array,
        "precursor_mz": precursor_mz,
        "precursor_charge": precursor_charge,
        "num_peaks": num_peaks,
    }


def infer(client: grpcclient.InferenceServerClient, model_name: str, batch: dict[str, np.ndarray]) -> grpcclient.InferResult:
    inputs = []
    for name, values, dtype in (
        ("mz_array", batch["mz_array"], "FP32"),
        ("intensity_array", batch["intensity_array"], "FP32"),
        ("precursor_mz", batch["precursor_mz"], "FP32"),
        ("precursor_charge", batch["precursor_charge"], "INT32"),
        ("num_peaks", batch["num_peaks"], "INT32"),
    ):
        tensor = grpcclient.InferInput(name, values.shape, dtype)
        tensor.set_data_from_numpy(values)
        inputs.append(tensor)
    return client.infer(model_name, inputs=inputs)


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def output_to_columns(name: str, values: np.ndarray, row_idx: int) -> dict[str, Any]:
    row_values = values[row_idx]
    if row_values.shape == ():
        return {name: decode_value(row_values)}
    if row_values.size == 1:
        return {name: decode_value(row_values.reshape(-1)[0])}
    return {f"{name}_{idx + 1}": decode_value(value) for idx, value in enumerate(row_values.reshape(-1))}


def predictions_to_rows(batch_rows: list[dict[str, Any]], result: grpcclient.InferResult, output_names: list[str]) -> list[dict[str, Any]]:
    outputs = {name: result.as_numpy(name) for name in output_names}
    rows = []
    for row_idx, spectrum in enumerate(batch_rows):
        output_row = {
            "spectrum_index": spectrum["spectrum_index"],
            "title": spectrum["title"],
            "precursor_mz": spectrum["precursor_mz"],
            "precursor_charge": spectrum["precursor_charge"],
            "num_peaks": len(spectrum["mz_array"]),
        }
        for name, values in outputs.items():
            output_row.update(output_to_columns(name, values, row_idx))
        rows.append(output_row)
    return rows


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if not args.mgf.exists():
        raise SystemExit(f"Input MGF not found: {args.mgf}")

    import_dependencies()

    spectra = read_mgf(args.mgf)
    client = grpcclient.InferenceServerClient(url=args.server, ssl=args.ssl)
    if not client.is_model_ready(args.model):
        raise RuntimeError(f"Model {args.model} is not ready on {args.server}.")
    output_names = [output.name for output in client.get_model_metadata(args.model).outputs]

    output_rows = []
    for start in range(0, len(spectra), args.batch_size):
        batch_rows = spectra[start : start + args.batch_size]
        result = infer(client, args.model, make_batch(batch_rows))
        output_rows.extend(predictions_to_rows(batch_rows, result, output_names))

    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    print(f"Wrote {len(output_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
