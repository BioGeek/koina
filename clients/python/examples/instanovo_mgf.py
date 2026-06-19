#!/usr/bin/env python3
"""Run InstaNovo Koina predictions for spectra in an MGF file."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from queue import Queue
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
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of in-flight Triton requests. Values above 1 use async gRPC inference.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional client-side timeout in seconds for each Triton request.",
    )
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


def make_inputs(batch: dict[str, np.ndarray]) -> list[grpcclient.InferInput]:
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
    return inputs


def infer(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    batch: dict[str, np.ndarray],
    timeout: float | None,
) -> grpcclient.InferResult:
    return client.infer(model_name, inputs=make_inputs(batch), client_timeout=timeout)


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


def iter_batches(spectra: list[dict[str, Any]], batch_size: int) -> list[tuple[int, list[dict[str, Any]]]]:
    return [
        (batch_index, spectra[start : start + batch_size])
        for batch_index, start in enumerate(range(0, len(spectra), batch_size))
    ]


def report_progress(done: int, total: int) -> None:
    if total < 2:
        return
    print(f"\rCompleted {done}/{total} batches", end="", file=sys.stderr, flush=True)
    if done == total:
        print(file=sys.stderr)


def predict_sync(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    batches: list[tuple[int, list[dict[str, Any]]]],
    output_names: list[str],
    timeout: float | None,
) -> list[dict[str, Any]]:
    output_rows = []
    for done, (_, batch_rows) in enumerate(batches, start=1):
        result = infer(client, model_name, make_batch(batch_rows), timeout)
        output_rows.extend(predictions_to_rows(batch_rows, result, output_names))
        report_progress(done, len(batches))
    return output_rows


def predict_async(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    batches: list[tuple[int, list[dict[str, Any]]]],
    output_names: list[str],
    concurrency: int,
    timeout: float | None,
) -> list[dict[str, Any]]:
    completed: Queue[tuple[int, grpcclient.InferResult | None, Exception | None]] = Queue()
    rows_by_batch: dict[int, list[dict[str, Any]]] = {}
    next_batch = 0
    in_flight = 0
    done = 0

    def submit(batch_index: int, batch_rows: list[dict[str, Any]]) -> None:
        def callback(result: grpcclient.InferResult | None, error: Exception | None) -> None:
            completed.put((batch_index, result, error))

        client.async_infer(
            model_name=model_name,
            request_id=str(batch_index),
            inputs=make_inputs(make_batch(batch_rows)),
            callback=callback,
            client_timeout=timeout,
        )

    while next_batch < len(batches) and in_flight < concurrency:
        submit(*batches[next_batch])
        next_batch += 1
        in_flight += 1

    while in_flight:
        batch_index, result, error = completed.get()
        in_flight -= 1
        if error is not None:
            raise RuntimeError(f"Inference failed for batch {batch_index}: {error}") from error
        if result is None:
            raise RuntimeError(f"Inference failed for batch {batch_index}: Triton returned no result.")

        rows_by_batch[batch_index] = predictions_to_rows(batches[batch_index][1], result, output_names)
        done += 1
        report_progress(done, len(batches))

        while next_batch < len(batches) and in_flight < concurrency:
            submit(*batches[next_batch])
            next_batch += 1
            in_flight += 1

    output_rows = []
    for batch_index in sorted(rows_by_batch):
        output_rows.extend(rows_by_batch[batch_index])
    return output_rows


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1.")
    if not args.mgf.exists():
        raise SystemExit(f"Input MGF not found: {args.mgf}")

    import_dependencies()

    spectra = read_mgf(args.mgf)
    client = grpcclient.InferenceServerClient(url=args.server, ssl=args.ssl)
    if not client.is_model_ready(args.model):
        raise RuntimeError(f"Model {args.model} is not ready on {args.server}.")
    output_names = [output.name for output in client.get_model_metadata(args.model).outputs]

    batches = iter_batches(spectra, args.batch_size)
    if args.concurrency == 1:
        output_rows = predict_sync(client, args.model, batches, output_names, args.timeout)
    else:
        output_rows = predict_async(client, args.model, batches, output_names, args.concurrency, args.timeout)

    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    print(f"Wrote {len(output_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
