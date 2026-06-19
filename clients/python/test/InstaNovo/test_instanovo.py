from __future__ import annotations

import numpy as np
import pytest
import requests
import tritonclient.grpc as grpcclient

from test.server_config import SERVER_GRPC, SERVER_HTTP


MODEL_NAMES = ("InstaNovo", "InstaNovoPlus", "InstaNovoWithRefinement")


def _inputs() -> list[grpcclient.InferInput]:
    mz_array = np.asarray(
        [
            [110.0, 180.0, 250.0, 400.0, 0.0],
            [120.0, 190.0, 275.0, 410.0, 600.0],
        ],
        dtype=np.float32,
    )
    intensity_array = np.asarray(
        [
            [0.1, 0.9, 0.4, 0.2, 0.0],
            [0.2, 0.8, 0.5, 0.3, 0.1],
        ],
        dtype=np.float32,
    )
    precursor_mz = np.asarray([[400.0], [500.0]], dtype=np.float32)
    precursor_charge = np.asarray([[2], [2]], dtype=np.int32)
    num_peaks = np.asarray([[4], [5]], dtype=np.int32)

    tensors = []
    for name, values, dtype in (
        ("mz_array", mz_array, "FP32"),
        ("intensity_array", intensity_array, "FP32"),
        ("precursor_mz", precursor_mz, "FP32"),
        ("precursor_charge", precursor_charge, "INT32"),
        ("num_peaks", num_peaks, "INT32"),
    ):
        tensor = grpcclient.InferInput(name, values.shape, dtype)
        tensor.set_data_from_numpy(values)
        tensors.append(tensor)
    return tensors


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_available_http(model_name: str) -> None:
    req = requests.get(f"{SERVER_HTTP}/v2/models/{model_name}", timeout=1)
    assert req.status_code == 200


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_available_grpc(model_name: str) -> None:
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(model_name)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_inference_contract(model_name: str) -> None:
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    result = triton_client.infer(model_name, inputs=_inputs())

    assert result.as_numpy("predictions").shape == (2, 1)
    assert result.as_numpy("predictions_tokenised").shape == (2, 1)
    assert result.as_numpy("token_log_probs").shape == (2, 1)
    assert result.as_numpy("log_probs").shape == (2, 1)
    assert result.as_numpy("confidence").shape == (2, 1)
    assert result.as_numpy("delta_mass_ppm").shape == (2, 1)
    assert result.as_numpy("beam_predictions").shape == (2, 5)
    assert result.as_numpy("beam_predictions_tokenised").shape == (2, 5)
    assert result.as_numpy("beam_token_log_probs").shape == (2, 5)
    assert result.as_numpy("beam_log_probs").shape == (2, 5)
    assert result.as_numpy("beam_confidence").shape == (2, 5)
    assert result.as_numpy("beam_delta_mass_ppm").shape == (2, 5)
    assert np.all(np.isfinite(result.as_numpy("confidence")))

    predictions = result.as_numpy("predictions")
    assert all(isinstance(value, bytes) for value in predictions.reshape(-1))

    if model_name == "InstaNovoWithRefinement":
        assert result.as_numpy("instanovo_predictions").shape == (2, 1)
        assert result.as_numpy("instanovo_log_probs").shape == (2, 1)
        assert result.as_numpy("instanovo_confidence").shape == (2, 1)
