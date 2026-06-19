from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils


TOP_OUTPUTS = {
    "predictions",
    "predictions_tokenised",
    "token_log_probs",
    "log_probs",
    "confidence",
    "delta_mass_ppm",
    "instanovo_predictions",
    "instanovo_log_probs",
    "instanovo_confidence",
}


class TritonInstaNovoModel:
    """Triton Python backend wrapper for InstaNovo Koina adapters."""

    MODEL_KIND = "transformer"

    def initialize(self, args: dict[str, Any]) -> None:
        from instanovo.serving.koina import CombinedKoinaModel, DiffusionKoinaModel, ServingConfig, TransformerKoinaModel

        self.model_config = json.loads(args["model_config"])
        self.output_dtypes = {
            output["name"]: pb_utils.triton_string_to_numpy(output["data_type"])
            for output in self.model_config["output"]
        }
        device = os.getenv("INSTANOVO_DEVICE")
        num_beams = int(os.getenv("INSTANOVO_NUM_BEAMS", "5"))
        max_length = int(os.getenv("INSTANOVO_MAX_LENGTH", "40"))
        force_fp32 = os.getenv("INSTANOVO_FORCE_FP32", "0").lower() in {"1", "true", "yes"}

        if self.MODEL_KIND == "transformer":
            config = ServingConfig(
                model_id=os.getenv("INSTANOVO_MODEL_ID", "instanovo-v1.2.0"),
                device=device,
                num_beams=num_beams,
                max_length=max_length,
                force_fp32=force_fp32,
            )
            self.model = TransformerKoinaModel.from_pretrained(config)
        elif self.MODEL_KIND == "diffusion":
            config = ServingConfig(
                model_id=os.getenv("INSTANOVOPLUS_MODEL_ID", "instanovoplus-v1.1.0"),
                device=device,
                num_beams=num_beams,
                max_length=max_length,
                force_fp32=force_fp32,
            )
            self.model = DiffusionKoinaModel.from_pretrained(config)
        elif self.MODEL_KIND == "combined":
            transformer_config = ServingConfig(
                model_id=os.getenv("INSTANOVO_MODEL_ID", "instanovo-v1.2.0"),
                device=device,
                num_beams=num_beams,
                max_length=max_length,
                force_fp32=force_fp32,
            )
            diffusion_config = ServingConfig(
                model_id=os.getenv("INSTANOVOPLUS_MODEL_ID", "instanovoplus-v1.1.0"),
                device=device,
                num_beams=num_beams,
                max_length=max_length,
                force_fp32=force_fp32,
            )
            self.model = CombinedKoinaModel.from_pretrained(transformer_config, diffusion_config)
        else:
            raise ValueError(f"Unknown InstaNovo model kind: {self.MODEL_KIND}")

    def execute(self, requests: list[Any]) -> list[Any]:
        responses = []
        for request in requests:
            output = self.model.predict(
                mz_array=self._input(request, "mz_array"),
                intensity_array=self._input(request, "intensity_array"),
                precursor_mz=self._input(request, "precursor_mz"),
                precursor_charge=self._input(request, "precursor_charge"),
                num_peaks=self._input(request, "num_peaks"),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=self._output_tensors(output)))
        return responses

    @staticmethod
    def _input(request: Any, name: str) -> np.ndarray:
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            raise ValueError(f"Missing required input tensor: {name}")
        return tensor.as_numpy()

    def _output_tensors(self, output: dict[str, np.ndarray]) -> list[Any]:
        tensors = []
        for name, dtype in self.output_dtypes.items():
            values = output[name]
            if name in TOP_OUTPUTS:
                values = values.reshape((-1, 1))
            tensors.append(pb_utils.Tensor(name, values.astype(dtype)))
        return tensors

    def finalize(self) -> None:
        pass
