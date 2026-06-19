from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

from koina_triton import TritonInstaNovoModel  # noqa: E402


class TritonPythonModel(TritonInstaNovoModel):
    MODEL_KIND = "diffusion"
