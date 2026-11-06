"""Audio-file helpers shared by the model tests."""

from __future__ import annotations

import numpy as np
import soundfile as sf


def write_wav(path, samples: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> str:
    """Write mono ``samples`` to the WAV file ``path``; returns ``str(path)``."""
    sf.write(path, samples, sample_rate, subtype=subtype)
    return str(path)
