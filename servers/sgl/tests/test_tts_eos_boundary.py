from types import SimpleNamespace

import pytest
import torch
from zonos3_sgl.core import TTSReq


def _req(stop_id):
    return TTSReq(input_ids=torch.full((4, 4), 9, dtype=torch.int32), table_idx=0, cached_len=0, output_len=32, uid=0, sampling_params=SimpleNamespace(ignore_eos=False), cache_handle=None, n_codebooks=3, audio_pad_id=9, stop_id=stop_id)


@pytest.mark.parametrize('ahead', [0, 1])
@pytest.mark.parametrize('codebook', [0, 1, 2])
def test_eoa_boundary_tracks_processed_frames_not_submitted_work(ahead, codebook):
    req = _req(8)
    stop_at = 5 + codebook
    for step in range(stop_at + 1):
        req.total_generated = step + 1 + ahead
        frame = [1, 2, 3]
        if step == stop_at:
            frame[codebook] = 8
        assert not req.check_eos(frame)
    assert req.eos_frame == 5
    for _ in range(2):
        assert not req.check_eos([1, 2, 3])
    assert req.check_eos([1, 2, 3])


def test_audio_prefix_uses_audio_pad_id():
    assert not _req(8).has_audio_prefix
    rows = torch.full((4, 4), 9, dtype=torch.int32)
    rows[-1, 1] = 8
    req = TTSReq(input_ids=rows, table_idx=0, cached_len=0, output_len=32, uid=0, sampling_params=SimpleNamespace(ignore_eos=False), cache_handle=None, n_codebooks=3, audio_pad_id=9, stop_id=8)
    assert req.has_audio_prefix


def test_span_request_stops_on_span_end_and_ignores_eoa():
    eoa, span_end = 8, 10
    req = _req(span_end)
    for _ in range(6):
        assert not req.check_eos([eoa, eoa, eoa])
    assert req.eos_frame == -1
    assert not req.check_eos([1, span_end, 3])
    assert req.eos_frame == 5
    for _ in range(2):
        assert not req.check_eos([eoa, 2, 3])
    assert req.check_eos([1, 2, 3])
