"""AV-sync lip-sync step of the alignment worker (dub_align_worker.py; runs in .venv-dub-align).

AV-sync (https://github.com/Ryu1845/AV-sync, "Dynamic Temporal Alignment of Speech to Lips", Halperin et al.) is
imported from its checkout (--avsync-home, cloned at a pinned commit by tools/setup_dub_align_env.sh): the S3FD face
detector, SyncNet, the lip / audio embeddings, the banded dynamic programming and its smoothing filters. Its two entry
points are scripts that run on import, so the glue between those pieces is ported here, kept to
their constants: the face tracking and crop of run_pipeline.py (prepare) and the path post-processing of
align_audio.py main() / smooth_flow() (warp_frames). Alignment uses modality "va": the lips of the source video
against the dub's audio.

Differences from the scripts: face crops go straight from the decoder into SyncNet (the scripts re-encode them to
cropped.avi first), and the warp is applied with WSOLA (plan.wsola) to the full-rate 44.1 kHz dub, where align_audio.py
runs a phase vocoder (+ Griffin-Lim) on a 16 kHz, peak-normalized copy. WSOLA copies waveform segments, so it needs no
phase reconstruction: the phase vocoder alone lost 1.5-3 dB on warped dub speech and needed 10 Griffin-Lim iterations
to keep the level. Outside the face track the offset at its nearer end holds (plan.LipMap), so the whole dub stays
continuous.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from zonos3.dub import plan

# run_pipeline.py defaults
CROP_SCALE, MIN_TRACK, FACEDET_SCALE, MIN_CONF = 0.5, 30, 0.25, 0.9
IOU_THRES, NUM_FAIL, MIN_SIZE = 0.5, 3, 0.05
# align_audio.py defaults
MAX_DIFF_SMOOTH, GAUSSIAN_SIGMA = 0.05, 10


def models_present(home: str) -> bool:
    data = Path(home) / "data"
    return (data / "syncnet_v2.model").is_file() and (data / "sfd_face.pth").is_file()


def _ffmpeg(args: list[str], what: str) -> None:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if proc.returncode:
        raise ValueError(f"ffmpeg could not {what}: {proc.stderr[-400:]}")


class AvSync:
    def __init__(self, home: str, device: str):
        if not models_present(home):
            raise RuntimeError(f"AV-sync models missing under {home}/data (run tools/setup_dub_align_env.sh)")
        sys.path.insert(0, str(home))
        import avsnap_utils
        import s3fd
        import SyncNetInstance

        dev = torch.device(device)
        avsnap_utils.device = SyncNetInstance.device = dev
        avsnap_utils.amp = True  # align_audio.py default (--amp)
        self.utils = utils = avsnap_utils
        self.model = SyncNetInstance.SyncNetInstance()
        self.model.loadParameters(str(Path(home) / "data" / "syncnet_v2.model"))
        self.model.__S__.eval()
        self.detector = s3fd.S3FD(str(Path(home) / "data" / "sfd_face.pth"), dev)
        self.det_batch = 16 if dev.type == "cuda" else 1  # run_pipeline.py det_batch_size
        self.fps = utils.FPS

    # ---- face track + lip embeddings (run_pipeline.py) ---------------------------------------------------------

    def prepare(self, video: str, out: str) -> dict:
        """Face track of ``video`` (the first one, as run_pipeline.py) and its SyncNet lip embeddings, saved to ``out``."""
        utils = self.utils
        with tempfile.TemporaryDirectory(prefix="avsync-", dir=str(Path(out).parent)) as tmp:
            video25 = str(Path(tmp) / "video.avi")
            _ffmpeg(["-i", video, "-qscale:v", "4", "-r", str(self.fps), "-vf", "yadif=deint=interlaced", "-an", video25],
                    "read this video")
            fw, fh, _ = utils.probe_vid(video25)
            faces = self._detect(video25)
            n_frames = len(faces)
            tracks = self._track(faces)
            if not tracks:
                raise ValueError(f"no face track found (AV-sync needs one face visible for more than {MIN_TRACK} frames)")
            frames, boxes = tracks[0]
            emb = utils.vid2emb(self._crops(video25, frames, boxes, fw, fh), self.model, every_frame=True)
        first, last = int(frames[0]), int(frames[-1])
        torch.save({"emb": emb, "first_frame": first, "last_frame": last, "fps": self.fps}, out)
        return {"start": first / self.fps, "end": (last + 1) / self.fps, "track_frames": last - first + 1,
                "video_frames": n_frames, "tracks": len(tracks)}

    def _detect(self, video25: str) -> list[list]:
        dets, batch = [], []

        def flush():
            boxes = self.detector.detect_batch([im for _, im in batch], conf_th=MIN_CONF, scale=FACEDET_SCALE)
            for (num, _), frame_boxes in zip(batch, boxes):
                dets.append([[num, box[:4], box[4]] for box in frame_boxes.tolist()])
            batch.clear()

        for num, image in enumerate(self.utils.iter_frames(video25)):
            batch.append((num, image))
            if len(batch) == self.det_batch:
                flush()
        if batch:
            flush()
        return dets

    def _track(self, scenefaces: list[list]) -> list:
        """run_pipeline.py track_shot (its in-place list surgery included, so the same tracks come out)."""
        tracks = []
        while True:
            track = []
            for faces in scenefaces:
                for face in faces:
                    if track == []:
                        track.append(face)
                        faces.remove(face)
                    elif face[0] - track[-1][0] <= NUM_FAIL:
                        if _iou(face[1], track[-1][1]) > IOU_THRES:
                            track.append(face)
                            faces.remove(face)
                            continue
                    else:
                        break
            if track == []:
                break
            if len(track) > MIN_TRACK:
                framenum = torch.tensor([f[0] for f in track], dtype=torch.float64)
                bboxes = torch.tensor([f[1] for f in track], dtype=torch.float64)
                frame_i = torch.arange(track[0][0], track[-1][0] + 1)
                bboxes_i = self.utils.interp(frame_i.double(), framenum, bboxes)
                if torch.mean(bboxes_i[:, 3] - bboxes_i[:, 1]) > MIN_SIZE:
                    tracks.append([frame_i, bboxes_i])
        return tracks

    def _crops(self, video25: str, frames, boxes, fw: int, fh: int):
        """run_pipeline.py crop_video's 224 x 224 face crops, streamed."""
        utils = self.utils
        size, cx, cy = [], [], []
        for det in boxes.tolist():
            size.append(((det[3] - det[1]) * fw + (det[2] - det[0]) * fh) / 4)
            cx.append((det[1] + det[3]) * fw / 2)
            cy.append((det[0] + det[2]) * fh / 2)
        size = utils.medfilt(torch.tensor(size, dtype=torch.float64), kernel_size=5).tolist()
        cx = utils.medfilt(torch.tensor(cx, dtype=torch.float64), kernel_size=5).tolist()
        cy = utils.medfilt(torch.tensor(cy, dtype=torch.float64), kernel_size=7).tolist()
        cs = CROP_SCALE
        video = itertools.islice(utils.iter_frames(video25), int(frames[0]), None)
        for bs, mx0, my0, frame in zip(size, cx, cy, video):
            bsi = int(bs * (1 + 2 * cs))
            my, mx = my0 + bsi, mx0 + bsi
            ya, ystart, ystop, yb = _pad_span(int(my - bs), int(my + bs * (1 + 2 * cs)), fh, bsi)
            xa, xstart, xstop, xb = _pad_span(int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs)), fw, bsi)
            face = F.pad(frame[ystart:ystop, xstart:xstop].permute(2, 0, 1), (xa, xb, ya, yb))
            resized = F.interpolate(face[None].float(), size=(224, 224), mode="bilinear", antialias=True)[0]
            yield resized.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)

    # ---- alignment + warp (align_audio.py) ---------------------------------------------------------------------

    def sync(self, lips: str, audio: str, out: str, max_rate: float | None = None, max_shift_s: float | None = None) -> dict:
        """Warp the dub ``audio`` onto the lips of ``lips`` (prepare's file); writes the warped dub to ``out``. With
        ``max_rate`` and ``max_shift_s`` the warp is bounded first (plan.guard_lip_frames)."""
        utils = self.utils
        track = torch.load(lips, weights_only=True)
        start, end = track["first_frame"] / track["fps"], (track["last_frame"] + 1) / track["fps"]
        wav, sr = sf.read(audio, dtype="float64", always_2d=True)
        wav = wav.mean(axis=1)
        vid_emb = track["emb"]
        aud_emb = self._span_embedding(audio, start, end)
        dp = utils.dynamic_programming(vid_emb, aud_emb)  # modality "va": lips of the video, audio of the dub
        path, cost = utils.find_path(dp)
        src = self.warp_frames(path)
        if max_rate is not None and max_shift_s is not None:
            src = torch.from_numpy(plan.guard_lip_frames(src.numpy(), utils.WINDOWS_PER_SECOND, max_rate, max_shift_s))
        lip_map = plan.LipMap.from_frames(start, utils.WINDOWS_PER_SECOND, src.tolist())

        # output sample t plays the dub at lip_map.src_time(t / sr)
        n_out = int(round(lip_map.out_time(len(wav) / sr) * sr))
        y = plan.wsola(wav, lambda t: lip_map.src_time(t / sr) * sr, n_out)
        sf.write(out, y.astype(np.float32), sr, subtype="FLOAT")

        synced_emb = self._span_embedding(out, start, end)
        return {
            "start": start,
            "end": end,
            "rate": utils.WINDOWS_PER_SECOND,
            "src_frames": src.tolist(),
            "dp_cost": float(cost),
            "sync_distance_before": self._diag_distance(vid_emb, aud_emb),
            "sync_distance_after": self._diag_distance(vid_emb, synced_emb),
        }

    def _span_embedding(self, audio: str, start: float, end: float) -> torch.Tensor:
        """SyncNet audio embeddings (100 / s) of [start, end) of ``audio``: run_pipeline.py's 16 kHz pcm_s16le crop."""
        with tempfile.TemporaryDirectory(prefix="avsync-") as tmp:
            wav16 = str(Path(tmp) / "audio.wav")
            _ffmpeg(["-i", audio, "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", str(self.utils.SR),
                     "-af", "apad", "-ss", "%.3f" % start, "-to", "%.3f" % end, wav16], "crop the dub")
            aud = self.utils.read_aud(wav16, verbose=False)
        return self.utils.aud2emb(aud, self.model, True)[1]

    @staticmethod
    def _diag_distance(vid_emb: torch.Tensor, aud_emb: torch.Tensor) -> float:
        """Mean SyncNet distance of the lips and the audio at equal times (the dp's data cost, unwarped)."""
        n = min(len(vid_emb), len(aud_emb))
        return float(((vid_emb[:n].double() - aud_emb[:n].double()) ** 2).mean())

    def warp_frames(self, path: torch.Tensor) -> torch.Tensor:
        """align_audio.py main() + smooth_flow(): dp path (video row, audio column) -> for every output STFT frame of
        the track span (16 kHz, hop 256) the source frame of the dub, smoothed."""
        utils = self.utils
        path = path.tolist()
        # every audio frame should be associated with a single video frame
        dp_path = [path[0][1]]
        for i in range(1, len(path)):
            if path[i][0] > path[i - 1][0]:
                dp_path.append(path[i][1])
        dp_path = [0, 1] + [p + 2 for p in dp_path]  # embeddings refer to the middle frame
        dp_path = torch.tensor(dp_path, dtype=torch.float64)
        new_size = utils.WINDOWS_PER_FRAME / 4 * dp_path.numel()
        flow = utils.resize_vec(dp_path - torch.arange(dp_path.numel()), new_size)

        factor, max_diff, sigma, smoothed = utils.WINDOWS_PER_SECOND, 0, 1, flow
        while max_diff < MAX_DIFF_SMOOTH and sigma < 10000:
            sigma *= 1.5
            smoothed = utils.stabilizeSequence(flow, sigma)
            max_diff = (smoothed / factor - flow / factor).abs().max().item()
        smoothed = utils.gaussian_filter(smoothed, GAUSSIAN_SIGMA)
        return smoothed + torch.arange(smoothed.numel())


def _iou(box_a, box_b) -> float:
    """run_pipeline.py bb_intersection_over_union."""
    x_a, y_a = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x_b, y_b = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter)


def _pad_span(start, stop, size, pad):
    """run_pipeline.py pad_span: slice [start, stop) of an axis of length size zero padded by pad on both sides, in
    padded coordinates -> (zeros before, frame start, frame stop, zeros after)."""
    start, stop = (min(max(c, 0), size + 2 * pad) - pad for c in (start, stop))
    lo, hi = min(max(start, 0), size), min(max(stop, 0), size)
    return max(0, min(stop, 0) - start), lo, max(lo, hi), max(0, stop - max(start, size))
