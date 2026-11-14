"""The MLX server's generation scheduler: one worker thread owns every LM computation, one codec thread the streamed
audio decodes.

Each tick the worker runs the queued work items (codec decodes, speaker / reference encodes), admits waiting jobs into
free slots (``--batch``; a CFG job takes two), and advances every active job by one row (its branches' LM forwards in
one MLX evaluation). The per-job logic is ``zonos3.reference.synth.generate_rows`` / ``_laci`` (sampling, CFG
guidance, the EOS countdown, the probe readout to LACI and span-follow, LACI rollbacks re-prefilled in place) over the
MLX LM, after the ggml server's scheduler (servers/ggml/src/server/scheduler.cpp ``worker_loop``). Rows are sampled on
the device (``zonos3_mlx.sampling``); a job without a probe reader queues its next forwards on the lazy sample before
the host reads it.

Jobs report through their ``sink``: ``("row", row, eos_frame | None)`` per final row, then ``("done", laci_record |
None)``, or ``("error", message)``. A sink hands its audio decoding to ``codec_submit`` (the codec thread: one FIFO
for every job, so each job's chunks keep their order), and the backbone steps on while the codec runs; MLX >= 0.31.2
evaluates independent graphs from several threads, each on its own default stream.
"""

from __future__ import annotations

import collections
import logging
import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable

import mlx.core as mx
import numpy as np
import torch
from zonos3.reference.synth import ProbeState, Sampling, fade_out, steer_fn
from zonos3.tts.audio import output_pcm
from zonos3.tts.laci import DurationTarget, LaciConfig, LaciSession, RateBuckets
from zonos3.tts.plan import Branch
from zonos3.tts.probe import locate_text_span
from zonos3.tts.sampler import clamp_sampling
from zonos3.tts.span_follow import FollowConfig, SpanFollower

from zonos3_mlx.engine import Engine, Plan
from zonos3_mlx.models.lm import LMState
from zonos3_mlx.sampling import guided_logits_mx, noise_for, repetition_ids, sample_row_mx

logger = logging.getLogger("zonos3_mlx.scheduler")


@dataclass
class Job:
    plan: Plan
    sampling: Sampling
    sink: Callable[[tuple], None]
    stop_id: int
    laci_hold: bool  # LACI: a whole decode holds every row until the result is final
    aborted: bool = False

    @property
    def slots(self) -> int:
        return 2 if self.plan.twin is not None else 1


@dataclass
class _Active:
    """A running job's generation state (one attempt; a LACI rollback resets it in place)."""

    job: Job
    branches: list  # [(Branch, LMState)]
    outs: list
    generator: torch.Generator
    history: torch.Tensor
    has_audio_prefix: bool
    remain: int
    history_rows: int = 0
    rows: list = field(default_factory=list)
    eos_frame: int = -1
    countdown: int = -1
    step: int = 0
    probe: ProbeState | None = None
    observer: object | None = None
    follower: SpanFollower | None = None
    session: LaciSession | None = None
    prompt: torch.Tensor | None = None  # LACI: the prompt without its prefill rows
    delta: torch.Tensor | None = None  # the hidden delta the attempt steers with
    accents: object | None = None
    text_span: tuple[int, int] = (0, 0)


class Scheduler:
    def __init__(self, engine: Engine, batch: int = 4):
        self.engine = engine
        self.batch = max(1, int(batch))
        self.config = engine.config
        self._waiting: collections.deque[Job] = collections.deque()
        self._work: collections.deque[tuple[Callable, Future]] = collections.deque()
        self._active: list[_Active] = []
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._codec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zonos3-mlx-codec")
        self._running = False
        self.laci_config = LaciConfig()

    # ------------------------------------------------------------------------------------------------- control
    def start(self, warmup: bool | None = None) -> None:
        if warmup is None:
            warmup = not os.environ.get("ZONOS3_NO_WARMUP")
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="zonos3-mlx-worker", daemon=True)
        self._thread.start()
        if warmup:
            self.submit(self._warmup).result()
            self.codec_submit(self._warmup_codec).result()

    def stop(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30)
        self._codec.shutdown(wait=True, cancel_futures=True)

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Run ``fn`` on the worker thread (every MLX call but the streamed decodes); returns its future."""
        future: Future = Future()
        with self._cv:
            self._work.append((lambda: fn(*args, **kwargs), future))
            self._cv.notify_all()
        return future

    def codec_submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Run ``fn`` on the codec thread (the streamed audio decodes of every job, in submission order)."""
        return self._codec.submit(fn, *args, **kwargs)

    def enqueue(self, job: Job) -> Job:
        if job.slots > self.batch:
            raise ValueError("CFG needs --batch >= 2 (the guidance twin takes a second slot)")
        with self._cv:
            self._waiting.append(job)
            self._cv.notify_all()
        return job

    def abort(self, job: Job) -> None:
        job.aborted = True
        with self._cv:
            self._cv.notify_all()

    # ------------------------------------------------------------------------------------------------- worker
    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._running and not (self._work or self._waiting or self._active):
                    self._cv.wait(timeout=5.0)
                    if not (self._work or self._waiting or self._active):
                        self.engine.unload_idle_reference_encoder()
                if not self._running:
                    break
                work = list(self._work)
                self._work.clear()
            for fn, future in work:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(fn())
                    except BaseException as exc:  # noqa: BLE001 - delivered to the waiter
                        future.set_exception(exc)
            self._admit()
            for active in list(self._active):
                if active.job.aborted:
                    self._active.remove(active)
                    continue
                try:
                    self._tick(active)
                except Exception as exc:  # noqa: BLE001 - one job's failure must not stop the worker
                    logger.exception("generation failed")
                    self._active.remove(active)
                    active.job.sink(("error", str(exc)))
        for _fn, future in self._work:
            future.cancel()

    def _admit(self) -> None:
        with self._cv:
            used = sum(a.job.slots for a in self._active)
            while self._waiting:
                job = self._waiting[0]
                if job.aborted:
                    self._waiting.popleft()
                    continue
                if used + job.slots > self.batch:
                    break
                self._waiting.popleft()
                used += job.slots
                admitted = job
                try:
                    active = self._start(admitted)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("prefill failed")
                    admitted.sink(("error", str(exc)))
                    used -= admitted.slots
                    continue
                self._active.append(active)

    # ------------------------------------------------------------------------------------------------- jobs
    def _forward(self, a: _Active, branch: Branch, state: LMState, ids, frame: int | None, with_probe: bool):
        """One branch's forward graph (unevaluated; ``_run`` evaluates a tick's branches together)."""
        hidden = None
        if frame is not None and branch.steered:
            steer = steer_fn(a.job.plan, a.follower, a.delta)
            if steer is not None:
                hidden = steer(frame)
        return self.engine.lm.forward_lazy(
            ids,
            state,
            speaker_rows=branch.speaker_rows(),
            condition_rows=branch.condition_rows(),
            hidden_deltas=hidden,
            accents=a.accents,
            probe=a.probe.spec() if with_probe and a.probe is not None else None,
        )

    def _run(self, pending: list) -> list:
        """Evaluate a job's branch forwards (the CFG pair) in one MLX evaluation."""
        return self.engine.lm.finish(pending)

    def _start(self, job: Job) -> _Active:
        plan, c, engine = job.plan, self.config, self.engine
        accents = None
        if plan.accent is not None:
            accents = engine.accents.state(plan.accent[0], plan.accent[1])
        follower = None
        if plan.follow_tokens is not None and plan.schedule is not None:
            lead, lag = plan.follow_gate
            follower = SpanFollower(
                plan.follow_tokens, plan.schedule[:, 3:], FollowConfig(lead=float(lead), lag=float(lag))
            )
        text_span = (0, 0)
        if plan.laci is not None or follower is not None:
            text_span = locate_text_span(plan.cond.input_ids, c.n_codebooks)
        session = None
        if plan.laci is not None:
            session = self._laci_session(plan, follower, job.laci_hold)
        branches = [(plan.cond, LMState())] + ([] if plan.twin is None else [(plan.twin, LMState())])
        a = _Active(
            job=job,
            branches=branches,
            outs=[],
            generator=torch.Generator().manual_seed(int(plan.seed)),
            history=plan.cond.input_ids[:, : c.n_codebooks].clone(),
            has_audio_prefix=bool((plan.cond.input_ids[-1, : c.n_codebooks] != c.audio_pad_id).any()),
            remain=int(plan.max_tokens),
            follower=follower,
            session=session,
            delta=plan.hidden_delta,
            accents=accents,
            text_span=text_span,
        )
        if session is not None:
            template = plan.cond.input_ids
            a.prompt = template[: len(template) - session.n_prefill]
            a.probe = ProbeState(session.heads.heads, *text_span, mask=session.probe_mask())
            a.observer = session
        elif follower is not None:
            a.probe = ProbeState(engine.laci_heads[plan.follow_language].heads, *text_span)
            a.observer = follower
        a.outs = self._run([self._forward(a, b, state, b.input_ids, None, False) for b, state in branches])
        return a

    def _laci_session(self, plan: Plan, follower: SpanFollower | None, hold: bool) -> LaciSession:
        c = self.config
        names = list(c.condition_names)
        edges = c.condition_edges.get("speaking_rate")
        rates = (
            RateBuckets(
                names.index("speaking_rate"), tuple(edges), c.condition_bucket_counts[names.index("speaking_rate")]
            )
            if c.condition_count and "speaking_rate" in names and edges
            else None
        )
        take, duration, bucket, rate_deltas = plan.laci_take, None, None, None
        if take is not None:
            duration = DurationTarget(float(take.target_frames), adjust_rate=bool(take.adjust_rate))
            if take.rate_edges is not None:
                if rates is None:
                    raise ValueError("an extended speaking-rate scale needs the checkpoint's speaking-rate condition")
                rates = RateBuckets(
                    rates.index, tuple(float(e) for e in take.rate_edges), rates.n_buckets, int(take.rate_first)
                )
                bucket = int(take.rate_bucket)
                if not rates.first <= bucket <= rates.last:
                    raise ValueError(
                        f"speaking-rate bucket {bucket} is outside the scale [{rates.first}, {rates.last}]"
                    )
                rate_deltas = list(take.rate_deltas)
        width = c.n_codebooks + 1
        prefill = plan.prefill_rows if plan.prefill_rows is not None else torch.zeros((0, width), dtype=torch.int32)
        return LaciSession(
            heads=self.engine.laci_heads[plan.laci],
            cfg=self.laci_config,
            prefill_rows=prefill,
            seed=plan.seed,
            max_tokens=plan.max_tokens,
            n_codebooks=c.n_codebooks,
            hold=hold,
            duration=duration,
            rates=rates,
            condition_values=None if plan.cond.condition_values is None else list(plan.cond.condition_values),
            bucket=bucket,
            rate_deltas=rate_deltas,
            follower=follower,
            short_text=plan.short_text,
        )

    def _sample(self, a: _Active) -> mx.array:
        """The next row of ``a`` on the device (lazy, ``[n_codebooks + 1]`` int32): the CFG mix of the branches' last
        logits and the reference sampler on the job's generator (``zonos3_mlx.sampling``)."""
        job, plan, c = a.job, a.job.plan, self.config
        n_cb = c.n_codebooks
        logits = a.outs[0].logits[-1]
        if len(a.branches) > 1:
            logits = guided_logits_mx(logits, a.outs[1].logits[-1], plan.cfg_scale)
        total_generated = len(a.rows)
        s = clamp_sampling(job.sampling, n_codebooks=n_cb, codebook_size=c.codebook_size)
        repetition = repetition_ids(
            s,
            a.history,
            min(total_generated + a.history_rows, len(a.history)),
            n_codebooks=n_cb,
            codebook_size=c.codebook_size,
        )
        min_frames = job.sampling.min_frames
        stop_blocked = None
        if min_frames > 0:
            stop_blocked = np.array([total_generated < min_frames + cb for cb in range(n_cb)])
        width = max(job.stop_id, c.audio_pad_id) + 1
        return sample_row_mx(
            logits,
            mx.array(noise_for(a.generator, n_cb, width)),
            s,
            repetition,
            codebook_size=c.codebook_size,
            codebook_sizes=c.codebook_sizes,
            audio_pad_id=c.audio_pad_id,
            stop_id=job.stop_id,
            stop_blocked=stop_blocked,
            total_generated=total_generated,
            has_audio_prefix=a.has_audio_prefix,
            text_vocab=c.text_vocab,
            stop_bias=0.0 if a.session is None else a.session.stop_boost(),
        )

    def _tick(self, a: _Active) -> None:
        job, c = a.job, self.config
        n_cb = c.n_codebooks
        total_generated = len(a.rows)
        sampled = self._sample(a)
        ahead = None
        if a.probe is None and a.remain > 1:
            # Nothing the host reads steers the next row (no probe observer): queue its forwards on the lazy sample,
            # so the GPU runs them while the host reads the token and does the bookkeeping. Dropped if the row ends
            # the job (the EOS countdown).
            mx.async_eval(sampled)
            frame = total_generated + 1 + a.history_rows
            ahead = self.engine.lm.finish_async(
                [self._forward(a, b, state, sampled[None], frame, False) for b, state in a.branches]
            )
        token = torch.from_numpy(np.array(sampled))
        if a.step > 0 and a.probe is not None and a.observer is not None:
            a.probe.steps += 1
            if not a.probe.stop:
                pos, m = a.outs[0].probe[-1]
                a.observer.on_step(a.probe, a.probe.steps, int(pos), float(m))
            if a.probe.stop:
                a.remain = 0
        a.rows.append(token)
        a.history = torch.cat((a.history, token[None, :n_cb]), dim=0)
        a.remain -= 1
        finished = a.remain <= 0
        if a.eos_frame < 0:
            cols = [i for i, tok in enumerate(token[:n_cb].tolist()) if tok == job.stop_id]
            if cols:
                a.eos_frame = max(0, total_generated - max(cols))
                a.countdown = n_cb + 1
        if a.countdown > 0:
            a.countdown -= 1
            if a.countdown == 0:
                finished = True
        eos = a.eos_frame if a.eos_frame >= 0 else None
        if a.session is None:
            job.sink(("row", token, eos))
            if finished:
                self._active.remove(a)
                job.sink(("done", None))
                return
        else:
            session = a.session
            session.append(token, eos)
            if finished:
                cont = session.finish_attempt()
                rows = session.drain(final=cont is None)
                if cont is None:
                    final_eos = session.eos_frame()
                    for k, row in enumerate(rows):
                        job.sink(("row", row, final_eos if k == len(rows) - 1 else None))
                    self._active.remove(a)
                    job.sink(("done", session.record()))
                    return
                for row in rows:
                    job.sink(("row", row, None))
                self._resume(a, cont)
                return
            for row in session.drain():
                job.sink(("row", row, None))
        a.step += 1
        if ahead is not None:
            a.outs = ahead
            return
        frame = len(a.rows) + a.history_rows
        ids = token[None]
        a.outs = self._run([self._forward(a, b, state, ids, frame, i == 0) for i, (b, state) in enumerate(a.branches)])

    def _resume(self, a: _Active, cont) -> None:
        """A LACI rollback: re-prefill the prompt plus the kept rows (synth._laci, ggml laci_resume)."""
        branch, _ = a.branches[0]
        branch = replace(
            branch,
            input_ids=torch.cat((a.prompt, cont.rows.to(a.prompt.dtype)), dim=0),
            condition_values=cont.condition_values if cont.condition_values is not None else branch.condition_values,
        )
        if cont.swap_delta:
            a.delta = cont.steering_delta
        c = self.config
        a.branches = [(branch, LMState())]
        a.generator = torch.Generator().manual_seed(int(cont.seed if cont.seed is not None else a.job.plan.seed))
        a.history = branch.input_ids[:, : c.n_codebooks].clone()
        a.has_audio_prefix = bool((branch.input_ids[-1, : c.n_codebooks] != c.audio_pad_id).any())
        a.remain, a.history_rows = int(cont.max_tokens), int(cont.history_rows)
        a.rows, a.eos_frame, a.countdown, a.step = [], -1, -1, 0
        a.probe = ProbeState(a.session.heads.heads, *a.text_span, mask=a.session.probe_mask())
        a.outs = self._run([self._forward(a, branch, a.branches[0][1], branch.input_ids, None, False)])

    def _warmup(self) -> None:
        c = self.config
        ids = torch.full((12, c.n_codebooks + 1), c.audio_pad_id, dtype=torch.int32)
        ids[:, c.n_codebooks] = c.text_vocab
        state = LMState()
        self.engine.lm.forward(ids, state)
        self.engine.lm.forward(ids[:1], state)
        for dec in self.engine.decoders.values():
            dec.decode(np.zeros((4, c.n_codebooks), dtype=np.int64))

    def _warmup_codec(self) -> None:
        for dec in self.engine.decoders.values():
            dec.decode(np.zeros((4, self.config.n_codebooks), dtype=np.int64))

    # ------------------------------------------------------------------------------------------------- offline
    def synthesize(self, request: dict, speaker=None):
        """One whole request without HTTP (tools generate): plan, generate, decode at end of stream."""
        from zonos3_mlx.engine import plan_request

        plan = plan_request(self.engine, request, speaker)
        events: queue.Queue = queue.Queue()
        job = self.enqueue(
            Job(plan, Sampling.from_request(plan.request), events.put, self.config.eoa_id, laci_hold=True)
        )
        rows, eos, record = [], None, None
        while True:
            kind, *rest = events.get()
            if kind == "row":
                rows.append(rest[0])
                if rest[1] is not None:
                    eos = rest[1]
            elif kind == "done":
                record = rest[0]
                break
            else:
                raise RuntimeError(rest[0])
        dec = self.engine.decoder_of(plan.request.decoder)
        pcm = self.submit(
            dec.decode_rows,
            rows,
            prefill=None if plan.prefill_rows is None else plan.prefill_rows.numpy(),
            eos_frame=eos,
            context_codes=None if plan.context_codes is None else plan.context_codes.numpy(),
            audio_pad_id=self.config.audio_pad_id,
        ).result()
        pcm = fade_out(output_pcm(pcm, dec.output_gain(plan.request.headroom_db)), float(plan.request.fade_out_ms))
        return _Result(pcm.astype(np.float32), plan.seed, record, rows, eos, plan, job)


@dataclass
class _Result:
    pcm: np.ndarray
    seed: int
    laci: dict | None
    rows: list
    eos_frame: int | None
    plan: Plan
    job: Job
