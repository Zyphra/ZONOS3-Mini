"""ZONOS3 inference server on Apple Silicon (MLX): the v2 HTTP API over the ggml server's GGUF files.

Threading rule: every MLX computation (LM, codec decoder, speaker encoder, reference encoder) runs on the scheduler's
worker thread, except the streamed requests' codec decodes, which run on its codec thread (``Scheduler.codec_submit``);
other threads submit work items to them and wait on their futures.
"""
