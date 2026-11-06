// FuseAlign forced alignment in the browser: the WebGPU aligner and its host-side resampler and limits.
export { Aligner } from './aligner';
export { ALIGN_MAX_BYTES, ALIGN_MAX_SECONDS, resampleTo16k, type AlignedWord } from './host';
