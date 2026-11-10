/** `/v2/capabilities.decoders` of the offline backend: the release codec's two decoder heads, the default first. */
import type { DecoderInfo } from '../api';

const CODEC = 'opus_dual_3n_600k_sched600k_r4096_step600000';
const HEADS = 'hf://Zyphra/ZONOS3_Mini_release_ckpts/autoencoder/heads';

export const FIXTURE_DECODERS: DecoderInfo[] = [
  {
    id: 'keep2_plain',
    kind: 'istft_mr',
    label: 'Multi-resolution iSTFT head + linear STFT loss, KEEP-2 pruned decoder (istft_head_a_20261002/full-unfreezeall-mr-drop30-4x10-50k, final step 50,000); causal, fp32',
    checkpoint: `${HEADS}/keep2_plain/step_0050000.pt`,
    expect_step: 50000,
    headroom_db: 3,
    causal: true,
    codec: CODEC,
  },
  {
    id: 'full_mr',
    kind: 'istft_mr',
    label: 'Multi-resolution iSTFT head + linear STFT loss, decoder.0-8 unfrozen (istft_head_a_20261002/full-unfreezeall-mr-50k, final step 50,000; release copy = step + trainable state); causal, fp32',
    checkpoint: `${HEADS}/full_mr/step_0050000.pt`,
    expect_step: 50000,
    headroom_db: 3,
    causal: true,
    codec: CODEC,
  },
];
