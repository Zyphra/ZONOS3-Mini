// WebGPU device bring-up, kernel pipelines, dummy bindings and readback.
import commonSrc from './kernels/common.wgsl?raw';

const kernelSources = import.meta.glob('./kernels/*.wgsl', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

export const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export async function initGpu(onLost?: (message: string) => void): Promise<GPUDevice> {
  if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found');
  // The speaker encoder's stage-1 activations ([T,80,128] f32: 245.7 MB at 60 s) need the adapter's full limits.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then(info => {
    if (info.reason !== 'destroyed') onLost?.('GPU device lost: ' + info.message + '; reload the page');
  });
  device.addEventListener('uncapturederror', e => console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message));
  return device;
}

interface BindingInfo { binding: number; name: string; writable: boolean; uniform: boolean }
interface FieldInfo { name: string; type: 'u32' | 'f32' | 'i32' }
export interface Kernel {
  name: string;
  pipeline: GPUComputePipeline;
  bindings: BindingInfo[];
  fields: FieldInfo[];
}

const H8_ROWS = /^gemm_h8_r(\d+)_(\d+)x(\d+)$/;
const H8_KS = /^gemm_h8_ks(\d+)$/;

/** A templated kernel name: its source file and the module constants prepended to it. */
function kernelTemplate(name: string): { base: string; header: string } {
  let m = H8_ROWS.exec(name);
  if (m) return { base: 'gemm_h8', header: `const R = ${m[1]}u;\nconst LX = ${m[2]}u;\nconst LY = ${m[3]}u;\n` };
  m = H8_KS.exec(name);
  if (m) return { base: 'gemm_h8_ks', header: `const S = ${m[1]}u;\n` };
  return { base: name, header: '' };
}

/** Dispatch grid of a gemm_h8 variant over T rows x O outputs (8 outputs per thread). */
export function h8Grid(name: string, T: number, O: number): [number, number, number] {
  const OG = Math.ceil(O / 8);
  let m = H8_ROWS.exec(name);
  if (m) {
    const [r, lx, ly] = [Number(m[1]), Number(m[2]), Number(m[3])];
    return [Math.ceil(OG / lx), Math.ceil(Math.ceil(T / r) / ly), 1];
  }
  m = H8_KS.exec(name);
  if (m) return [Math.ceil(OG / 8), 1, Math.ceil(T / 4)];
  throw new Error(`h8Grid: ${name} is not a gemm_h8 kernel`);
}

export class Gpu {
  private kernels = new Map<string, Kernel>();
  private dummies = new Map<number, GPUBuffer>();
  private staging = new Map<number, GPUBuffer[]>();

  constructor(public device: GPUDevice) {}

  kernel(name: string): Kernel {
    let k = this.kernels.get(name);
    if (k) return k;
    const { base, header } = kernelTemplate(name);
    const src = kernelSources[`./kernels/${base}.wgsl`];
    if (src === undefined) throw new Error(`unknown kernel ${name}`);
    const code = commonSrc + '\n' + header + src;
    const module = this.device.createShaderModule({ code, label: name });
    const pipeline = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: name });
    const bindings: BindingInfo[] = [];
    for (const m of src.matchAll(/@binding\((\d+)\)\s+var(<[^>]+>)?\s+(\w+)/g)) {
      const q = m[2] ?? '';
      bindings.push({ binding: Number(m[1]), name: m[3], writable: q.includes('read_write'), uniform: q.includes('uniform') });
    }
    const fields: FieldInfo[] = [];
    const sm = /struct P\s*\{([^}]*)\}/.exec(src);
    if (sm) for (const f of sm[1].matchAll(/(\w+)\s*:\s*(u32|f32|i32)/g)) fields.push({ name: f[1], type: f[2] as FieldInfo['type'] });
    k = { name, pipeline, bindings, fields };
    this.kernels.set(name, k);
    return k;
  }

  /** Never-written 256-byte buffer for an unused optional binding slot (one per slot index). */
  dummy(slot: number): GPUBuffer {
    let b = this.dummies.get(slot);
    if (!b) {
      b = this.device.createBuffer({ size: 256, usage: STORAGE, label: `dummy${slot}` });
      this.dummies.set(slot, b);
    }
    return b;
  }

  private dummyU: GPUBuffer | null = null;
  /** Zero uniform for kernels whose `dyn` block is unused in a given dispatch. */
  dummyUniform(): GPUBuffer {
    if (!this.dummyU) this.dummyU = this.uniform(16, 'dyn0');
    return this.dummyU;
  }

  buffer(bytes: number, label?: string): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: STORAGE, label });
  }

  uniform(bytes = 16, label?: string): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
  }

  /** queue.writeBuffer for any typed array (sizes rounded up to 4 bytes). */
  write(buf: GPUBuffer, offset: number, data: ArrayBufferView): void {
    this.device.queue.writeBuffer(buf, offset, data.buffer as ArrayBuffer, data.byteOffset, Math.ceil(data.byteLength / 4) * 4);
  }

  upload(data: ArrayBufferView, label?: string): GPUBuffer {
    const b = this.buffer(data.byteLength, label);
    this.write(b, 0, data);
    return b;
  }

  private takeStaging(size: number): GPUBuffer {
    const list = this.staging.get(size);
    if (list && list.length) return list.pop()!;
    return this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  /**
   * Submit `encodeFn`'s commands, then copy `reads` into staging buffers and return them as Float32Arrays. Every
   * staging buffer maps concurrently; the queue submission happens before the first await.
   */
  async submitRead(encodeFn: (enc: GPUCommandEncoder) => void,
                   reads: { buf: GPUBuffer; offset: number; bytes: number }[] = []): Promise<Float32Array[]> {
    const enc = this.device.createCommandEncoder();
    encodeFn(enc);
    const st = reads.map(r => {
      const size = Math.ceil(r.bytes / 4) * 4;
      const s = this.takeStaging(Math.max(16, Math.ceil(size / 16) * 16));
      enc.copyBufferToBuffer(r.buf, r.offset, s, 0, size);
      return s;
    });
    this.device.queue.submit([enc.finish()]);
    if (!st.length) {
      await this.device.queue.onSubmittedWorkDone();
      return [];
    }
    await Promise.all(st.map(s => s.mapAsync(GPUMapMode.READ)));
    const out: Float32Array[] = [];
    for (let i = 0; i < st.length; i++) {
      out.push(new Float32Array(st[i].getMappedRange(0, Math.ceil(reads[i].bytes / 4) * 4).slice(0)));
      st[i].unmap();
      const list = this.staging.get(st[i].size) ?? [];
      list.push(st[i]);
      this.staging.set(st[i].size, list);
    }
    return out;
  }

  async read(buf: GPUBuffer, offset = 0, bytes = buf.size): Promise<Float32Array> {
    return (await this.submitRead(() => {}, [{ buf, offset, bytes }]))[0];
  }
}
