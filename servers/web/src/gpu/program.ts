// Pre-recorded GPU programs: dispatches with static bind groups plus buffer copies/clears.
import type { Gpu } from './device';

export type Step =
  | { k: 'd'; pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; wg: [number, number, number]; label: string }
  | { k: 'c'; src: GPUBuffer; srcOff: number; dst: GPUBuffer; dstOff: number; size: number }
  | { k: 'clear'; buf: GPUBuffer; off: number; size: number }
  | { k: 'split' };

export function encodeSteps(enc: GPUCommandEncoder, steps: Step[]): void {
  let pass: GPUComputePassEncoder | null = null;
  for (const s of steps) {
    if (s.k === 'd') {
      if (!pass) pass = enc.beginComputePass();
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bindGroup);
      pass.dispatchWorkgroups(s.wg[0], s.wg[1], s.wg[2]);
      continue;
    }
    if (pass) { pass.end(); pass = null; }
    if (s.k === 'c') enc.copyBufferToBuffer(s.src, s.srcOff, s.dst, s.dstOff, s.size);
    else if (s.k === 'clear') enc.clearBuffer(s.buf, s.off, s.size);
  }
  if (pass) pass.end();
}

export class Program {
  private segments: Step[][] | null = null;
  constructor(public steps: Step[], private owned: GPUBuffer[]) {}
  encode(enc: GPUCommandEncoder): void { encodeSteps(enc, this.steps); }

  /** Submits each Builder.split() segment as its own command buffer, so no single submission runs long enough
   *  to trip a GPU driver watchdog (on Linux a reset job surfaces as WebGPU device loss). `prelude` is encoded
   *  before the first segment; `reads` are copied out after the last. */
  run(gpu: Gpu, reads: Parameters<Gpu['submitRead']>[1] = [], prelude?: (enc: GPUCommandEncoder) => void): Promise<Float32Array[]> {
    if (!this.segments) {
      this.segments = [[]];
      for (const s of this.steps) {
        if (s.k === 'split') this.segments.push([]);
        else this.segments[this.segments.length - 1].push(s);
      }
    }
    const segs = this.segments;
    for (let i = 0; i < segs.length - 1; i++) {
      const enc = gpu.device.createCommandEncoder();
      if (i === 0) prelude?.(enc);
      encodeSteps(enc, segs[i]);
      gpu.device.queue.submit([enc.finish()]);
    }
    return gpu.submitRead(enc => {
      if (segs.length === 1) prelude?.(enc);
      encodeSteps(enc, segs[segs.length - 1]);
    }, reads);
  }

  /** Frees the params uniforms and every scratch buffer allocated through the Builder. */
  destroy(): void { for (const u of this.owned) u.destroy(); this.owned = []; }
}

export type Bindings = Record<string, GPUBuffer | null | undefined>;
export type Params = Record<string, number>;

const MAX_X = 65535;

/** Split a linear workgroup count over x/y; kernels rebuild the id as wid.x + wid.y * num_workgroups.x. */
export function linearWg(n: number): [number, number, number] {
  return n <= MAX_X ? [Math.max(1, n), 1, 1] : [MAX_X, Math.ceil(n / MAX_X), 1];
}

export class Builder {
  steps: Step[] = [];
  private uniforms: GPUBuffer[] = [];

  constructor(public gpu: Gpu) {}

  dispatch(kernelName: string, bindings: Bindings, params: Params, wg: [number, number, number]): void {
    const k = this.gpu.kernel(kernelName);
    const dev = this.gpu.device;
    const entries: GPUBindGroupEntry[] = [];
    const seenWritable = new Set<GPUBuffer>();
    const seenRead = new Set<GPUBuffer>();
    for (const b of k.bindings) {
      if (b.name === 'p') {
        const data = new ArrayBuffer(Math.max(16, Math.ceil(k.fields.length * 4 / 16) * 16));
        const dv = new DataView(data);
        for (const key of Object.keys(params)) {
          if (!k.fields.some(f => f.name === key)) throw new Error(`${kernelName}: unknown param ${key}`);
        }
        k.fields.forEach((f, i) => {
          const v = params[f.name] ?? 0;
          if (f.type === 'f32') dv.setFloat32(i * 4, v, true);
          else if (f.type === 'i32') dv.setInt32(i * 4, v, true);
          else dv.setUint32(i * 4, v >>> 0, true);
        });
        const u = dev.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true, label: kernelName + '.p' });
        new Uint8Array(u.getMappedRange()).set(new Uint8Array(data));
        u.unmap();
        this.uniforms.push(u);
        entries.push({ binding: b.binding, resource: { buffer: u } });
        continue;
      }
      let buf = bindings[b.name];
      if (b.uniform) {
        if (!buf) throw new Error(`${kernelName}: missing uniform ${b.name}`);
      } else if (!buf) {
        buf = this.gpu.dummy(b.binding);
      } else {
        if (b.writable ? seenRead.has(buf) || seenWritable.has(buf) : seenWritable.has(buf))
          throw new Error(`${kernelName}: buffer bound twice with a writable alias (${b.name})`);
        (b.writable ? seenWritable : seenRead).add(buf);
      }
      entries.push({ binding: b.binding, resource: { buffer: buf } });
    }
    for (const key of Object.keys(bindings)) {
      if (!k.bindings.some(b => b.name === key)) throw new Error(`${kernelName}: unknown binding ${key}`);
    }
    const bindGroup = dev.createBindGroup({ layout: k.pipeline.getBindGroupLayout(0), entries, label: kernelName });
    this.steps.push({ k: 'd', pipeline: k.pipeline, bindGroup, wg, label: kernelName });
  }

  private freeBySize = new Map<number, GPUBuffer[]>();

  /** Scratch buffer owned (and destroyed) by the resulting Program. */
  buf(bytes: number, label?: string): GPUBuffer {
    const b = this.gpu.buffer(bytes, label);
    this.uniforms.push(b);
    return b;
  }

  /** Scratch buffer that may reuse one handed back via release(); steps run in order, so reuse after
   *  the last recorded reader is safe. */
  tmp(bytes: number, label?: string): GPUBuffer {
    const size = Math.max(16, Math.ceil(bytes / 16) * 16);
    return this.freeBySize.get(size)?.pop() ?? this.buf(size, label);
  }

  release(buf: GPUBuffer): void {
    const list = this.freeBySize.get(buf.size) ?? [];
    list.push(buf);
    this.freeBySize.set(buf.size, list);
  }

  copy(src: GPUBuffer, srcOff: number, dst: GPUBuffer, dstOff: number, size: number): void {
    if (size > 0) this.steps.push({ k: 'c', src, srcOff, dst, dstOff, size });
  }

  clear(buf: GPUBuffer, off: number, size: number): void {
    if (size > 0) this.steps.push({ k: 'clear', buf, off, size });
  }

  /** Ends a submission segment for Program.run (Program.encode ignores it). */
  split(): void {
    this.steps.push({ k: 'split' });
  }

  finish(): Program {
    const p = new Program(this.steps, this.uniforms);
    this.steps = [];
    this.uniforms = [];
    return p;
  }
}
