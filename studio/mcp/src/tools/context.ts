/** What every generating tool needs from the server, fetched together. */
import type { ControlsSpec, Speaker } from '../../../src/api';
import type { Session } from '../project';
import { caps, NO_STEERING, spec } from '../backend';
import type { Capabilities } from '../zonos';
import * as zonos from '../zonos';

export interface Ctx {
  caps: Capabilities;
  spec: ControlsSpec;
  steering: boolean;
  speakers: Speaker[];
}

export async function loadCtx(): Promise<Ctx> {
  const [c, s, speakers] = await Promise.all([caps(), spec(), zonos.speakers()]);
  return { caps: c, spec: s.spec, steering: s.steering, speakers };
}

/** Steering input on a server without a bundle is refused (the server would ignore it). */
export function requireSteering(ctx: Ctx, asked: boolean) {
  if (asked && !ctx.steering) throw new Error(NO_STEERING);
}

/** Voices `openProject` registered again are new to the server: list them too. */
export async function withSession(ctx: Ctx, sess: Session): Promise<Session> {
  if (sess.map.size) ctx.speakers = await zonos.speakers();
  return sess;
}
