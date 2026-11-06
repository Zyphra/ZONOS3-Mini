// Python semantics the speech-edit port needs to reproduce exactly: str.split() / str.strip() over Python whitespace,
// round() (half to even), format(x, '.Nf') (correctly rounded, exact ties to even) and CPython 3.12's sum() of a mix
// of ints and floats (Neumaier-compensated float additions, plain int additions). Strings are handled as code points.
import { pyIsSpace } from '../request/markup';

const isSpace = (ch: string) => pyIsSpace(ch.codePointAt(0)!);

/** str.split() with no separator: the non-whitespace runs. */
export function pySplit(text: string): string[] {
  const out: string[] = [];
  let cur = '';
  for (const ch of text) {
    if (isSpace(ch)) {
      if (cur) out.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  if (cur) out.push(cur);
  return out;
}

/** str.strip() with no argument. */
export function pyStrip(text: string): string {
  const cps = Array.from(text);
  let a = 0, b = cps.length;
  while (a < b && isSpace(cps[a])) a++;
  while (b > a && isSpace(cps[b - 1])) b--;
  return a === 0 && b === cps.length ? text : cps.slice(a, b).join('');
}

/** Length in code points (Python len of a str). */
export function pyLen(text: string): number {
  let n = 0;
  for (const _ of text) n++;
  return n;
}

/** round(x) of a float: the nearest integer, exact halves to the even one. */
export function pyRound(x: number): number {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** format(x, `.${digits}f`): Number.toFixed rounds exact ties away from zero, Python to even. */
export function pyFixed(x: number, digits: number): string {
  const fixed = x.toFixed(digits);
  if (!Number.isFinite(x)) return fixed;
  const exact = Math.abs(x).toFixed(100);
  const dot = exact.indexOf('.');
  if (!/^50*$/.test(exact.slice(dot + 1 + digits))) return fixed;
  const truncated = exact.slice(0, digits ? dot + 1 + digits : dot);
  if (Number(truncated.at(-1)) % 2 !== 0) return fixed;
  return (x < 0 ? '-' : '') + truncated;
}

/** A sum() term: its value and whether Python holds it as an int. */
export interface PyNumber { value: number; int: boolean }

/** sum() of ints and floats as CPython 3.12 computes it from the start value 0. */
export function pySum(terms: Iterable<PyNumber>): number {
  let ints = 0, f = 0, c = 0, float = false;
  for (const t of terms) {
    if (!float) {
      if (t.int) { ints += t.value; continue; }
      f = ints + t.value;
      float = true;
      continue;
    }
    if (t.int) { f += t.value; continue; }
    const x = t.value, s = f + x;
    c += Math.abs(f) >= Math.abs(x) ? (f - s) + x : (x - s) + f;
    f = s;
  }
  if (!float) return ints;
  return c && Number.isFinite(c) ? f + c : f;
}

/** sum() of floats. */
export function pySumFloats(values: Iterable<number>): number {
  return pySum((function* () { for (const value of values) yield { value, int: false }; })());
}
