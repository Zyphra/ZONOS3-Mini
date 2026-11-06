// Request validation against servers/api/openapi.json for the keyword subset the spec uses, with FastAPI-style 422 details
// (servers/ggml/src/server/schema.cpp: loc = ["body", path...], type "value_error"; JSON errors "json_invalid").
import spec from '../../../api/openapi.json';

export interface ValidationItem { loc: (string | number)[]; msg: string; type: string }

interface Schema {
  $ref?: string;
  type?: string | string[];
  properties?: Record<string, Schema>;
  required?: string[];
  additionalProperties?: boolean | Schema;
  items?: Schema;
  minItems?: number;
  maxItems?: number;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  oneOf?: Schema[];
  propertyNames?: Schema;
}

const schemas: Record<string, Schema> = spec.components.schemas as Record<string, Schema>;
const patterns = new Map<string, RegExp>();
const REF = '#/components/schemas/';

function resolve(s: Schema): Schema {
  while (s.$ref) {
    if (!s.$ref.startsWith(REF)) throw new Error(`validate: unsupported $ref ${s.$ref}`);
    const t = schemas[s.$ref.slice(REF.length)];
    if (!t) throw new Error(`validate: unknown schema ${s.$ref}`);
    s = t;
  }
  return s;
}

function typeOf(v: unknown): string {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  if (typeof v === 'number') return Number.isInteger(v) ? 'integer' : 'number';
  return typeof v;
}

function check(s0: Schema, v: unknown, path: (string | number)[], out: ValidationItem[]): void {
  const s = resolve(s0);
  const err = (msg: string, at = path) => out.push({ loc: ['body', ...at], msg, type: 'value_error' });
  if (s.type !== undefined) {
    const types = Array.isArray(s.type) ? s.type : [s.type];
    const k = typeOf(v);
    if (!types.some(t => t === k || (t === 'number' && k === 'integer'))) {
      err('unexpected instance type');
      return;
    }
  }
  if (s.enum && !s.enum.some(e => JSON.stringify(e) === JSON.stringify(v))) err('instance not found in required enum');
  if (s.oneOf) {
    let matched = 0;
    for (const alt of s.oneOf) {
      const sub: ValidationItem[] = [];
      check(alt, v, path, sub);
      if (!sub.length) matched++;
    }
    if (matched !== 1) err(matched ? 'more than one subschema has succeeded, but exactly one of them is required to validate'
      : 'no subschema has succeeded, but one of them is required to validate');
  }
  if (typeof v === 'number') {
    if (s.minimum !== undefined && v < s.minimum) err(`instance is below minimum of ${s.minimum}`);
    if (s.maximum !== undefined && v > s.maximum) err(`instance exceeds maximum of ${s.maximum}`);
    if (s.exclusiveMinimum !== undefined && v <= s.exclusiveMinimum) err(`instance is below or equals minimum of ${s.exclusiveMinimum}`);
  }
  if (typeof v === 'string') {
    const n = Array.from(v).length;
    if (s.minLength !== undefined && n < s.minLength) err(`instance is too short as per minLength:${s.minLength}`);
    if (s.maxLength !== undefined && n > s.maxLength) err(`instance is too long as per maxLength:${s.maxLength}`);
    if (s.pattern !== undefined) {
      let re = patterns.get(s.pattern);
      if (!re) { re = new RegExp(s.pattern, 'u'); patterns.set(s.pattern, re); }
      if (!re.test(v)) err(`instance does not match regex pattern: ${s.pattern}`);
    }
  }
  if (Array.isArray(v)) {
    if (s.minItems !== undefined && v.length < s.minItems) err(`array has too few items`);
    if (s.maxItems !== undefined && v.length > s.maxItems) err(`array has too many items`);
    if (s.items) v.forEach((x, i) => check(s.items!, x, [...path, i], out));
  }
  if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
    const obj = v as Record<string, unknown>;
    for (const r of s.required ?? []) if (!(r in obj)) err(`required property '${r}' not found in object`);
    for (const [k, x] of Object.entries(obj)) {
      if (s.propertyNames) check(s.propertyNames, k, [...path, k], out);
      const p = s.properties?.[k];
      if (p) { check(p, x, [...path, k], out); continue; }
      if (s.additionalProperties === false) {
        err(`validation failed for additional property '${k}': instance invalid as per false-schema`, [...path, k]);
      } else if (s.additionalProperties && typeof s.additionalProperties === 'object') {
        check(s.additionalProperties, x, [...path, k], out);
      }
    }
  }
}

/** Violations of `body` against #/components/schemas/<name> (empty: conforms). */
export function validate(name: string, body: unknown): ValidationItem[] {
  const s = schemas[name];
  if (!s) throw new Error(`servers/api/openapi.json has no schema ${name}`);
  const out: ValidationItem[] = [];
  check(s, body, [], out);
  return out;
}
