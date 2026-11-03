/** Environment, read once at start. */
const env = process.env;

export const config = {
  /** The ZONOS3 server (same variable the studio's vite proxy uses). */
  server: (env.ZONOS_SERVER || 'http://127.0.0.1:1919').replace(/\/+$/, ''),
  /** Sent as `Authorization: Bearer <token>` when set. */
  token: env.ZONOS3_ACCESS_TOKEN || null,
  host: env.ZONOS_MCP_HOST || '127.0.0.1',
  port: Number(env.ZONOS_MCP_PORT) || 3001,
  /** The project tools write into until another is opened or created. */
  project: env.ZONOS_MCP_PROJECT || null,
};

export const VERSION = '0.1.0';
