/** Entry: `--stdio` for local clients, else Streamable HTTP (stateless) on ZONOS_MCP_HOST:ZONOS_MCP_PORT/mcp. */
import { createMcpExpressApp } from '@modelcontextprotocol/express';
import { NodeStreamableHTTPServerTransport } from '@modelcontextprotocol/node';
import { StdioServerTransport } from '@modelcontextprotocol/server/stdio';
import cors from 'cors';
import type { Request, Response } from 'express';
import { config } from './config';
import { createServer } from './server';

// stdout carries the stdio protocol: every log goes to stderr.
async function main() {
  if (process.argv.includes('--stdio')) {
    await createServer().connect(new StdioServerTransport());
    console.error(`Zonos MCP on stdio → ${config.server}`);
    return;
  }
  // Takes and clips travel as base64 JSON: well above express's 100 kB default.
  const app = createMcpExpressApp({ host: config.host, jsonLimit: '50mb' });
  app.use(cors());
  app.all('/mcp', async (req: Request, res: Response) => {
    const server = createServer();
    const transport = new NodeStreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    res.on('close', () => {
      void transport.close();
      void server.close();
    });
    try {
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
    } catch (e) {
      console.error('[zonos-mcp] request failed:', e);
      if (!res.headersSent) res.status(500).json({ jsonrpc: '2.0', error: { code: -32603, message: 'Internal server error' }, id: null });
    }
  });
  app.listen(config.port, config.host, (err?: Error) => {
    if (err) {
      console.error(err);
      process.exit(1);
    }
    console.error(`Zonos MCP on http://${config.host}:${config.port}/mcp → ${config.server}`);
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
