// free-model-aggregator-02: one server, two wires, live free models.
//
// Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages
// (Anthropic), in JSON and server-sent events, plus GET /v1/models in the
// shape the caller's auth header names. Every completion is dialled live
// through the grader's gateway ($OPENAI_BASE_URL/chat/completions) on the
// trial's key, choosing the house's highest-rated free model that meets
// the request (JSON mode, tools, vision) and failing over down the rating
// order when a provider refuses. In a trial it registers with the env and
// exits 0 when GET $MAKEFUN_ENV_URL/work says done.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import {
  buildCatalog,
  loadBrief,
  orderCandidates,
  FALLBACK_MODELS,
  type Catalog,
  type FreeModel,
  type Requirements,
} from "./lib/models.ts";
import {
  anthropicMessage,
  anthropicModelList,
  anthropicStream,
  anthropicToOpenAI,
  isRecord,
  openAIModelList,
  openAIStream,
  requestText,
  viewOf,
  type Json,
} from "./lib/translate.ts";
import { dialWithFailover, extractToken, type CallBudget, type TrailRow } from "./lib/dial.ts";

const brief = loadBrief(process.env.MAKEFUN_PROMPT_FILE);
const models: FreeModel[] = brief !== null ? brief.models : FALLBACK_MODELS;
const catalog: Catalog = buildCatalog(models);
const catalogJson = JSON.stringify(catalog);

if (process.argv.includes("--list")) {
  process.stdout.write(`${catalogJson}\n`);
  process.exit(0);
}

const gatewayBase = process.env.OPENAI_BASE_URL ?? "https://openrouter.ai/api/v1";
const gatewayKey = process.env.OPENAI_API_KEY ?? process.env.OPENROUTER_API_KEY ?? "";
const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const budget: CallBudget = { used: 0, cap: 56 };

function requirementsOf(wire: "openai" | "anthropic", body: Json): Requirements {
  let vision = false;
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === "image_url" || part.type === "image") vision = true;
      }
    }
  }
  return {
    json: wire === "openai" && body.response_format !== undefined && body.response_format !== null,
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
  };
}

// An OpenAI-shaped client body carried through to the gateway: everything
// the client sent stays, the model is chosen by the dialer, the answer is
// always collected whole (stream: false) so it can be verified and, when
// the client asked for events, re-emitted as a well-formed stream.
function carriedOpenAIBody(body: Json): Json {
  const out: Json = { ...body };
  delete out.stream;
  delete out.stream_options;
  if (typeof out.max_tokens !== "number" && typeof out.max_completion_tokens !== "number") {
    out.max_tokens = 1024;
  }
  return out;
}

function trailHeader(trail: TrailRow[]): string | null {
  if (trail.length === 0) return null;
  return JSON.stringify(trail.slice(0, 8));
}

function sendJson(res: ServerResponse, status: number, doc: Json, trail: TrailRow[]): void {
  const headers: Record<string, string> = { "content-type": "application/json" };
  const trailValue = trailHeader(trail);
  if (trailValue !== null) headers["x-makefun-upstream-failures"] = trailValue;
  res.writeHead(status, headers);
  res.end(JSON.stringify(doc));
}

function sendStream(res: ServerResponse, body: string, trail: TrailRow[]): void {
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
  };
  const trailValue = trailHeader(trail);
  if (trailValue !== null) headers["x-makefun-upstream-failures"] = trailValue;
  res.writeHead(200, headers);
  res.end(body);
}

function sendError(
  res: ServerResponse,
  wire: "openai" | "anthropic",
  status: number,
  message: string,
  code: string,
  trail: TrailRow[],
): void {
  if (wire === "openai") {
    sendJson(
      res,
      status,
      { error: { message, type: "upstream_error", code: code !== "" ? code : "upstream_error", param: null } },
      trail,
    );
  } else {
    sendJson(res, status, { type: "error", error: { type: "api_error", message } }, trail);
  }
}

async function handleCompletion(
  wire: "openai" | "anthropic",
  body: Json,
  res: ServerResponse,
): Promise<void> {
  const wantsStream = body.stream === true;
  const openAIBody = wire === "openai" ? carriedOpenAIBody(body) : anthropicToOpenAI(body);
  const candidates = orderCandidates(models, requirementsOf(wire, body));
  const text = requestText(body);
  const token = extractToken(text);
  const result = await dialWithFailover(candidates, openAIBody, token, {
    baseUrl: gatewayBase,
    apiKey: gatewayKey,
    budget,
  });
  if (!result.ok) {
    sendError(res, wire, result.status, result.message, result.code, result.trail);
    return;
  }
  const dialedId = result.dialed.id;
  const view = viewOf(result.completion, text);
  if (wire === "openai") {
    if (wantsStream) {
      sendStream(res, openAIStream(view, dialedId), result.trail);
      return;
    }
    const message: Json = { role: "assistant", content: view.text };
    if (view.toolCalls.length > 0) {
      message.tool_calls = view.toolCalls.map((call) => ({
        id: call.id,
        type: "function",
        function: { name: call.name, arguments: call.argumentsJson },
      }));
    }
    sendJson(
      res,
      200,
      {
        id: view.id !== "" ? view.id : `chatcmpl-${dialedId}`,
        object: "chat.completion",
        created: view.created,
        model: dialedId,
        choices: [{ index: 0, message, finish_reason: view.finishReason }],
        usage: {
          prompt_tokens: view.inputTokens,
          completion_tokens: view.outputTokens,
          total_tokens: view.inputTokens + view.outputTokens,
        },
      },
      result.trail,
    );
    return;
  }
  if (wantsStream) {
    sendStream(res, anthropicStream(view, dialedId), result.trail);
    return;
  }
  sendJson(res, 200, anthropicMessage(view, dialedId), result.trail);
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > 8 * 1024 * 1024) throw new Error("request body too large");
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString("utf8");
}

const server = createServer((req, res) => {
  void (async () => {
    try {
      const path = (req.url ?? "").split("?")[0];
      if (req.method === "GET" && path === "/v1/models") {
        const anthropic = req.headers["x-api-key"] !== undefined;
        sendJson(res, 200, anthropic ? anthropicModelList(models) : openAIModelList(models), []);
        return;
      }
      if (req.method === "POST" && (path === "/v1/chat/completions" || path === "/v1/messages")) {
        const wire = path === "/v1/messages" ? "anthropic" : "openai";
        let body: Json;
        try {
          const parsed: unknown = JSON.parse(await readBody(req));
          if (!isRecord(parsed)) throw new Error("body is not an object");
          body = parsed;
        } catch {
          sendError(res, wire, 400, "the request body is not a JSON object", "invalid_request", []);
          return;
        }
        await handleCompletion(wire, body, res);
        return;
      }
      sendJson(res, 404, { error: { message: "no such route", type: "not_found", code: "not_found" } }, []);
    } catch {
      if (!res.headersSent) {
        sendJson(res, 500, { error: { message: "internal error", type: "internal", code: "internal" } }, []);
      } else {
        res.end();
      }
    }
  })();
});

// ---------------------------------------------------------------------------
// Trial lifecycle: register with the env, then poll /work until done.
// ---------------------------------------------------------------------------

async function register(port: number): Promise<boolean> {
  try {
    const response = await fetch(`${envUrl}/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: 1, port, catalog: JSON.parse(catalogJson) }),
      signal: AbortSignal.timeout(5000),
    });
    await response.text();
    return response.ok;
  } catch {
    return false;
  }
}

async function pollUntilDone(): Promise<void> {
  for (;;) {
    try {
      const response = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
      const doc: unknown = await response.json();
      if (isRecord(doc) && doc.done === true) return;
    } catch {
      // the env is briefly unreachable: keep polling
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
}

async function main(): Promise<void> {
  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address() as AddressInfo;
  process.stderr.write(`free-model-aggregator-02: listening on 0.0.0.0:${address.port}\n`);
  if (envUrl === "") return; // standalone: serve until stopped
  let registered = false;
  for (let i = 0; i < 40 && !registered; i++) {
    registered = await register(address.port);
    if (!registered) await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (registered) {
    await pollUntilDone();
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 2000).unref();
  }
}

void main();
