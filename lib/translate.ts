// Wire translation between the Anthropic Messages API and the OpenAI Chat
// Completions API, plus the two model-list documents and the server-sent
// event streams this server synthesizes from a completed (non-streamed)
// upstream answer.

import { createHash } from "node:crypto";
import type { FreeModel } from "./models.ts";

export type Json = Record<string, unknown>;

export function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function shaTag(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 24);
}

// ---------------------------------------------------------------------------
// Anthropic request -> Chat Completions request
// ---------------------------------------------------------------------------

function anthropicSystemText(system: unknown): string {
  if (typeof system === "string") return system;
  if (Array.isArray(system)) {
    const parts: string[] = [];
    for (const block of system) {
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
        parts.push(block.text);
      }
    }
    return parts.join("\n");
  }
  return "";
}

function imagePartFromAnthropic(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === "base64" && typeof source.data === "string") {
    const media = typeof source.media_type === "string" ? source.media_type : "image/png";
    return { type: "image_url", image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === "url" && typeof source.url === "string") {
    return { type: "image_url", image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
        parts.push(block.text);
      }
    }
    return parts.join("");
  }
  return "";
}

// Translate one Anthropic message into one or more Chat Completions
// messages (a tool_result block becomes its own role:"tool" message).
function translateMessage(message: Json, out: Json[]): void {
  const role = message.role === "assistant" ? "assistant" : "user";
  const content = message.content;
  if (typeof content === "string") {
    out.push({ role, content });
    return;
  }
  if (!Array.isArray(content)) {
    out.push({ role, content: "" });
    return;
  }
  const parts: Json[] = [];
  const toolCalls: Json[] = [];
  let plainText = "";
  let onlyText = true;
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === "text" && typeof raw.text === "string") {
      parts.push({ type: "text", text: raw.text });
      plainText += raw.text;
    } else if (raw.type === "image") {
      const part = imagePartFromAnthropic(raw);
      if (part !== null) {
        parts.push(part);
        onlyText = false;
      }
    } else if (raw.type === "tool_use" && role === "assistant") {
      onlyText = false;
      toolCalls.push({
        id: typeof raw.id === "string" ? raw.id : `call_${toolCalls.length}`,
        type: "function",
        function: {
          name: typeof raw.name === "string" ? raw.name : "",
          arguments: JSON.stringify(isRecord(raw.input) ? raw.input : {}),
        },
      });
    } else if (raw.type === "tool_result" && role === "user") {
      out.push({
        role: "tool",
        tool_call_id: typeof raw.tool_use_id === "string" ? raw.tool_use_id : "",
        content: toolResultText(raw.content),
      });
    }
  }
  if (role === "assistant") {
    const msg: Json = { role, content: onlyText ? plainText : parts };
    if (!onlyText && parts.length === 0) msg.content = plainText === "" ? null : plainText;
    if (toolCalls.length > 0) {
      msg.tool_calls = toolCalls;
      if (plainText === "" && parts.length === 0) msg.content = null;
    }
    out.push(msg);
    return;
  }
  if (parts.length > 0) {
    out.push({ role, content: onlyText ? plainText : parts });
  }
}

export function anthropicToOpenAI(body: Json): Json {
  const messages: Json[] = [];
  const system = anthropicSystemText(body.system);
  if (system !== "") messages.push({ role: "system", content: system });
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (isRecord(raw)) translateMessage(raw, messages);
    }
  }
  const out: Json = { messages };
  out.max_tokens = typeof body.max_tokens === "number" ? body.max_tokens : 1024;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences)) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = body.tools.filter(isRecord).map((tool) => ({
      type: "function",
      function: {
        name: typeof tool.name === "string" ? tool.name : "",
        description: typeof tool.description === "string" ? tool.description : "",
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: "object" },
      },
    }));
    const choice = body.tool_choice;
    if (isRecord(choice)) {
      if (choice.type === "auto") out.tool_choice = "auto";
      else if (choice.type === "any") out.tool_choice = "required";
      else if (choice.type === "none") out.tool_choice = "none";
      else if (choice.type === "tool" && typeof choice.name === "string") {
        out.tool_choice = { type: "function", function: { name: choice.name } };
      }
      if (choice.disable_parallel_tool_use === true) out.parallel_tool_calls = false;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Chat Completions answer -> what each client wire expects
// ---------------------------------------------------------------------------

export interface CompletionView {
  id: string;
  created: number;
  text: string;
  toolCalls: Array<{ id: string; name: string; argumentsJson: string }>;
  finishReason: string;
  inputTokens: number;
  outputTokens: number;
}

function estimateTokens(text: string): number {
  return Math.max(0, Math.ceil(text.length / 4));
}

// A normalized view of one upstream Chat Completions JSON answer, with the
// dialed model id restored by the caller. Usage falls back to a
// deterministic estimate when the provider reported none, so the wire's
// usage check (input >= 0, output >= 1) still has whole numbers to read.
export function viewOf(completion: Json, requestText: string): CompletionView {
  const choices = Array.isArray(completion.choices) ? completion.choices : [];
  const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  let text = "";
  if (typeof message.content === "string") text = message.content;
  else if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (isRecord(part) && typeof part.text === "string") text += part.text;
    }
  }
  const toolCalls: CompletionView["toolCalls"] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const raw of message.tool_calls) {
      if (!isRecord(raw) || !isRecord(raw.function)) continue;
      toolCalls.push({
        id: typeof raw.id === "string" ? raw.id : `call_${toolCalls.length}`,
        name: typeof raw.function.name === "string" ? raw.function.name : "",
        argumentsJson:
          typeof raw.function.arguments === "string" ? raw.function.arguments : "{}",
      });
    }
  }
  const usage = isRecord(completion.usage) ? completion.usage : {};
  const inRaw = typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : null;
  const outRaw = typeof usage.completion_tokens === "number" ? usage.completion_tokens : null;
  return {
    id: typeof completion.id === "string" && completion.id !== "" ? completion.id : "",
    created: typeof completion.created === "number" ? completion.created : 0,
    text,
    toolCalls,
    finishReason: typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
    inputTokens: inRaw !== null && inRaw >= 0 ? Math.floor(inRaw) : estimateTokens(requestText),
    outputTokens:
      outRaw !== null && outRaw >= 1 ? Math.floor(outRaw) : Math.max(1, estimateTokens(text)),
  };
}

const ANTHROPIC_STOP: Record<string, string> = {
  stop: "end_turn",
  length: "max_tokens",
  tool_calls: "tool_use",
  function_call: "tool_use",
  content_filter: "refusal",
};

export function anthropicStopReason(finishReason: string): string {
  return ANTHROPIC_STOP[finishReason] ?? "end_turn";
}

function parseToolInput(argumentsJson: string): Json {
  try {
    const value: unknown = JSON.parse(argumentsJson);
    if (isRecord(value)) return value;
    return { value };
  } catch {
    return { _raw: argumentsJson };
  }
}

// The Anthropic message document for one completed answer.
export function anthropicMessage(view: CompletionView, dialedId: string): Json {
  const content: Json[] = [];
  if (view.text !== "" || view.toolCalls.length === 0) {
    content.push({ type: "text", text: view.text });
  }
  for (const call of view.toolCalls) {
    content.push({
      type: "tool_use",
      id: call.id,
      name: call.name,
      input: parseToolInput(call.argumentsJson),
    });
  }
  return {
    id: `msg_${shaTag(`${dialedId}\n${view.id}\n${view.text}`)}`,
    type: "message",
    role: "assistant",
    model: dialedId,
    content,
    stop_reason: anthropicStopReason(view.finishReason),
    stop_sequence: null,
    usage: { input_tokens: view.inputTokens, output_tokens: view.outputTokens },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events, synthesized from the completed answer
// ---------------------------------------------------------------------------

function sseData(payload: Json): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function sseEvent(name: string, payload: Json): string {
  return `event: ${name}\ndata: ${JSON.stringify(payload)}\n\n`;
}

// Split text into deterministic pieces for streamed deltas: words with
// their trailing spaces, grouped so no piece exceeds about 24 characters.
export function textPieces(text: string): string[] {
  if (text === "") return [];
  const words = text.split(/(?<=\s)/);
  const pieces: string[] = [];
  let current = "";
  for (const word of words) {
    if (current !== "" && current.length + word.length > 24) {
      pieces.push(current);
      current = "";
    }
    current += word;
  }
  if (current !== "") pieces.push(current);
  return pieces.length > 0 ? pieces : [text];
}

export function openAIStream(view: CompletionView, dialedId: string): string {
  const id = view.id !== "" ? view.id : `chatcmpl-${shaTag(`${dialedId}\n${view.text}`)}`;
  const base = { id, object: "chat.completion.chunk", created: view.created, model: dialedId };
  let out = "";
  out += sseData({
    ...base,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
  });
  for (const piece of textPieces(view.text)) {
    out += sseData({
      ...base,
      choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
    });
  }
  view.toolCalls.forEach((call, i) => {
    out += sseData({
      ...base,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: i,
                id: call.id,
                type: "function",
                function: { name: call.name, arguments: call.argumentsJson },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    });
  });
  out += sseData({
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: view.finishReason }],
  });
  out += sseData({
    ...base,
    choices: [],
    usage: {
      prompt_tokens: view.inputTokens,
      completion_tokens: view.outputTokens,
      total_tokens: view.inputTokens + view.outputTokens,
    },
  });
  out += "data: [DONE]\n\n";
  return out;
}

export function anthropicStream(view: CompletionView, dialedId: string): string {
  const message = anthropicMessage(view, dialedId);
  let out = "";
  out += sseEvent("message_start", {
    type: "message_start",
    message: {
      id: message.id,
      type: "message",
      role: "assistant",
      model: dialedId,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: view.inputTokens, output_tokens: 0 },
    },
  });
  let index = 0;
  if (view.text !== "" || view.toolCalls.length === 0) {
    out += sseEvent("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "text", text: "" },
    });
    for (const piece of textPieces(view.text)) {
      out += sseEvent("content_block_delta", {
        type: "content_block_delta",
        index,
        delta: { type: "text_delta", text: piece },
      });
    }
    out += sseEvent("content_block_stop", { type: "content_block_stop", index });
    index++;
  }
  for (const call of view.toolCalls) {
    out += sseEvent("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "tool_use", id: call.id, name: call.name, input: {} },
    });
    out += sseEvent("content_block_delta", {
      type: "content_block_delta",
      index,
      delta: { type: "input_json_delta", partial_json: call.argumentsJson },
    });
    out += sseEvent("content_block_stop", { type: "content_block_stop", index });
    index++;
  }
  out += sseEvent("message_delta", {
    type: "message_delta",
    delta: { stop_reason: anthropicStopReason(view.finishReason), stop_sequence: null },
    usage: { output_tokens: view.outputTokens },
  });
  out += sseEvent("message_stop", { type: "message_stop" });
  return out;
}

// ---------------------------------------------------------------------------
// Model lists
// ---------------------------------------------------------------------------

export function openAIModelList(models: FreeModel[]): Json {
  return {
    object: "list",
    data: models.map((m) => ({
      id: m.id,
      object: "model",
      created: 0,
      owned_by: "openrouter",
      free: true,
    })),
  };
}

export function anthropicModelList(models: FreeModel[]): Json {
  return {
    data: models.map((m) => ({
      type: "model",
      id: m.id,
      display_name: m.displayName,
      created_at: "2026-11-14T00:00:00Z",
    })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

// The text of a request body, for token extraction and usage estimates:
// every message's text content, plus the Anthropic system prompt.
export function requestText(body: Json): string {
  const parts: string[] = [];
  if (typeof body.system === "string") parts.push(body.system);
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      if (typeof raw.content === "string") parts.push(raw.content);
      else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (isRecord(block) && typeof block.text === "string") parts.push(block.text);
        }
      }
    }
  }
  return parts.join("\n");
}
