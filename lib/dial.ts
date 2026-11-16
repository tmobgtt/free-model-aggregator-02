// Dialing the grader's gateway with rating-ordered failover.
//
// Every attempt is one Chat Completions POST to $OPENAI_BASE_URL with the
// trial's dummy key; the gateway swaps in the entrant's real key. Attempts
// are deterministic: candidates in the order given, at most four per
// prompt, inside a per-prompt deadline and a shared trial-wide call
// budget. Refusals are collected as the trail the failover column reads
// (x-makefun-upstream-failures).

import type { FreeModel } from "./models.ts";
import { isRecord, type Json } from "./translate.ts";

export interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

export interface DialSuccess {
  ok: true;
  completion: Json;
  rawBody: string;
  dialed: FreeModel;
  trail: TrailRow[];
  tokenVerified: boolean;
}

export interface DialFailure {
  ok: false;
  status: number;
  message: string;
  code: string;
  trail: TrailRow[];
}

export type DialResult = DialSuccess | DialFailure;

export interface CallBudget {
  used: number;
  cap: number;
}

const ATTEMPT_TIMEOUT_MS = 12_000;
const PROMPT_DEADLINE_MS = 27_000;
const MAX_ATTEMPTS = 4;
const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/g;

// The token a prompt asks for: the last WORD-1234 shape in the request's
// text (the instruction carrying it is the final user message).
export function extractToken(text: string): string | null {
  const matches = text.match(TOKEN_PATTERN);
  return matches !== null && matches.length > 0 ? matches[matches.length - 1] : null;
}

function errorCodeOf(rawBody: string): string {
  try {
    const value: unknown = JSON.parse(rawBody);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === "string" && value.error.code !== "") {
        return value.error.code.slice(0, 64);
      }
      if (typeof value.error.type === "string") return value.error.type.slice(0, 64);
    }
  } catch {
    // not JSON: no code to report
  }
  return "";
}

function errorMessageOf(rawBody: string, status: number): string {
  try {
    const value: unknown = JSON.parse(rawBody);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === "string") {
      return value.error.message;
    }
  } catch {
    // fall through to the generic message
  }
  return `the upstream model refused the request (HTTP ${status})`;
}

export interface DialOptions {
  baseUrl: string;
  apiKey: string;
  budget: CallBudget;
}

export async function dialWithFailover(
  candidates: FreeModel[],
  openAIBody: Json,
  token: string | null,
  opts: DialOptions,
): Promise<DialResult> {
  const trail: TrailRow[] = [];
  const startedAt = Date.now();
  let fallback: DialSuccess | null = null;
  let lastStatus = 502;
  let lastMessage = "no free model could be dialled";
  let lastCode = "";

  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS) break;
    if (opts.budget.used >= opts.budget.cap) break;
    if (attempts > 0 && Date.now() - startedAt > PROMPT_DEADLINE_MS) break;
    attempts++;
    opts.budget.used++;

    const url = `${opts.baseUrl.replace(/\/+$/, "")}/chat/completions`;
    let status = 0;
    let rawBody = "";
    try {
      // Clamp the attempt's timeout to the prompt's remaining budget, so a
      // late failover attempt can never push the answer past the client's
      // own wait limit.
      const remaining = PROMPT_DEADLINE_MS - (Date.now() - startedAt);
      const timeout = Math.max(1_000, Math.min(ATTEMPT_TIMEOUT_MS, remaining));
      const response = await fetch(url, {
        method: "POST",
        headers: {
          authorization: `Bearer ${opts.apiKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ ...openAIBody, model: candidate.id, stream: false }),
        signal: AbortSignal.timeout(timeout),
      });
      status = response.status;
      rawBody = await response.text();
    } catch (err) {
      const timedOut = err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError");
      status = timedOut ? 504 : 502;
      lastStatus = status;
      lastCode = timedOut ? "upstream_timeout" : "upstream_unreachable";
      lastMessage = timedOut
        ? "the upstream model did not answer in time"
        : "the upstream model could not be reached";
      trail.push({ status, code: lastCode, model: candidate.id });
      continue;
    }

    if (status < 200 || status >= 300) {
      lastStatus = status;
      lastCode = errorCodeOf(rawBody);
      lastMessage = errorMessageOf(rawBody, status);
      trail.push({ status, code: lastCode, model: candidate.id });
      continue;
    }

    let completion: Json | null = null;
    try {
      const value: unknown = JSON.parse(rawBody);
      if (isRecord(value)) completion = value;
    } catch {
      completion = null;
    }
    // A 2xx that is not a JSON object, or that carries no choice at all,
    // is not an answer: try the next model.
    if (completion === null) continue;
    if (!Array.isArray(completion.choices) || completion.choices.length === 0) continue;

    const success: DialSuccess = {
      ok: true,
      completion,
      rawBody,
      dialed: candidate,
      trail,
      tokenVerified: token === null || rawBody.includes(token),
    };
    if (success.tokenVerified) return success;
    if (fallback === null) fallback = success; // served, but the token is missing: keep as a last resort
  }

  if (fallback !== null) return fallback;
  return { ok: false, status: lastStatus, message: lastMessage, code: lastCode, trail };
}
