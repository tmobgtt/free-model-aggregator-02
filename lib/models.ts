// Brief parsing and the model catalog for free-model-aggregator-02.
//
// The trial brief (MAKEFUN_PROMPT_FILE) is prose with two JSON documents
// embedded in it: the prompts manifest and the house's free-model table.
// This module pulls those documents out, and falls back to a built-in copy
// of the same table when no brief is readable (standalone runs, --list).

import { readFileSync } from "node:fs";

export interface FreeModel {
  id: string;
  displayName: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface Brief {
  models: FreeModel[];
  promptCount: number;
}

const MODELS_SCHEMA = "free-model-aggregator-02/models/v1";
const PROMPTS_SCHEMA = "free-model-aggregator-02/prompts/v1";

// The house's free catalog, in the brief's own order. Ratings are the
// house's quality scale (rating_milli); higher is stronger.
export const FALLBACK_MODELS: FreeModel[] = [
  { id: "openrouter/baidu/ernie-5.5-flash-preview:free", displayName: "Baidu: ERNIE-5.5-Flash-Preview (free)", ratingMilli: 1150, contextLength: 1024000, tools: true, vision: false, json: true },
  { id: "openrouter/cohere/north-mini-code:free", displayName: "Cohere: North Mini Code (free)", ratingMilli: 1200, contextLength: 256000, tools: true, vision: false, json: false },
  { id: "openrouter/dots-studio/dots-3-note-preview:free", displayName: "Dots Studio: Dots3-Note Preview (free)", ratingMilli: 1150, contextLength: 512000, tools: true, vision: true, json: true },
  { id: "openrouter/google/gemma-4-26b-a4b-it:free", displayName: "Google: Gemma 4 26B A4B (free)", ratingMilli: 1150, contextLength: 262144, tools: true, vision: true, json: true },
  { id: "openrouter/google/gemma-4-31b-it:free", displayName: "Google: Gemma 4 31B (free)", ratingMilli: 1210, contextLength: 262144, tools: true, vision: true, json: true },
  { id: "openrouter/inclusionai/ling-3.2-flash:free", displayName: "inclusionAI: Ling 3.2 Flash (free)", ratingMilli: 1150, contextLength: 1048576, tools: true, vision: true, json: true },
  { id: "openrouter/liquid/lfm-3-1.7b:free", displayName: "LiquidAI: LFM3-1.7B (free)", ratingMilli: 1150, contextLength: 262144, tools: true, vision: false, json: true },
  { id: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", displayName: "NVIDIA: Nemotron 3 Nano Omni (free)", ratingMilli: 1150, contextLength: 256000, tools: true, vision: true, json: false },
  { id: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", displayName: "NVIDIA: Nemotron 3 Ultra (free)", ratingMilli: 1285, contextLength: 1000000, tools: true, vision: false, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-lightning:free", displayName: "NVIDIA: Nemotron 3.5 Lightning (free)", ratingMilli: 1221, contextLength: 1000000, tools: true, vision: false, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-super-vl:free", displayName: "NVIDIA: Nemotron 3.5 Super VL (free)", ratingMilli: 1303, contextLength: 1000000, tools: true, vision: true, json: true },
  { id: "openrouter/particle-ai/particle-1:free", displayName: "Particle AI: Particle-1 (free)", ratingMilli: 1150, contextLength: 1048576, tools: true, vision: true, json: true },
  { id: "openrouter/recursive/flint-v1.0:free", displayName: "Recursive: Flint (free)", ratingMilli: 1150, contextLength: 262144, tools: true, vision: false, json: true },
  { id: "openrouter/thinkingmachines/inkling-2-small:free", displayName: "Thinking Machines: Inkling 2 Small (free)", ratingMilli: 1475, contextLength: 1048576, tools: true, vision: true, json: false },
];

// Find the JSON document whose "schema" field equals `schema` inside prose
// text: locate the schema string, walk back to the opening brace, then scan
// forward with a brace counter that respects string literals and escapes.
export function extractDocument(text: string, schema: string): unknown {
  const at = text.indexOf(schema);
  if (at < 0) return null;
  const open = text.lastIndexOf("{", at);
  if (open < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(open, i + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function modelFromRow(row: Record<string, unknown>): FreeModel | null {
  if (typeof row.id !== "string" || row.id === "") return null;
  return {
    id: row.id,
    displayName: typeof row.display_name === "string" ? row.display_name : row.id,
    ratingMilli: typeof row.rating_milli === "number" ? row.rating_milli : 0,
    contextLength: typeof row.context_length === "number" ? row.context_length : null,
    tools: row.tools === true,
    vision: row.vision === true,
    json: row.json === true,
  };
}

// Read and parse the brief. Returns null when the file is unreadable or
// carries no usable model table; the caller then uses FALLBACK_MODELS.
export function loadBrief(path: string | undefined): Brief | null {
  if (!path) return null;
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    return null;
  }
  const doc = extractDocument(text, MODELS_SCHEMA) as Record<string, unknown> | null;
  if (!doc || !Array.isArray(doc.models)) return null;
  const models: FreeModel[] = [];
  for (const raw of doc.models) {
    if (raw !== null && typeof raw === "object" && !Array.isArray(raw)) {
      const model = modelFromRow(raw as Record<string, unknown>);
      if (model !== null && (raw as Record<string, unknown>).free !== false) models.push(model);
    }
  }
  if (models.length === 0) return null;
  const promptsDoc = extractDocument(text, PROMPTS_SCHEMA) as Record<string, unknown> | null;
  const promptCount =
    promptsDoc && Array.isArray(promptsDoc.prompts) ? promptsDoc.prompts.length : 0;
  return { models, promptCount };
}

// The catalog document: one provider, the fourteen dialable ids, each free.
// Key order is fixed so the registered document and --list's print of it are
// the same bytes.
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: CatalogModel[];
  }>;
}

export function buildCatalog(models: FreeModel[]): Catalog {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({
          id: m.id,
          free: true,
          context_length: m.contextLength,
        })),
      },
    ],
  };
}

export interface Requirements {
  json: boolean;
  tools: boolean;
  vision: boolean;
}

// Candidate ids for one request, best first: the models that meet the
// request's requirements, ordered by the house rating, ties keeping the
// brief's order so the choice is deterministic.
export function orderCandidates(models: FreeModel[], req: Requirements): FreeModel[] {
  const eligible = models.filter(
    (m) => (!req.json || m.json) && (!req.tools || m.tools) && (!req.vision || m.vision),
  );
  const pool = eligible.length > 0 ? eligible : models;
  return pool
    .map((m, i) => ({ m, i }))
    .sort((a, b) => b.m.ratingMilli - a.m.ratingMilli || a.i - b.i)
    .map((x) => x.m);
}
