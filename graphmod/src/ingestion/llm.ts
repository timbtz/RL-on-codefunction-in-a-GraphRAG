// ============================================================
// llm.ts — thin, metered, STUBBABLE LLM client + cost Meter
// ============================================================
//
// The LLM-extraction lever (extract.ts) is the optimizer-insertable, DEFAULT-OFF
// way to recover relations buried in prose. This module provides:
//   - `LlmClient` interface so a unit test can inject a fake (no network).
//   - `getLlmClient()` factory: returns a real OpenRouter/OpenAI client when an
//     API key is present, else an OFFLINE client that THROWS if ever called —
//     it never opens a socket. The seed path never invokes it.
//   - `Meter`: accumulates {tokens, usd} per call and serializes ingest_cost.json
//     per the CONTRACT schema/path.
//
// Provider: OpenRouter / OpenAI (env OPENROUTER_API_KEY / OPENAI_API_KEY).

import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

// ---- Client interface ----

export interface LlmRequest {
  /** Logical field this call extracts for, e.g. "chat.msg:eng:7" (used in cost log). */
  field: string;
  /** System/instruction prompt. */
  system: string;
  /** User content (the text to extract from + schema constraints). */
  user: string;
  /** Optional model override. */
  model?: string;
}

export interface LlmResult {
  /** Parsed JSON object the model returned (caller validates shape). */
  json: unknown;
  tokens: number;
  usd: number;
}

export interface LlmClient {
  complete(req: LlmRequest): Promise<LlmResult>;
}

/** Raised when the offline stub is invoked (no API key configured). */
export class OfflineLlmError extends Error {
  constructor() {
    super(
      "LLM extraction requested but no API key (OPENROUTER_API_KEY/OPENAI_API_KEY) is set. " +
        "The offline client makes no network calls.",
    );
    this.name = "OfflineLlmError";
  }
}

// ---- Pricing ----

const DEFAULT_MODEL = process.env.INGEST_LLM_MODEL ?? "openai/gpt-4o-mini";
// USD per 1K tokens (blended). Overridable via env so selection can re-price.
const USD_PER_1K = Number(process.env.INGEST_LLM_USD_PER_1K ?? "0.0006");

function priceTokens(tokens: number): number {
  return (tokens / 1000) * USD_PER_1K;
}

// ---- Real client (OpenRouter / OpenAI compatible chat completions) ----

class HttpLlmClient implements LlmClient {
  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
  ) {}

  async complete(req: LlmRequest): Promise<LlmResult> {
    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: req.model ?? DEFAULT_MODEL,
        messages: [
          { role: "system", content: req.system },
          { role: "user", content: req.user },
        ],
        response_format: { type: "json_object" },
        temperature: 0,
      }),
    });
    if (!res.ok) {
      throw new Error(`LLM HTTP ${res.status}: ${await res.text()}`);
    }
    const body = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
      usage?: { total_tokens?: number };
    };
    const content = body.choices?.[0]?.message?.content ?? "{}";
    const tokens = body.usage?.total_tokens ?? 0;
    let json: unknown = {};
    try {
      json = JSON.parse(content);
    } catch {
      json = {};
    }
    return { json, tokens, usd: priceTokens(tokens) };
  }
}

/** Offline stub: never makes a network call; throws if the lever invokes it. */
class OfflineLlmClient implements LlmClient {
  async complete(_req: LlmRequest): Promise<LlmResult> {
    throw new OfflineLlmError();
  }
}

/**
 * Returns a real client when an API key is configured, otherwise an offline
 * stub that throws on use. Crucially, with no key NO network call is possible.
 */
export function getLlmClient(): LlmClient {
  const key = process.env.OPENROUTER_API_KEY ?? process.env.OPENAI_API_KEY;
  if (!key) return new OfflineLlmClient();
  const baseUrl = process.env.OPENROUTER_API_KEY
    ? process.env.INGEST_LLM_BASE_URL ?? "https://openrouter.ai/api/v1"
    : process.env.INGEST_LLM_BASE_URL ?? "https://api.openai.com/v1";
  return new HttpLlmClient(key, baseUrl);
}

// ---- Cost meter ----

export interface IngestCostCall {
  field: string;
  tokens: number;
  usd: number;
}

export interface IngestCost {
  ingest_hash: string;
  tokens: number;
  usd: number;
  calls: IngestCostCall[];
}

/**
 * Accumulates LLM cost across an ingest run and serializes ingest_cost.json
 * (CONTRACT path: graphmod/.ingest_cost/<hash>.json + a stable ingest_cost.json).
 * A zero-LLM seed run yields `{tokens:0, usd:0, calls:[]}`.
 */
export class Meter {
  tokens = 0;
  usd = 0;
  readonly calls: IngestCostCall[] = [];

  constructor(public ingest_hash = "seed") {}

  /** Record one metered call. */
  record(field: string, tokens: number, usd: number): void {
    this.tokens += tokens;
    this.usd += usd;
    this.calls.push({ field, tokens, usd });
  }

  toJSON(): IngestCost {
    return {
      ingest_hash: this.ingest_hash,
      tokens: this.tokens,
      usd: this.usd,
      calls: this.calls,
    };
  }

  /**
   * Write both the per-hash cost file and the stable last-run file.
   * @param graphmodDir absolute path to the graphmod/ root.
   * @returns the path of the stable ingest_cost.json.
   */
  writeCostFiles(graphmodDir: string): string {
    const dir = join(graphmodDir, ".ingest_cost");
    mkdirSync(dir, { recursive: true });
    const json = JSON.stringify(this.toJSON(), null, 2) + "\n";
    writeFileSync(join(dir, `${this.ingest_hash}.json`), json);
    const stable = join(graphmodDir, "ingest_cost.json");
    writeFileSync(stable, json);
    return stable;
  }
}
