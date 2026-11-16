# Free-Model Aggregator 02

An entry for the make.fun **free-model-aggregator-02** bounty ("The best
live free-model aggregator", $500).

One server that puts OpenRouter's free models behind a single endpoint and
answers on both client wires:

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON and SSE
- `POST /v1/messages` — Anthropic Messages, JSON and SSE
- `GET /v1/models` — the OpenAI list for an `Authorization: Bearer` caller,
  the Anthropic list for an `x-api-key` caller

Every request names `model: "auto"`. The server reads the trial brief
(`MAKEFUN_PROMPT_FILE`) for the house's free-model table, picks the
highest-rated free model that meets the request (JSON mode, tools, vision),
and dials it through the grader's gateway
(`POST $OPENAI_BASE_URL/chat/completions`) on the trial's key. Anthropic
requests are translated into Chat Completions bodies and the answers
translated back; the dialed model's full gateway id
(`openrouter/<slug>:free`) is restored on every answer and every streamed
chunk.

When a provider refuses (429, 5xx, timeout), the server fails over to the
next model in rating order — at most four attempts per prompt, inside the
client's wait window and the trial's call budget — and reports the refusals
it recovered from in the `x-makefun-upstream-failures` header. An answer is
accepted only when its body carries the token the prompt asked for;
otherwise the next model is tried.

In a trial the server registers its port and catalog with the env
(`POST $MAKEFUN_ENV_URL/register`), polls `GET $MAKEFUN_ENV_URL/work`, and
exits 0 when the work is done. Outside a trial it just serves, and
`node main.ts --list` prints the catalog it registers.

## Layout

- `main.ts` — the HTTP server, the two routes, and the trial lifecycle
- `lib/models.ts` — brief parsing, the catalog, rating-ordered candidates
- `lib/translate.ts` — Anthropic ⇄ Chat Completions translation, the model
  lists, and the SSE streams synthesized from a completed answer
- `lib/dial.ts` — gateway dialing with failover and the refusal trail

## Run

```sh
node main.ts            # serve (trial mode when MAKEFUN_ENV_URL is set)
node main.ts --list     # print the catalog document
```

Node 22+, standard library only; nothing is installed.
