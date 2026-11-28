# Web interface — Lutong Bahay

A local, chat-first web UI for the Pinoy Recipe RAG pipeline. It lives entirely in
`webapp/`, so it does not modify any tracked file outside this folder.

```
webapp/
├─ app.py                 FastAPI entry point (routes + SSE streaming)
├─ retrieval.py           parquet -> assets -> hybrid hits -> recipe cards
├─ prompts.py             hits -> LLM prompt -> streamed answer (Ollama)
├─ demo.py                deterministic answers + SSE helpers
├─ requirements.txt       fastapi, uvicorn, rank_bm25, pyarrow
├─ static/
│  ├─ index.html
│  ├─ styles.css
│  └─ app.js
└─ tools/
   └─ mock_ollama.py      fake Ollama server for testing without Ollama
```

## Quick start

From the repository root, with `.venv` activated:

```powershell
pip install -r webapp/requirements.txt

# Terminal 1 — only if you have Ollama
ollama serve

# Terminal 2
uvicorn app:app --app-dir webapp --reload --port 8000
```

Open <http://localhost:8000>.

The app reads these local artifacts (already produced by the notebooks):

```
data/processed/cleaned_recipes.parquet
data/processed/alias_map.json
data/processed/recipe_vecs_multilingual-e5-small.npy
```

If the embedding cache is missing, `rag.py` rebuilds it on first start, exactly
like notebook 04. The parquet and alias map must exist — run notebooks 01 and
02 if they don't.

## How the interface works

- **Horizontal chat layout** — conversation list on the left, chat in the middle.
  The sidebar collapses with the toggle button in the header (the choice is
  remembered in `localStorage`); on phones it becomes a slide-over drawer.
- **Multi-turn** — the first message is your pantry ("may sitaw, bawang, at manok
  ako"). Follow-ups ("pick the fastest one", "show the steps") are sent with the
  whole conversation, so the model can answer in context.
- **Query understanding** — apart from ingredients, the parser reads the
  **time budget** ("i only have half an hour", "isang oras lang", "45 minutes
  max"), the **lifestyle** ("vegetarian", "vegan", "pescatarian", "gulay lang"),
  and applies them as filters with honest fallback ("no vegetarian match —
  closest shown").
- **Cuts and specifics** — "pork belly" / "liempo" / "kasim" map to the exact
  cut and also expand to their parent (`pork`), so if the dataset has a belly
  dish you get it, and otherwise the closest pork dish — and a generic "pork"
  pantry still matches belly recipes. Fish types (bangus, tilapia, galunggong,
  tulingan…) fall back to `fish` the same way.
- **Fixed retrieval** — the UI always uses the notebook's `hybrid` retriever. There
  is no method picker, so the interface shows the pipeline as intended.
- **One recipe at a time** — the answer shows only the most relevant recipe.
  More appears when the top dish has **other versions** (pork adobo style,
  auto-detected by dish name) or when you click **"Show more recipes"** / ask
  "show me more recipes" in chat.
- **Recipe cards** — clicking the card opens a rounded modal with the full
  ingredient list (with quantities) and method, in the spirit of CookWell.
- **Keep chatting while reading** — "Open in side panel" docks the same recipe
  as a right-hand column, so the chat stays usable. Follow-ups sent while a
  recipe is open are focused on that recipe.
- **Look and feel** — the "Sunfruit Summer" palette (lime, sunflower yellow,
  orange, pink on cream) taken from the group's mood board, with the pixel rat
  pizza sticker as the mascot: app icon, assistant avatar, and page cursor
  (`webapp/static/rat-pizza.png`, `webapp/static/cursor.png`).

The alias map itself lives in `src/alias_map.py` (the maintained source of
truth). Query-side aliases (isda -> fish, kasim -> pork shoulder, bihon ->
rice noodles, ...) apply immediately after regenerating
`data/processed/alias_map.json` via `write_alias_map()`. Corpus-side coverage
of the new canonicals (galunggong, miso, hotdog, ...) lands when notebooks 01
and 02 are re-run on the improved map.

## The NLP mechanism lives in `src/`, not in the webapp

The query-understanding engine is a project module, so the notebooks and the
interface share exactly the same tokenisation:

- `src/ingredient_utils.py` — phrase splitting, longest-first canonical
  matching, and `parse_query` for raw pantry text (English/Tagalog/Taglish)
- `src/query_understanding.py` — time budgets (half an hour, isang oras, and
  "no time limit"), diets with negation ("no longer a vegetarian" clears the
  tag; most recent statement wins), dish types (soup, stew, breakfast,
  noodles, grilled, dessert...), allergies (with seafood/dairy/egg/peanut
  groups), pantry accumulation across a conversation, constraint filtering,
  and pantry-fit ranking
- `src/dish_groups.py` — version grouping and per-recipe facts
- `src/alias_map.py` — the canonical ingredient lexicon and parent links

`webapp/rag.py` imports these modules; it does not define its own copy. To try
the mechanism without the interface:

```powershell
python examples/query_understanding_demo.py
python examples/query_understanding_demo.py "i want a soup, under 30 minutes"
```


## Answer modes

The UI always sends `mode: "auto"`, so it works with or without a local model.

| Mode | What happens |
|---|---|
| **auto** | Uses Ollama when it is reachable and `llama3.2:3b` is pulled; otherwise falls back to a local demo answer. |
| **demo** | Deterministic answer built from the retrieved recipes. No model, no network. |
| **llm** | Always calls Ollama. If it isn't running you get a clear error instead of a 500. |

Retrieval (`/api/search`) never needs Ollama. Only the answer text does. The
demo path also handles the common follow-ups: fastest option, fewest missing
ingredients, substitutions, and cooking steps.

## Testing without Ollama

Two options, both local:

1. **Let it fall back** — with no Ollama running, the sidebar pill reads
   **demo mode** and every answer comes from the deterministic path. This
   exercises the full UI, the SSE stream, multi-turn history, the recipe modal,
   and the side panel.
2. **Mock Ollama** — exercises the real `ollama.chat` client path:

   ```powershell
   # Terminal 1
   python webapp/tools/mock_ollama.py --port 11435

   # Terminal 2
   $env:OLLAMA_HOST = "http://localhost:11435"
   uvicorn app:app --app-dir webapp --reload --port 8000
   ```

   The health pill should read **Ollama ready**. The answer is canned by the
   mock, but the streaming protocol, error handling, and rendering are real.

To point the app at a groupmate's Ollama over the network, set `OLLAMA_HOST` to
her machine, e.g. `http://192.168.1.20:11434`. She must start Ollama with
`OLLAMA_HOST=0.0.0.0` and allow the port through her firewall.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /api/health?fresh=1` | — | assets loaded, recipe count, Ollama host/models (`fresh` skips the 5 s probe cache) |
| `POST /api/search` | `{query, method, k}` | normalized ingredients + recipe cards |
| `POST /api/chat` | `{pantry, messages, focus_row?, k?}` | SSE stream: `meta`, `token`, `error`, `done` |

`method` is `hybrid`, `overlap`, `bm25`, or `dense` (`/api/search` only). `mode`
is `auto`, `demo`, or `llm` and is optional on `/api/chat` (defaults to `auto`).
`messages` is the visible conversation, including the newest user turn; the
backend swaps the first pantry turn for the retrieval context prompt.

Recipe cards in `meta.recipes` include `name`, `time`, `servings`, `matched`,
`missing`, `instructions` (steps), and `ingredient_lines` (quantities) for the
recipe reader.

Example:

```powershell
curl.exe -s http://localhost:8000/api/chat `
  -H "Content-Type: application/json" `
  -d '{\"pantry\":\"may sitaw, bawang, at manok ako\",\"messages\":[{\"role\":\"user\",\"content\":\"may sitaw, bawang, at manok ako\"}]}'
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Pill says **setup needed** | Run notebooks 01 and 02, then restart the app. |
| Pill says **demo mode** | Ollama is not reachable. Start `ollama serve` and click the pill to re-check, or keep using demo answers. |
| Pill says **model missing** | `ollama pull llama3.2:3b`. |
| First start is slow | SentenceTransformer + embeddings load once at startup. |
| Port already in use | Change `--port 8000`. |

## Merge-safety notes

- Everything here is new files under `webapp/`; no existing file was edited.
- Do not edit the root `README.md`, `requirements.txt`, or `.gitignore` from
  this branch — keep web docs and dependencies in `webapp/`.
- `rag.py` is a copy of notebook 04's retrieval logic. If notebook 04 changes,
  mirror it here (or extract a shared `src/rag.py` in a separate PR).
- The only coupling to `src/` is `ingredient_utils` (`load_alias_map`,
  `build_lookup`, `normalize_text`) and `dish_groups.version_facts`.
