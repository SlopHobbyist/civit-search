# Civit-Search - Project Summary

> **Note**: This entire project was written by AI (Claude Code: Sonnet 4.5, and Antigravity: Gemini 3 Pro (High)). All code, architecture decisions, and implementation details were generated through AI assistance.
>
> I am a strong advocate for never mixing generated code into real repos.
> Projects like these should clearly disclose as such.

## What It Is

A small command-line tool for searching [Civitai](https://civitai.com) from the terminal. Given a query (or a model hash), it queries the Civitai API, applies filters (model type, base model, exclude terms, training-data-only), caches results locally, and prints/saves the matches.

## Current Implementation

### Core Features

- Text search: `python civitai_search.py "anime character"`
- Hash search (e.g. from PNG metadata): `python civitai_search.py --hash 0873291ac5`
- Filter by model type (repeatable): `--type LORA --type Checkpoint`
  - Supported: `LORA`, `Checkpoint`, `TextualInversion`, `Hypernetwork`, `AestheticGradient`, `Controlnet`, `Poses`
- Filter by base model: `--base-model Illustrious` (also `SDXL`, `"SD 1.5"`, `Flux`, …)
- Exclude terms (matched case-insensitively against name + tags, repeatable): `--exclude nsfw --exclude realistic`
- Pagination: `--max-pages` (default 5, 100 per page), `--max-results`
- Local JSON cache in `cache/` (24h TTL by default; `--no-cache`, `--cache-max-age`, `--clear-cache`)
- Save results to JSON with `--save`
- Civitai API key required via `CIVITAI_API_KEY` env var, `key.txt`, or `--api-key`

### Exclusive Feature
- Training-data-only filter: `--training-data-only` — only show models whose version includes a training-data archive. This includes models trained with the on-site trainer, even if the user selected to hide it! Use responsibly, and don't tell Civitai!

> **Note**: Use a separate tool like civit-dl to actually download models. https://github.com/SlopHobbyist/civit-dl

### How It Works

1. Reads the API key from `CIVITAI_API_KEY`, `key.txt`, or `--api-key` (in that order).
2. Builds a cache key from the query parameters (incl. exclude terms) and returns cached results if fresh.
3. Otherwise calls `GET https://civitai.com/api/v1/models` (or `/model-versions/by-hash/{hash}` for hash search), paginating until `--max-pages` or the result tail.
4. Applies post-filters (training-data-only, exclude terms) client-side and prints/saves the matches.

## Supported Platforms

Anywhere Python 3.8+ runs (developed/tested on Windows). No OS-specific code.

## How to Run

```bash
pip install -r requirements.txt
```

**Required** — provide your Civitai API key (get one at https://civitai.com/user/account) via any of:
- `CIVITAI_API_KEY` environment variable
- a `key.txt` file in the repo root containing only the key (gitignored)
- the `--api-key` CLI flag

Examples:
```bash
# General searches
python civitai_search.py "anime character"
python civitai_search.py "style" --type LORA --base-model Illustrious

# Exclude unwanted results
python civitai_search.py "character" --exclude nsfw --exclude realistic

# Training-data-only
python civitai_search.py "realistic" --training-data-only
python civitai_search.py "character" --type LORA --training-data-only

# Hash lookup
python civitai_search.py --hash 0873291ac5

# Save matches to JSON
python civitai_search.py "anime" --type LORA --save
```

Cache files land in `./cache/`. Saved result JSONs land in the working directory.

## License

**CC0 1.0 Universal (Public Domain)**

This work has been dedicated to the public domain under CC0 1.0 Universal.

You can:
- Use this code for any purpose (commercial, personal, educational, etc.)
- Modify and distribute it freely
- Use it without any attribution required

To the extent possible under law, the author has waived all copyright and related rights to this work.

For more information: [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)
