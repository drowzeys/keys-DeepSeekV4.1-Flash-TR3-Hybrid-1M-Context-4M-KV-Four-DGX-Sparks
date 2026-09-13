# Prepare Hermes for this TR3-Hybrid serve

First Hermes turn on a fresh 1M TR3 boot is the slow one. The engine is fine;
the **first real agent request** is ~25 kB system + ~23 tool schemas, which
misses the tiny CUDA-graph shapes captured at boot. Telegram then looks hung
(20–200 s) even though `/v1/models` is up.

Do this **after** `oneshot.sh` reports SERVING, **before** you type in Telegram.

## 1. Warm the Hermes-shaped prefill

```bash
python3 serve/hermes-warmup.py --api http://10.100.10.1:8000/v1
```

Expect the fat warmup to take tens of seconds once; the follow-up ping should
be well under a second (`SPARK-TR3-OK`).

`oneshot.sh` runs this automatically after the API is up.

## 2. Point Hermes at the live API

`~/.hermes/config.yaml` (backup first):

```yaml
model:
  default: deepseek-v4.1-flash
  provider: custom
  base_url: http://10.100.10.1:8000/v1
  api_key: EMPTY
  max_tokens: 12288
  context_length: 1048576
  extra_body:
    temperature: 0.6
    top_p: 0.95
    chat_template_kwargs:
      thinking: false
      enable_thinking: false
agent:
  reasoning_effort: false    # MUST be YAML boolean false. null/None falls back to xhigh
  environment_probe: false
  # keep your no-catalog environment_hint; do not paste skills lists
```

**`reasoning_effort: false` is load-bearing.** `xhigh` / `null` makes V4.1 fill
`max_tokens` in the think channel and Telegram sees empty `content` (this lab:
first Hello sat **232 s**).

Restart the gateway after edits:

```bash
sudo systemctl restart hermes-gateway   # this fleet
# then /new in Telegram
```

## 3. First-prompt hygiene (every time you switch models)

- `/new` after pointing Hermes here.
- Do **not** inject extra first-turn protocol text on a clean session — the
  environment_hint already carries it. Extra injection fattens prefill.
- `environment_probe: false`.
- `tools.tool_search.enabled: auto` (defer unused tools).
- Greetings: one short sentence, no `skill_view`, no catalog.

## 4. Abliterated weights

Same Hermes prep. Serve
`MODEL_DIR=DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated` (or the HF pack
[`drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated-Cybersecurity-Unleashed`](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-TR3-Hybrid-Abliterated-Cybersecurity-Unleashed)).
Recipe: [ABLIT.md](ABLIT.md).
