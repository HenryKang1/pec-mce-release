# Model Artifact Manifest

This manifest records the exact local GGUF artifacts used for the
MCE-COMPASS experiments. Base model IDs identify the public model cards;
GGUF source repos identify the quantized artifacts downloaded by
`shared/scripts/download_models.py`.

| Reader key | Base model ID | GGUF source repo | GGUF filename | Size bytes | SHA-256 |
|---|---|---|---|---:|---|
| `lfm2.5-1.2b-instruct` | `LiquidAI/LFM2.5-1.2B-Instruct` | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | 730895168 | `B1B3DE114215D9507409A662A501A631095A479A419584E8A2DED6304B19B4F5` |
| `qwen3-4b` | `Qwen/Qwen3-4B` | `unsloth/Qwen3-4B-GGUF` | `Qwen3-4B-Q4_K_M.gguf` | 2497281312 | `F6F851777709861056EFCDAD3AF01DA38B31223A3BA26E61A4F8BF3A2195813A` |
| `gemma-4-e4b` | `google/gemma-4-E4B-it` | `bartowski/google_gemma-4-E4B-it-GGUF` | `google_gemma-4-E4B-it-Q4_K_M.gguf` | 5405168384 | `51865750ADAFD22DE56994A343D5A887CC1A589B9BAE41D62B748C8BD0CA9C76` |

The corresponding configuration entries are in
`shared/utils/config.py` under `GGUF_MODELS`. Before anonymous
submission, verify that public model-card links and quantized-artifact
links remain accessible under the submission's anonymity policy.
