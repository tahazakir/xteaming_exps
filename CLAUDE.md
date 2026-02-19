# X-Teaming Project Notes

## Project Overview

X-Teaming is a framework for systematic exploration of multi-turn jailbreaks against LLMs. It uses collaborative agents (attacker, target, evaluator) with TextGrad-based prompt optimization.

**Key architecture**: All LLM calls route through `BaseAgent` (agents/base_agent.py), which abstracts providers (OpenAI, SGLang, Ollama, transformers, etc.). `GPTJudge` (agents/gpt_evaluator.py) also uses `BaseAgent` for provider flexibility.

## Current Work: Local Execution with HuggingFace Transformers

**Goal**: Run the entire pipeline locally without any API keys, using HuggingFace `transformers` for model inference on an H100/A100 (~80GB VRAM). Development/testing on M4 MacBook with small models.

### Implementation Checklist

- [x] **Add `transformers` provider to `BaseAgent`** (agents/base_agent.py)
  - Added `elif self.provider == "transformers"` branch in `__init__` to load model/tokenizer
  - Added `_call_transformers()` method for generation
  - Added module-level model cache (`_HF_MODEL_CACHE`) to avoid loading same model multiple times
  - Added threading locks (`_HF_LOCKS`) for thread safety with `ThreadPoolExecutor`
  - Added `"transformers"` entry to `provider_configs` dict

- [x] **Make all provider imports lazy** (agents/base_agent.py)
  - Moved `openai`, `aisuite`, `google.auth`, `vertexai` imports into provider-specific branches
  - `transformers` provider no longer requires OpenAI/Google packages to be installed

- [x] **Modify `GPTJudge` to use `BaseAgent`** (agents/gpt_evaluator.py)
  - Replaced hardcoded `OpenAI()` with `BaseAgent` instance
  - Updated `__init__` signature to accept config dict
  - JSON parsing uses existing fallback logic (extract between `{` and `}`)

- [x] **Update `main.py`**
  - Pass full eval config dict to `GPTJudge` (instead of just model name)
  - Pass eval_config to `TGAttackerAgent`

- [x] **Fix `TGAttackerAgent`** (agents/attacker_agent.py)
  - Updated to accept `eval_config` parameter and pass it to `GPTJudge`
  - Falls back to attacker config if no eval_config provided

- [x] **Update `config.yaml`** (config/config.yaml)
  - Set all components to `provider: "transformers"`
  - Added `provider` and `max_retries` fields to `evaluation` section
  - Using `Qwen/Qwen2.5-0.5B-Instruct` for local testing, swap to 32B+ on cluster
  - Minimal settings for quick testing: max_turns=3, max_retries=1, textgrad disabled

- [x] **Create `test_pipeline.py`** — minimal end-to-end test
  - Runs with a hardcoded benign behavior + strategy (no external data files needed)
  - Tests full attacker -> target -> evaluator flow
  - Saves results to `test_results/`

- [ ] **Verify end-to-end**
  - [x] Test `BaseAgent` standalone with transformers provider
  - [ ] Run `python test_pipeline.py` for quick pipeline validation
  - [ ] Run `python generate_attack_plans.py` (requires HarmBench CSV)
  - [ ] Run `python main.py` (requires generated attack plans)

### Key Files

| File | Role |
|------|------|
| agents/base_agent.py | Universal LLM provider abstraction (8 providers including transformers) |
| agents/gpt_evaluator.py | Jailbreak scoring judge (now uses BaseAgent) |
| agents/attacker_agent.py | Multi-turn attack orchestration |
| agents/target_model.py | Target LLM wrapper |
| tgd.py | TextGrad engine wrapping BaseAgent |
| main.py | Main entry point with ThreadPoolExecutor |
| generate_attack_plans.py | Attack strategy generation |
| test_pipeline.py | Minimal end-to-end pipeline test (no external data needed) |
| config/config.yaml | All model/provider configuration |

### Important Notes

- **Model caching**: Multiple `BaseAgent` instances share the same model via `_HF_MODEL_CACHE` to avoid OOM
- **Thread safety**: `model.generate()` is not thread-safe; use locks. Reduce `max_workers` to 1-2
- **Lazy imports**: Provider-specific packages (openai, aisuite, google-auth, vertexai) are imported only when that provider is used
- **tiktoken**: Used for token counting in attacker_agent.py; works fine as approximate counter for non-OpenAI models
- **TextGrad**: `TGBaseAgentEngine` (tgd.py) wraps `BaseAgent`, so it automatically gets transformers support
- **Development workflow**: Use small models (0.5B/1.5B) on M4 MacBook for testing, swap model name in config.yaml for cluster
- **Quick test**: Run `python test_pipeline.py` — uses hardcoded benign test data, no CSV or pre-generated plans needed
