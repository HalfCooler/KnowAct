# GUIClaw CLI and Configuration Reference

This document covers the standalone `guiclaw` command and the GUIClaw adapter
bundled with nanobot. For a shorter introduction, see the repository
[README](../../README.md).

## Installation modes

GUIClaw can be used in two ways:

| Mode | Entry point | Use when |
| --- | --- | --- |
| Bundled host | `nanobot webui`, `nanobot gateway`, or `nanobot agent` | You want GUI routing alongside chat channels, files, shell, web, MCP, memory, and other nanobot tools. The `gui_task` adapter is already wired. |
| Standalone CLI | `guiclaw` | Another agent host, script, or terminal process should invoke GUI automation as a subprocess. |

The Python distribution is currently named `nanobot-ai` and contains both
packages. CLI-only installation means using only the `guiclaw` executable at
runtime; it is not a separate PyPI distribution. The `guiclaw` package itself
does not import `nanobot`.

## Requirements

- Python 3.11 or newer
- A multimodal model available through an OpenAI-compatible Chat Completions API
- [uv](https://docs.astral.sh/uv/) for the commands below
- Backend-specific software for the target platform

Clone the repository once:

```bash
git clone https://github.com/HITsz-TMG/KnowAct.git
cd KnowAct
```

### Install the complete host

```bash
cd GUIClaw
uv sync --extra web --extra desktop --extra cjk
uv run nanobot onboard --wizard
```

The host configuration is stored in `~/.nanobot/config.json`. See
[nanobot adapter configuration](#nanobot-adapter-configuration).

### Install the standalone command

Run these commands from the `KnowAct` repository root. Choose the extra that
matches the backend:

```bash
# Android, HarmonyOS, or dry-run
uv tool install ./GUIClaw

# Local desktop
uv tool install './GUIClaw[desktop]'

# iOS through WebDriverAgent
uv tool install './GUIClaw[ios]'
```

Add `cjk` when CJK tokenization is needed for memory or skill retrieval, for
example `uv tool install './GUIClaw[desktop,cjk]'`.

Verify the command:

```bash
guiclaw --help
guiclaw shortcuts --help
```

## Standalone quick start

The standalone command requires a YAML configuration. Create
`~/.guiclaw/config.yaml`:

```yaml
provider:
  base_url: "https://api.example.com/v1"
  model: "your-vision-model"

max_steps: 15
stagnation_limit: 0
image_scale_ratio: 0.5
agent_profile: default
```

Keep credentials outside the file when possible:

```bash
export OPENAI_API_KEY="your-api-key"
```

Run dry-run first. A provider is still required because dry-run replaces the
device backend, not the model call.

```bash
guiclaw --dry-run "Describe the current screen and finish"
guiclaw --backend adb "Open Settings and enable Wi-Fi"
guiclaw --backend local "Open the browser and visit github.com"
```

Use `--json` when another process consumes the result:

```bash
guiclaw --backend adb --json --task "Open Contacts and search for John"
```

The JSON object contains `success`, `summary`, `model_summary`, `trace_path`,
`steps_taken`, and `error`. Exit code `0` means success; exit code `1` means the
GUI task failed.

## Advisory policy setup

On the first GUI task, GUIClaw checks the `POLICY` entries in its memory store. If
none exist, it creates one conservative default entry in
`~/.guiclaw/memory/policy.md`. The default setup gives the model these operating
preferences:

- Unless the task explicitly requires and authorizes a permission, choose the
  conservative option: deny, cancel, select **Not now**, or go back.
- Do not grant camera, microphone, location, contacts, notifications, storage,
  accessibility, screen-recording, or account permissions merely to make progress.
- If authorization is unclear and the permission blocks the task, request user
  intervention.
- Do not confirm purchases, payments, deletion, sending, publishing, account
  changes, or other irreversible/external actions unless the task requires them.

GUIClaw does not add or replace the default when user-defined `POLICY` entries
already exist. Edit `~/.guiclaw/memory/policy.md` using the structured memory-entry
format to change the setup or add concise hints for your device and workflow. Both
the standalone CLI and the bundled nanobot adapter inject all `POLICY` entries into
every GUI task without relevance filtering.

Each entry is an H2 section. Preserve `id`, `type`, and `platform`, then edit the
body after the blank line. A minimal custom entry is:

```markdown
## Ask before sharing data
id: custom-sharing-policy
type: policy
platform: all

Ask for user confirmation before sharing data with another application.
```

Policy memory is prompt guidance, not a hard security or authorization control. A
model may misunderstand or fail to follow it. Enforce mandatory restrictions with
OS/app permissions, backend allowlists, host-side intervention approval, or
sandboxing.

## Available commands

| Command | Description |
| --- | --- |
| `guiclaw TASK` | Run one GUI task with the default local backend. |
| `guiclaw --task TASK` | Run one GUI task using the named task option. |
| `guiclaw --dry-run TASK` | Exercise the model and agent loop without changing a real device. |
| `guiclaw --backend adb TASK` | Run on an Android device or emulator selected by ADB. |
| `guiclaw --backend ios TASK` | Run through WebDriverAgent. |
| `guiclaw --backend hdc TASK` | Run on a HarmonyOS device selected by HDC. |
| `guiclaw --backend local TASK` | Run foreground desktop automation. |
| `guiclaw shortcuts SOURCE` | Infer Android shortcut candidates from a manifest file or directory. |
| `guiclaw shortcuts CACHE.json --validate` | Validate an existing shortcut cache on an ADB device. |
| `guiclaw shortcuts CACHE.json --validate --promote` | Validate candidates and append eligible shortcuts to `skills.py`. |
| `guiclaw --help` | Show task command options. |
| `guiclaw shortcuts --help` | Show shortcut inference and validation options. |

## `guiclaw` options

Synopsis:

```text
guiclaw [OPTIONS] TASK
guiclaw [OPTIONS] --task TASK
```

| Option | Default | Description |
| --- | --- | --- |
| `TASK` | — | Positional task description. |
| `--task TEXT` | — | Named task description. It must agree with `TASK` if both are supplied. |
| `--backend {adb,ios,hdc,local,dry-run}` | `local` | Device or desktop backend. |
| `--dry-run` | off | Alias for `--backend dry-run`. |
| `--agent-profile PROFILE` | config value, then `default` | Override the model prompt and action format. |
| `--json` | off | Print one machine-readable JSON result. |
| `--config PATH` | `~/.guiclaw/config.yaml` | Use another standalone YAML config. |
| `--background` | off | Request isolated background desktop execution. Linux uses Xvfb; Windows uses the isolated desktop backend when available. |
| `--require-isolation` | off | Fail instead of falling back when isolation is unavailable. Requires `--background`. |
| `--target-app-class CLASS` | `classic-win32` on Windows background runs | Windows isolation hint: `classic-win32`, `uwp`, `directx`, `gpu-heavy`, or `electron-gpu`. |
| `--display-num N` | `99` | Xvfb display number. |
| `--width N` | `1280` | Background display width. |
| `--height N` | `720` | Background display height. |

`--background` is only valid with the local backend. macOS currently supports
foreground desktop automation only.

## Standalone YAML configuration

Only `provider.base_url` and `provider.model` are required. Unknown or malformed
sections are rejected when they affect a supported field.

| Field | Default | Description |
| --- | --- | --- |
| `provider.base_url` | required | OpenAI-compatible API base URL. |
| `provider.model` | required | Vision model identifier. |
| `provider.api_key` | `OPENAI_API_KEY` | Optional inline key; the environment variable is used when omitted. |
| `provider.temperature` | server default | Optional sampling temperature. |
| `provider.top_p` | server default | Optional nucleus-sampling value in `(0, 1]`. |
| `provider.vl_high_resolution_images` | automatic | Optional provider-specific high-resolution image request. Supported endpoints are detected automatically. |
| `embedding` | omitted | Optional OpenAI-compatible embedding endpoint. |
| `embedding.base_url` | required when `embedding` is set | Embedding API base URL. |
| `embedding.model` | required when `embedding` is set | Embedding model identifier. |
| `embedding.api_key` | provider key | Separate embedding key. |
| `max_steps` | `15` | Maximum GUI decision steps. Must be positive. |
| `stagnation_limit` | `0` | Repeated-screen limit; `0` disables the detector. |
| `image_scale_ratio` | `0.5` | Screenshot scale in `(0, 1]` for model and validation calls. Some profiles apply their own preprocessing. |
| `agent_profile` | `default` | Prompt and action contract. |
| `memory_dir` | `~/.guiclaw/memory` | Policy and extracted-memory storage. Retrieval additionally requires `embedding`. |
| `skills_dir` | `~/.guiclaw/skill` | Flat skill store for online reuse and extraction. |
| `enable_skill_execution` | `false` | Retrieve skills and expose `use_skill` to the GUI agent. Uses BM25 without `embedding` and hybrid retrieval with it. |
| `enable_skill_extraction` | `false` | Extract or evolve skills after each run. |
| `enable_memory_extraction` | `false` | Extract durable GUI memory after each run. |
| `adb.serial` | ADB-selected device | Android device serial. |
| `adb.adb_path` | `adb` | ADB executable. |
| `adb.capture_source` | `auto` | `auto`, `scrcpy`, or `screencap`. `auto` uses a fresh ADB screencap for `gui_owl` and scrcpy for other profiles. |
| `scrcpy.max_fps` | `12` | Maximum scrcpy stream frame rate. |
| `scrcpy.jpeg_quality` | `80` | JPEG quality for streamed frames. |
| `scrcpy.frame_timeout_ms` | `3000` | Frame wait timeout. |
| `scrcpy.max_frame_age_ms` | `1000` | Maximum accepted frame age. |
| `ios.wda_url` | `http://localhost:8100` | WebDriverAgent endpoint. |
| `hdc.serial` | HDC-selected device | HarmonyOS device serial. |
| `hdc.hdc_path` | `hdc` | HDC executable. |

See [Model and provider compatibility](model-providers.md) for tested profiles,
coordinate contracts, provider-specific request fields, and deployment
examples.

Example with skill reuse and post-run learning enabled:

```yaml
provider:
  base_url: "https://api.example.com/v1"
  model: "your-vision-model"

embedding:
  base_url: "https://api.example.com/v1"
  model: "your-embedding-model"

memory_dir: "~/.guiclaw/memory"
skills_dir: "~/.guiclaw/skill"
agent_profile: default
enable_skill_execution: true
enable_skill_extraction: true
enable_memory_extraction: true
```

The standalone CLI uses the memory retriever when `embedding` is configured.
Skill execution also works without embeddings through BM25 keyword retrieval.
When either extraction switch is enabled, the command reuses the same
`PostRunProcessor` as the nanobot adapter and waits for it before returning.
Skills are written to `skills_dir/skills.py`; memory is written to
`memory_dir/gui_memory_bank.jsonl`.

Desktop skill features are permanently disabled even when the individual skill
switches are enabled. macOS, Linux, and Windows do not retrieve, inject,
execute, extract, or evolve skills because desktop geometry and accessibility
state are not stable enough for deterministic replay and validation. Memory
extraction remains available.

## nanobot adapter configuration

Run the wizard first:

```bash
cd GUIClaw
uv run nanobot onboard --wizard
```

Then add a `gui` block to `~/.nanobot/config.json`. Provider and model names
must refer to a configured nanobot provider and a multimodal model.

```json
{
  "providers": {
    "custom": {
      "apiKey": "your-api-key",
      "apiBase": "https://api.example.com/v1"
    }
  },
  "gui": {
    "backend": "adb",
    "provider": "custom",
    "model": "your-vision-model",
    "agentProfile": "default",
    "adb": { "captureSource": "auto" },
    "maxSteps": 15,
    "enableSkillExecution": true,
    "enablePromptSkillSelection": true,
    "promptSkillTopK": 5,
    "promptShortcutOnly": false,
    "promptSkillAppFilter": false,
    "enableSkillExtraction": false,
    "enableMemoryExtraction": false
  }
}
```

The example enables online skill selection and disables the app filter so that
cross-app candidates can be retrieved. These are recommended settings for the
shared skill store, not the schema defaults.

### nanobot `gui` defaults

Both camelCase and snake_case keys are accepted.

| Field | Default |
| --- | --- |
| `backend` | `adb` |
| `model`, `provider` | host-selected when unset |
| `agentProfile` | `default` when unset |
| `adb.captureSource` | `auto`; fresh screencap for `gui_owl`, scrcpy otherwise |
| `artifactsDir` | `gui_runs` → `~/.guiclaw/gui_runs` |
| `shortcutCacheDir` | `shortcut_cache` → `~/.guiclaw/shortcut_cache` |
| `maxSteps` | `15` |
| `stagnationLimit` | `0` |
| `imageScaleRatio` | `0.5` |
| `background` | `false` |
| `displayWidth`, `displayHeight` | `1280`, `720` |
| `enableSkillExecution` | `false` |
| `enablePromptSkillSelection` | `false` |
| `promptSkillTopK` | `5` |
| `promptShortcutOnly` | `false` |
| `promptSkillAppFilter` | `true` |
| `skillValidStateMode` | `strict` |
| `enableSkillExtraction` | `false` |
| `enableMemoryExtraction` | `false` |
| `alwaysOnSkillTags` | `["compact_action"]` |

When enabled, post-run extraction writes skills to
`~/.guiclaw/skill/skills.py` and memory to
`~/.guiclaw/memory/gui_memory_bank.jsonl`.

The adapter also uses the shared `POLICY` entries in
`~/.guiclaw/memory/policy.md`; there is no separate nanobot policy setting.

## Agent profiles

| Profile | Expected model output |
| --- | --- |
| `default` | Provider-native OpenAI-style tool calls. |
| `general_e2e` | MobileWorld GeneralE2E JSON action text. |
| `gui_owl` | GUI-Owl action format. |
| `venus` | UI-Venus action format. |
| `seed` | Seed XML-style function calls. |
| `qwen3vl` | Qwen3-VL mobile action format. |
| `mai_ui` | MAI-UI action format. |
| `gelab` | GELab tab-separated action format. |

Use `default` for models with reliable native function calling. Select another
profile only when the model is trained or prompted for that exact format.

## Backends

### Android (`adb`)

Install Android platform tools, enable USB debugging, and verify the target:

```bash
adb devices
```

Use `adb.serial` or `ANDROID_SERIAL` when multiple devices are connected.

### iOS (`ios`)

Install the `ios` extra. WebDriverAgent must be built, signed, installed, and
running on the device. Confirm that the configured `ios.wda_url` is reachable.

### HarmonyOS (`hdc`)

Install HDC and the device-side UI test service, then verify the target:

```bash
hdc list targets
```

### Local desktop (`local`)

Install the `desktop` extra. macOS requires Accessibility and Screen Recording
permissions. Linux desktop sessions need a working display; background mode
uses Xvfb. Windows background isolation availability depends on the target
application class.

### Dry-run (`dry-run`)

Dry-run does not change a real device, but it still calls the configured model.
Use it to check configuration, prompting, output parsing, and result handling.

## Android shortcut workflow

### 1. Infer candidates statically

The source may be one `AndroidManifest.xml` or a directory containing multiple
manifests:

```bash
guiclaw shortcuts path/to/AndroidManifest.xml
guiclaw shortcuts path/to/manifests/
```

The default output is
`~/.guiclaw/shortcut_cache/<package>.json`. Static extraction identifies
declared intents and deep links; it does not prove that a candidate reaches the
expected page.

### 2. Validate on a device

```bash
guiclaw shortcuts ~/.guiclaw/shortcut_cache/com.example.app.json \
  --validate \
  --query "hello world"
```

Validation resolves and launches candidates on the selected ADB device. It can
change application state. Use a test device or account.

### 3. Add visual verification and promote

```bash
export DASHSCOPE_API_KEY="your-api-key"

guiclaw shortcuts ~/.guiclaw/shortcut_cache/com.example.app.json \
  --validate \
  --promote \
  --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --llm-model qwen3.5-flash \
  --llm-api-key-env DASHSCOPE_API_KEY
```

By default, promotion accepts page-validated candidates. The visual verifier is
configured by `--llm-base-url` and `--llm-model`. Use
`--allow-launchable-promote` only when launchability alone is an acceptable
contract.

Validation sidecars are written next to the GUIClaw home under
`~/.guiclaw/shortcut_cache_validation/`. Promoted shortcuts are inserted before
extracted skills in `~/.guiclaw/skill/skills.py`.

### Shortcut options

| Option | Default | Description |
| --- | --- | --- |
| `SOURCE` | required | Manifest file, manifest directory, or cache JSON. |
| `--output DIR` | `~/.guiclaw/shortcut_cache` | Static cache output. |
| `--validate` | off | Resolve, launch, and validate candidates through ADB. |
| `--serial SERIAL` | ADB-selected device | Device serial. |
| `--task TEXT` | empty | Target task used to focus probe plans. |
| `--query TEXT` | empty | Search or text payload used in probe variants. |
| `--max-candidates N` | `20` | Maximum candidates considered per plan. |
| `--max-probe-plans N` | `12` | Maximum capability probe plans. |
| `--max-try N` | `5` | Maximum variants tried per candidate. |
| `--include-risky` | off | Include pay, share, push, and authentication candidates. |
| `--validation-root DIR` | `~/.guiclaw/shortcut_cache_validation` | Validation sidecar directory. |
| `--promote` | off | Add eligible records to `skills.py`. |
| `--allow-launchable-promote` | off | Permit launchable-only records during promotion. |
| `--skill-store-root DIR` | `~/.guiclaw/skill` | Destination flat skill store. |
| `--llm-base-url URL` | empty | Visual verifier API base. |
| `--llm-model MODEL` | empty | Visual verifier model. |
| `--llm-api-key KEY` | empty | Inline verifier key; prefer the environment option. |
| `--llm-api-key-env NAME` | `OPENAI_API_KEY` | Environment variable holding the verifier key. |
| `--llm-temperature FLOAT` | `0.0` | Verifier temperature. |
| `--shortcut-postprocess {off,rules,llm}` | `rules` | Normalize and deduplicate promoted records; `llm` also refines names and descriptions. |

## Data directories

| Path | Contents |
| --- | --- |
| `~/.guiclaw/config.yaml` | Standalone CLI configuration. |
| `~/.guiclaw/gui_runs/` | Run directories, screenshots, `traj.json`, and `result.json`. |
| `~/.guiclaw/shortcut_cache/` | Static Android shortcut candidates. |
| `~/.guiclaw/shortcut_cache_validation/` | Runtime shortcut validation sidecars. |
| `~/.guiclaw/skill/skills.py` | Validated shortcuts followed by extracted GUI skills. |
| `~/.guiclaw/memory/policy.md` | Always-injected `POLICY` entries and the default permission setup. |
| `~/.guiclaw/memory/gui_memory_bank.jsonl` | Induced GUI memory items. |

Relative `artifactsDir` and `shortcutCacheDir` values in nanobot configuration
are resolved under `~/.guiclaw`. Absolute paths remain absolute.

## Calling GUIClaw from another agent

The smallest integration is a subprocess call with `--json`:

```python
import json
import subprocess


completed = subprocess.run(
    [
        "guiclaw",
        "--backend",
        "adb",
        "--json",
        "--task",
        "Open Contacts and search for John",
    ],
    check=False,
    capture_output=True,
    text=True,
)
result = json.loads(completed.stdout)
if not result["success"]:
    raise RuntimeError(result["error"] or result["summary"])
```

For an in-process integration, implement the `LLMProvider` and `DeviceBackend`
protocols and construct `GuiAgent`. See [GUIClaw Adapter Patterns](../ADAPTERS.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Config file not found` | Create `~/.guiclaw/config.yaml` or pass `--config PATH`. |
| Authentication error | Export `OPENAI_API_KEY` or set `provider.api_key`. |
| Model returns no usable action | Confirm that `agent_profile` matches the model output format. |
| No Android device | Run `adb devices`; set `adb.serial` when more than one device is present. |
| No iOS session | Check WebDriverAgent signing, device trust, port forwarding, and `ios.wda_url`. |
| Desktop capture or input denied | Grant Screen Recording and Accessibility permissions, or check the Linux display session. |
| No skills are selected in nanobot | Desktop skills are unsupported. On mobile, enable both `enableSkillExecution` and `enablePromptSkillSelection`; inspect `~/.guiclaw/skill/skills.py`. |
| No skills are selected in standalone CLI | Desktop skills are unsupported. On mobile, set `enable_skill_execution: true`; configure `embedding` for semantic retrieval or omit it for BM25-only retrieval. |
| Standalone run produces no extracted skill or memory | Enable `enable_skill_extraction` and/or `enable_memory_extraction`; inspect the task trace and the configured storage paths. |
| Shortcut validation does not promote | Use a visual verifier for page validation or explicitly allow launchable-only promotion. |

For exact options in the installed version, treat command help as authoritative:

```bash
guiclaw --help
guiclaw shortcuts --help
```
