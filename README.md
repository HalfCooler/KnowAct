<p align="center">
  <img src="GUIClaw/assets/logo.png" width="480" alt="GUIClaw" />
</p>

<p align="center" style="font-size:32px;">
KnowAct-GUIClaw: Know Deeply, Act Perfectly, Personal GUI Assistant with Self-Evolving Memory and Skill
</p>

<p align="center">
Lychee Team, Harbin Institute of Technology, Shenzhen
</p>

<p align="center">
  Personal Assistant for desktop, Android, iOS, and HarmonyOS.
</p>

<p align="center">
  <a href="README_CN.md">简体中文</a> ·
  <a href="https://arxiv.org/abs/2607.12625">Paper</a> ·
  <a href="https://shibosusu.github.io/KnowAct-GUIClaw/">Website</a> ·
  <a href="https://shibosusu.github.io/KnowAct-GUIClaw/#demos">Demo</a> ·
  <a href="GUIClaw/docs/guiclaw-cli.md">CLI reference</a> ·
  <a href="GUIClaw/ADAPTERS.md">Adapter contract</a> ·
</p>



You can see the real-device experimental logs and trajectories at [here](https://github.com/HITsz-TMG/KnowAct/releases/tag/Result) 

![KnowAct-GUIClaw results overview](GUIClaw/assets/front.png)

 KnowAct-GUIClaw + open-source Kimi-2.6 achieves a state-of-the-art 64.1% on the long-horizon MobileWorld benchmark, outperforming all open agent frameworks and closed agents (Seed-2.0-Pro, GPT-5.5). The framework's knowledge memory and execution capabilities generalize to various base models: +8.5% on Kimi-2.6 and +16.2% on Qwen3.5-35B-A3B.

> [!IMPORTANT]
> GUI automation can click, type, launch applications, and change device state.
> Start with `--dry-run` and use a test device or account for validation tasks.

## Architecture

GUIClaw organizes personal GUI assistance as a Know–Route–Act–Reflect loop.

![Know–Route–Act–Reflect architecture](GUIClaw/assets/know-route-act-reflect.png)

## Choose an installation path

| Use case | Install | Entry point |
| --- | --- | --- |
| Complete agent host | Install the repository environment. The nanobot adapter is already configured in the codebase. | `nanobot webui`, `nanobot gateway`, `nanobot agent` |
| Standalone GUI tool | Install the local project as a uv tool and call it from a terminal, script, Hermes, OpenClaw, or another host. | `guiclaw` |

The current Python distribution is named `nanobot-ai` and ships both packages.
The standalone path uses only the `guiclaw` executable at runtime; `guiclaw`
does not import `nanobot`.

## Quick start: complete host

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), a multimodal
model, and the tooling required by the selected backend.

```bash
git clone https://github.com/HITsz-TMG/KnowAct.git
cd KnowAct/GUIClaw
uv sync --extra web --extra desktop --extra cjk
uv run nanobot onboard --wizard
```

Add a `gui` block to `~/.nanobot/config.json`. The provider must also be
configured for nanobot.

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
    "maxSteps": 15,
    "enableSkillExecution": true,
    "enablePromptSkillSelection": true,
    "promptSkillTopK": 5,
    "promptShortcutOnly": false,
    "promptSkillAppFilter": false
  }
}
```

Start the WebUI:

```bash
uv run nanobot webui
```

The host registers GUIClaw as `gui_task`. It routes screen work to GUIClaw and
keeps files, shell, web, MCP, memory, and other tools in the host runtime.

## Quick start: standalone CLI

Clone the repository, then install the command from the repository root:

```bash
git clone https://github.com/HITsz-TMG/KnowAct.git
cd KnowAct

# Android, HarmonyOS, or dry-run
uv tool install ./GUIClaw

# Use this instead for local desktop automation
# uv tool install './GUIClaw[desktop]'
```

Create `~/.guiclaw/config.yaml`:

```yaml
provider:
  base_url: "https://api.example.com/v1"
  model: "your-vision-model"

max_steps: 15
stagnation_limit: 0
image_scale_ratio: 0.5
agent_profile: default
```

Export the API key and run a smoke test:

```bash
export OPENAI_API_KEY="your-api-key"

guiclaw --dry-run "Describe the current screen and finish"
guiclaw --backend adb "Open Settings and enable Wi-Fi"
```

For another agent or script, request JSON output:

```bash
guiclaw --backend adb --json --task "Open Contacts and search for John"
```

See the [CLI and configuration reference](GUIClaw/docs/guiclaw-cli.md) for
installation extras, every flag, complete defaults, backend setup, shortcut
validation, and subprocess integration.

## Available commands

| Command | Description |
| --- | --- |
| `guiclaw TASK` | Run a GUI task with the default local backend. |
| `guiclaw --backend adb TASK` | Run on an Android device or emulator. |
| `guiclaw --backend ios TASK` | Run through WebDriverAgent. |
| `guiclaw --backend hdc TASK` | Run on a HarmonyOS device. |
| `guiclaw --backend local TASK` | Run foreground desktop automation. |
| `guiclaw --dry-run TASK` | Test the model and agent loop without changing a real device. |
| `guiclaw shortcuts SOURCE` | Infer Android shortcut candidates from a manifest or manifest directory. |
| `guiclaw shortcuts CACHE.json --validate` | Validate shortcut candidates on an ADB device. |
| `guiclaw shortcuts CACHE.json --validate --promote` | Add eligible validated shortcuts to the shared skill file. |
| `guiclaw --help` | Show task command options. |
| `guiclaw shortcuts --help` | Show shortcut command options. |

## Configuration and defaults

GUIClaw keeps its runtime data outside the nanobot workspace:

| Path | Purpose |
| --- | --- |
| `~/.guiclaw/config.yaml` | Standalone CLI configuration. |
| `~/.guiclaw/gui_runs/` | Screenshots, compact trajectories, and task results. |
| `~/.guiclaw/shortcut_cache/` | Statically inferred Android shortcuts. |
| `~/.guiclaw/shortcut_cache_validation/` | Runtime shortcut validation records. |
| `~/.guiclaw/skill/skills.py` | Validated shortcuts followed by extracted skills. |
| `~/.guiclaw/memory/policy.md` | Always-injected `POLICY` memory and its default conservative permission setup. |
| `~/.guiclaw/memory/gui_memory_bank.jsonl` | Induced GUI memory. |

When no `POLICY` entry exists, the first GUI task initializes a conservative
permission policy: deny, cancel, or defer requests unless the task explicitly
authorizes them. Users may edit the structured entry in `policy.md`. These rules
are model guidance, not guaranteed enforcement; use OS, backend, host approval,
or sandbox controls for mandatory restrictions.

Common defaults:

| Setting | Standalone CLI | nanobot adapter |
| --- | --- | --- |
| Backend | `local` | `adb` |
| Agent profile | `default` | `default` when unset |
| Maximum steps | `15` | `15` |
| Image scale | `0.5` | `0.5` |
| Prompt skill selection | `enable_skill_execution`, disabled by default | disabled by default |
| Skill and memory extraction | separate YAML switches, disabled by default | disabled by default |

The complete standalone and adapter field tables are in
[GUIClaw CLI and Configuration Reference](GUIClaw/docs/guiclaw-cli.md).

## Android shortcut validation

Static inference writes one cache file per package. Runtime validation can then
launch candidates on an ADB device and optionally use a vision model before
promotion:

```bash
export DASHSCOPE_API_KEY="your-api-key"

guiclaw shortcuts ~/.guiclaw/shortcut_cache/com.example.app.json \
  --validate \
  --promote \
  --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --llm-model qwen3.5-flash \
  --llm-api-key-env DASHSCOPE_API_KEY
```

Promotion writes eligible records to `~/.guiclaw/skill/skills.py`. Validation
launches intents and can change application state; use a test device when
possible.

## Backends

| Backend | Platform | Requirement |
| --- | --- | --- |
| `local` | macOS, Linux, Windows | Install the `desktop` extra. macOS requires Accessibility and Screen Recording permissions. |
| `adb` | Android | ADB-connected device or emulator. |
| `ios` | iOS | Install the `ios` extra and run a signed WebDriverAgent service. |
| `hdc` | HarmonyOS | HDC-connected device and the required UI test service. |
| `dry-run` | Tests and CI | No real device changes; the configured model is still called. |

Linux background runs use Xvfb. Windows can use an isolated desktop when the
target application is supported. macOS currently supports foreground desktop
automation only.

## Agent profiles

The CLI supports:

`default`, `general_e2e`, `gui_owl`, `venus`, `seed`, `qwen3vl`, `mai_ui`, and
`gelab`.

Use `default` for providers with reliable native function calling. Other
profiles reproduce the action format expected by their corresponding GUI model
families.

## Results

Pass@1 success rates reported in the paper:

![GUIClaw benchmark results](GUIClaw/assets/table.png)

## Repository layout

```text
KnowAct/
├── README.md
├── README_CN.md
└── GUIClaw/
    ├── guiclaw/       # Host-independent GUI runtime
    ├── nanobot/       # Bundled agent host and GUI adapter
    ├── webui/         # Browser workbench
    ├── tests/         # Python tests
    ├── docs/          # CLI, nanobot, and developer documentation
    └── pyproject.toml # nanobot and guiclaw entry points
```

## Development

```bash
cd GUIClaw
uv sync --extra dev --extra desktop --extra cjk
uv run pytest
uv run ruff check nanobot guiclaw tests
```

WebUI development additionally requires Bun:

```bash
cd GUIClaw/webui
bun install
bun run test
bun run build
```

## Documentation

- [GUIClaw framework overview](GUIClaw/README.md)
- [CLI and configuration reference](GUIClaw/docs/guiclaw-cli.md)
- [Adapter patterns](GUIClaw/ADAPTERS.md)
- [nanobot documentation](https://github.com/HKUDS/nanobot/tree/main/docs)
- [Contributing](https://github.com/HKUDS/nanobot/blob/main/CONTRIBUTING.md)
- [Security](https://github.com/HKUDS/nanobot/blob/main/SECURITY.md)
- [License](https://github.com/HKUDS/nanobot/blob/main/LICENSE)
- [Third-party notices](https://github.com/HKUDS/nanobot/blob/main/THIRD_PARTY_NOTICES.md)

## Citation

If you use KnowAct-GUIClaw in research, please cite:

```bibtex
@misc{li2026knowactguiclawknowdeeplyact,
  title        = {KnowAct-GUIClaw: Know Deeply, Act Perfectly, Personal GUI Assistant with Self-Evolving Memory and Skill},
  author       = {Yunxin Li and Jinchao Li and Shibo Su and Zhenran Xu and Chenrui Zhao and Tongshu Bian and Xiaoman Liang and Meishan Zhang and Baotian Hu and Min Zhang},
  year         = {2026},
  eprint       = {2607.12625},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2607.12625}
}
```

## Wechat

<img width="1216" height="1485" alt="13010e95273700ac0073da0dca954171" src="https://github.com/user-attachments/assets/80322a1d-fe58-4c7e-9881-011e684c949d" />




## Acknowledgements

GUIClaw builds on and draws inspiration from the following open-source and
research projects:

- [HKUDS/nanobot](https://github.com/HKUDS/nanobot) provides the lightweight
  agent host, provider, channel, tool, and configuration foundation used by the
  bundled distribution.
- [Tongyi-MAI/MobileWorld](https://github.com/Tongyi-MAI/MobileWorld/tree/main#-benchmark-statistics)
  informed the mobile GUI agent profiles, action conventions, and
  benchmark-oriented workflows.
- [stepfun-ai/gelab-zero](https://github.com/stepfun-ai/gelab-zero) informed
  GELab model integration and practical mobile GUI agent runtime patterns.
- [google-research/reasoning-bank](https://github.com/google-research/reasoning-bank)
  inspired the experience-driven memory induction used to retain reusable
  lessons from GUI trajectories.

We thank the authors and contributors of these projects for making their work
available to the community. Each project remains subject to its own license and
attribution requirements; see the linked repositories and nanobot's
[third-party notices](https://github.com/HKUDS/nanobot/blob/main/THIRD_PARTY_NOTICES.md)
for details. Mention here denotes technical influence or reuse where documented,
not official affiliation or endorsement.

## License and attribution

GUIClaw is distributed under the MIT License. The repository retains nanobot's
original [LICENSE](https://github.com/HKUDS/nanobot/blob/main/LICENSE) and
[THIRD_PARTY_NOTICES.md](https://github.com/HKUDS/nanobot/blob/main/THIRD_PARTY_NOTICES.md),
with additional attribution in [NOTICE](GUIClaw/NOTICE). GUIClaw is an independent
project and is not an official HKUDS/nanobot distribution.
