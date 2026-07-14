<p align="center">
  <img src="assets/logo.png" width="420" alt="GUIClaw" />
</p>

# GUIClaw

GUIClaw is a vision-based GUI automation framework for desktop, Android, iOS,
and HarmonyOS. It observes the current screen, asks a multimodal model for the
next action, executes that action through a device backend, and repeats until
the task completes.

GUIClaw can run as a standalone CLI or as a GUI tool inside another agent
framework. This repository includes an adapter for nanobot.

[Project overview](../README.md) · [简体中文](../README_CN.md) ·
[CLI reference](docs/guiclaw-cli.md) · [Adapter contract](ADAPTERS.md)

## Features

- Desktop, Android ADB, iOS WebDriverAgent, and HarmonyOS HDC backends
- Native tool-calling and model-specific GUI agent profiles
- Compact trajectories with screenshots, actions, token usage, and results
- Reusable GUI skills, shortcut discovery, and post-run skill extraction
- Standalone `guiclaw` CLI and a host-independent Python protocol boundary

## Architecture

`guiclaw/` does not depend on nanobot. Host integrations implement the
`LLMProvider` protocol and call `GuiAgent`; device operations are isolated behind
the `DeviceBackend` protocol. The bundled nanobot integration lives in
`nanobot/agent/gui_adapter.py` and `nanobot/agent/tools/gui.py`.

```text
Host agent or CLI
       │
       ▼
   GuiAgent ───────► LLMProvider
       │
       ├───────────► DeviceBackend
       │
       └───────────► trajectories, skills, and memory
```

## Install and run

Install the complete adapted distribution:

```bash
git clone https://github.com/HITsz-TMG/KnowAct.git
cd KnowAct/GUIClaw
uv sync
uv run nanobot gateway
```

Use GUIClaw directly:

```bash
cd KnowAct/GUIClaw
uv run guiclaw --backend adb "Open Settings and enable Wi-Fi"
```

Or install it as a CLI tool for Hermes, OpenClaw, or another host:

```bash
uv tool install ./GUIClaw
guiclaw --backend adb "Open Settings"
```

Runtime data is stored under `~/.guiclaw/` by default. See the
[CLI and configuration reference](docs/guiclaw-cli.md) for commands, model
configuration, skills, memory extraction, shortcuts, and storage paths.

On the first GUI task, an empty memory store receives a conservative `POLICY`
entry in `~/.guiclaw/memory/policy.md`: permission requests are denied, cancelled,
or deferred unless the task explicitly authorizes them. Existing policies are not
replaced, and users may edit the memory entry. Policy memory is an advisory model
hint, not an enforcement boundary; use OS permissions, backend restrictions, host
approval, or sandboxing when a rule must be enforced.

## nanobot documentation

This repository retains nanobot as the bundled host framework, but does not
duplicate nanobot's full documentation. For providers, channels, Gateway,
WebUI, configuration, API, SDK, deployment, and other non-GUIClaw features,
refer to the upstream project:

- [HKUDS/nanobot](https://github.com/HKUDS/nanobot)
- [nanobot README](https://github.com/HKUDS/nanobot#readme)
- [nanobot documentation](https://github.com/HKUDS/nanobot/tree/main/docs)

## Project documents

- [GUIClaw CLI and configuration](docs/guiclaw-cli.md)
- [Host adapter contract](ADAPTERS.md)
- [Security policy](SECURITY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [License](LICENSE)

## License and attribution

GUIClaw is distributed under the MIT License. It builds on nanobot and retains
the applicable license and third-party notices. GUIClaw is an independent
project and is not an official HKUDS/nanobot distribution.
