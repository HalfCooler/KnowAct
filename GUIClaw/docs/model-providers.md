# Model and Provider Compatibility

GUIClaw uses OpenAI-compatible APIs, but GUI models can differ in prompt format,
image preprocessing, coordinate space, and request extensions. Use this guide
with the [CLI and configuration reference](guiclaw-cli.md).

## Compatibility summary

| Model or endpoint | `agent_profile` | Coordinate contract | Recommended ADB capture |
| --- | --- | --- | --- |
| OpenAI-style model with native tool calling | `default` | Tool-schema coordinates | `auto` |
| Self-hosted GUI-Owl 1.5 | `gui_owl` | 0–1000 grid | `auto` |
| Alibaba Model Studio `gui-plus` | `gui_owl` | Smart-resized image pixels | `auto` |
| Versioned `gui-plus-*` models | `gui_owl` | 0–1000 grid | `auto` |

With `agent_profile: gui_owl`, `adb.capture_source: auto` resolves to a fresh
ADB screencap. Other profiles use scrcpy when `auto` is selected.

## Self-hosted GUI-Owl 1.5

Use the GUI-Owl profile directly and configure the input scale when needed:

```yaml
provider:
  base_url: "http://localhost:8000/v1"
  model: "mPLUG/GUI-Owl-1.5-8B-Instruct"
  reasoning_effort: "none"  # optional: disable thinking through vLLM
  # temperature: 0.2  # optional
  # top_p: 0.8

agent_profile: gui_owl
image_scale_ratio: 0.5  # applied before GUI-Owl factor-28 smart resize
history_image_window: 1  # current frame only; omit to keep GUI-Owl's default of 5
adb:
  capture_source: auto
```

The profile:

- keeps four prior screenshot turns plus the current screen by default, configurable with `history_image_window`;
- applies `image_scale_ratio`, then factor-28 smart resize;
- uses the 0–1000 coordinate grid;
- caps each GUI decision at 2048 output tokens.

Omit `temperature` and `top_p` to use the inference server's generation
defaults.

## Alibaba Model Studio GUI-Plus

Use the GUI-Owl action contract:

```yaml
provider:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "gui-plus"
  reasoning_effort: "none"  # optional: disable thinking on DashScope

postprocess_provider:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-plus"

agent_profile: gui_owl
adb:
  capture_source: auto
```

Set the API key in the configuration or export the default environment
variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

Keep text-heavy post-run work on a general model. In particular, compact skill
extraction can request up to 8192 output tokens, while some GUI-specialist
endpoints accept at most 2048. The nanobot adapter selects the host model for
post-processing automatically; standalone CLI users should set
`postprocess_provider` as shown above.

For DashScope or MAAS endpoints under `aliyuncs.com`, GUIClaw automatically
sends:

```json
{"vl_high_resolution_images": true}
```

This applies when the model basename is `gui-plus` or starts with
`gui-plus-`. Set the following only when the endpoint does not accept this
Alibaba-specific request field:

```yaml
provider:
  vl_high_resolution_images: false
```

### Thinking control

Standalone GUIClaw accepts `provider.reasoning_effort` and the same field under
`postprocess_provider`. Values `none`, `minimal`, and `minimum` disable
thinking; another non-empty value enables it. The request mapping is:

| Endpoint | Request body |
| --- | --- |
| DashScope (`aliyuncs.com`) | `{"enable_thinking": false}` |
| Local or other OpenAI-compatible endpoints, including vLLM | `{"chat_template_kwargs":{"enable_thinking": false}}` |

Omit `reasoning_effort` to preserve the server default. If an endpoint uses a
different contract, set `provider.extra_body`; it is merged last and therefore
overrides the automatic mapping.

### Coordinate behavior

The exact legacy model name `gui-plus` returns absolute coordinates in the
smart-resized image space. GUIClaw injects the resized dimensions into the
prompt and maps the returned coordinates back to the device screen.

Versioned names such as `gui-plus-2026-02-26` retain the GUI-Owl 0–1000
coordinate path. Do not rename a legacy deployment to a versioned identifier
unless it implements that coordinate contract.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Taps are consistently offset | Confirm `agent_profile: gui_owl` and verify whether the model uses smart-resized pixels or the 0–1000 grid. |
| GUI-Owl receives stale Android frames | Use `adb.capture_source: auto` or `screencap`. |
| Endpoint rejects `vl_high_resolution_images` | Set `provider.vl_high_resolution_images: false`. |
| The model still emits thinking after `reasoning_effort: none` | Confirm that the endpoint accepts the mapped field above; otherwise override it with `provider.extra_body`. |
| A self-hosted model loses small UI details | Increase `image_scale_ratio`; GUI-Owl applies it before factor-28 smart resize. |
| The model returns no usable action | Confirm that the selected profile matches the model's output format. |

Provider-specific notes should describe stable request or coordinate contracts.
Transient service outages and deployment-specific failures are better tracked
as issues rather than added to this reference.
