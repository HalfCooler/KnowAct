# Voice Assistant Shortcut Reference

Delegate natural-language tasks to the phone's built-in voice assistant via ADB. This bypasses the GUI agent entirely — no screenshots, no vision model, no multi-step action loop.

## Core pattern

All voice assistant shortcuts follow the same flow:

1. **Launch** the assistant's text-input activity via `adb shell am start`
2. **Input** the command text (via intent extra or `adb shell input text`)
3. **Confirm** by tapping the send button (located via `uiautomator dump`)

## Suitable tasks

Use voice assistant for tasks the assistant handles natively:

- Setting alarms, timers, countdowns
- Setting reminders
- Making phone calls ("call Mom", "call 13800138000")
- Sending SMS / messages
- Checking weather
- Playing music
- Controlling smart home devices
- Simple math / unit conversions
- Creating calendar events
- Navigation ("navigate to Beijing West Railway Station")

## Unsuitable tasks

Do NOT use voice assistant for:

- Complex app navigation (multi-screen flows)
- Reading or extracting screen content
- Tasks requiring visual confirmation of results
- Interacting with third-party apps the assistant doesn't integrate with
- File management operations
- Any task that needs precise UI element interaction

## Vendor detection

Determine which voice assistant is available before attempting:

```bash
adb shell pm list packages | grep -E 'speechassist|voiceassist|xiaoai|bixby|xiaoyi|jovi'
```

| Package pattern | Vendor | Assistant |
|-----------------|--------|-----------|
| `com.heytap.speechassist` | OPPO / OnePlus | Xiaobu (小布) |
| `com.miui.voiceassist` | Xiaomi / Redmi | Xiao Ai (小爱同学) |
| `com.samsung.android.bixby.agent` | Samsung | Bixby |
| `com.huawei.vassistant` | Huawei | Xiaoyi (小艺) |
| `com.vivo.ai.assistant` | vivo | Jovi |

If no known package is found, skip voice assistant and try the next priority level.

## Generic send-button helper

All vendors require a "tap send" step after inputting text. Use this reusable template — parameterize `RESOURCE_ID` per vendor:

```bash
# Tap a button by resource-id using uiautomator dump
tap_by_id() {
  local RID="$1"
  local RETRIES="${2:-10}"
  local XML=/sdcard/ui.xml
  for i in $(seq 1 "$RETRIES"); do
    adb shell uiautomator dump "$XML" >/dev/null 2>&1
    BOUNDS=$(adb shell cat "$XML" | tr -d '\r' \
      | grep "resource-id=\"$RID\"" \
      | sed -n 's/.*bounds="\[\([0-9]*\),\([0-9]*\)\]\[\([0-9]*\),\([0-9]*\)\]".*/\1 \2 \3 \4/p' \
      | head -1)
    if [ -n "$BOUNDS" ]; then
      read X1 Y1 X2 Y2 <<< "$BOUNDS"
      adb shell input tap $(((X1+X2)/2)) $(((Y1+Y2)/2))
      return 0
    fi
    sleep 0.3
  done
  return 1
}
```

**Fallback**: If the resource-id is not found, try matching by button text:

```bash
# Fallback: tap by visible text label
tap_by_text() {
  local TEXT="$1"
  local XML=/sdcard/ui.xml
  adb shell uiautomator dump "$XML" >/dev/null 2>&1
  BOUNDS=$(adb shell cat "$XML" | tr -d '\r' \
    | grep "text=\"$TEXT\"" \
    | sed -n 's/.*bounds="\[\([0-9]*\),\([0-9]*\)\]\[\([0-9]*\),\([0-9]*\)\]".*/\1 \2 \3 \4/p' \
    | head -1)
  if [ -n "$BOUNDS" ]; then
    read X1 Y1 X2 Y2 <<< "$BOUNDS"
    adb shell input tap $(((X1+X2)/2)) $(((Y1+Y2)/2))
    return 0
  fi
  return 1
}
```

---

## OPPO — Xiaobu (小布)

- **Package**: `com.heytap.speechassist`
- **Launch method**: `ACTION_PROCESS_TEXT` intent with text extra (single-step, text is passed directly)
- **Send button**: `com.heytap.speechassist:id/btn_send`
- **Wait before dump**: 0.8s
- **Tested on**: ColorOS 13+

### Command template

```bash
MSG="设置明天早上7点的闹钟"

# IMPORTANT: Use inner single quotes "'$MSG'" to preserve spaces in the
# text when passed through adb shell. Without them, Android's shell splits
# on whitespace and only the first word reaches the assistant.
adb shell am start -W -a android.intent.action.PROCESS_TEXT \
  -n com.heytap.speechassist/.sharereceive.AIChatShareReceiveActivity \
  --es android.intent.extra.PROCESS_TEXT "'$MSG'" && \
sleep 0.8 && \
B=$(adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1; \
    adb shell cat /sdcard/ui.xml | tr -d '\r' | \
    grep 'resource-id="com.heytap.speechassist:id/btn_send"' | \
    sed -n 's/.*bounds="\[\([0-9]*\),\([0-9]*\)\]\[\([0-9]*\),\([0-9]*\)\]".*/\1 \2 \3 \4/p') && \
read X1 Y1 X2 Y2 <<< "$B" && \
adb shell input tap $(((X1+X2)/2)) $(((Y1+Y2)/2))
```

### Notes

- **Quoting is critical**: `--es ... "'$MSG'"` — outer double quotes let the local shell expand `$MSG`, inner single quotes survive to the Android shell and prevent whitespace splitting. Without this, text after the first space is silently dropped.
- The `PROCESS_TEXT` action opens a chat-style interface where Xiaobu processes the text as a voice command.
- Text is passed via intent extra — no need to use `input text` separately.
- Only the send button tap requires uiautomator.

---

## Xiaomi — Xiao Ai (小爱同学)

- **Package**: `com.miui.voiceassist`
- **Launch method**: `ACTION_ASSIST` intent (opens voice mode first, must switch to text)
- **Text-mode toggle**: `com.miui.voiceassist:id/input_asr_text_view`
- **Text input field**: `com.miui.voiceassist:id/et_min_input`
- **Send button**: `com.miui.voiceassist:id/btn_text_send`
- **Tested on**: MIUI 14+

### Command template

```bash
MSG='设置明天早上7点的闹钟'
XML=/sdcard/ui.xml

# Helper: wait for a UI node by resource-id, return its center coordinates
wait_node() {
  local RID="$1"
  for i in $(seq 1 20); do
    adb shell uiautomator dump "$XML" >/dev/null 2>&1
    OUT=$(adb shell cat "$XML" | tr -d '\r' | python3 -c "
import re, sys
rid = sys.argv[1]
s = sys.stdin.read()
m = re.search(rf'resource-id=\"{re.escape(rid)}\".*?bounds=\"\[(\d+),(\d+)\]\[(\d+),(\d+)\]\"', s)
print('' if not m else f'{(int(m.group(1))+int(m.group(3)))//2} {(int(m.group(2))+int(m.group(4)))//2}')
" "$RID")
    [ -n "$OUT" ] && echo "$OUT" && return 0
    sleep 0.2
  done
  return 1
}

# 1. Launch assistant
adb shell am start -W -a android.intent.action.ASSIST

# 2. Switch to text input mode
read X Y <<< "$(wait_node com.miui.voiceassist:id/input_asr_text_view)"
adb shell input tap "$X" "$Y"
sleep 0.25

# 3. Tap text field and input text
read X Y <<< "$(wait_node com.miui.voiceassist:id/et_min_input)"
adb shell input tap "$X" "$Y"
sleep 0.2

# Encode spaces as %s for adb input text
ENCODED=$(echo "$MSG" | sed 's/ /%s/g')
adb shell input text "$ENCODED"
sleep 0.25

# 4. Tap send
read X Y <<< "$(wait_node com.miui.voiceassist:id/btn_text_send)"
adb shell input tap "$X" "$Y"
```

### Notes

- Xiao Ai launches in voice mode by default — the text-mode toggle step is required.
- The multi-step flow is more fragile than OPPO's single-intent approach.
- Chinese text with spaces needs `%s` encoding for `adb shell input text`. For complex Unicode, use ADBKeyboard if available.

---

## Adding a new vendor

Copy this template and fill in the fields:

```
## [Vendor] — [Assistant Name]

- **Package**: `com.example.assistant`
- **Launch method**: (intent action + component, or just action)
- **Text input method**: (intent extra, or separate `input text` step)
- **Send button**: (resource-id)
- **Wait before dump**: (seconds)
- **Tested on**: (OS version)

### Command template

(paste tested working command here)

### Notes

(vendor-specific quirks)
```

### Discovery steps for a new vendor

1. Find the package: `adb shell pm list packages | grep -i assistant`
2. List exported activities: `adb shell dumpsys package <PKG> | grep -A2 'exported=true'`
3. Try common intents:
   - `adb shell am start -a android.intent.action.ASSIST`
   - `adb shell am start -a android.intent.action.VOICE_ASSIST`
   - `adb shell am start -a android.intent.action.PROCESS_TEXT --es android.intent.extra.PROCESS_TEXT "test"`
4. Dump UI to find resource-ids: `adb shell uiautomator dump /sdcard/ui.xml && adb pull /sdcard/ui.xml`
5. Search for send/submit buttons: `grep -i 'send\|submit\|发送' ui.xml`

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Text after first space is silently dropped | `adb shell` splits unquoted arguments on whitespace | Wrap the value with inner single quotes: `--es ... "'$MSG'"` (outer double quotes expand the variable, inner single quotes survive to the Android shell) |
| `uiautomator dump` returns empty XML | Screen not ready | Increase sleep time before dump |
| Resource-id not found in XML | UI version changed | Try text-based fallback (`grep 'text="发送"'`); re-dump and search for the actual id |
| `Activity not found` or `unable to resolve Intent` | Assistant not installed or activity class renamed | Re-check with `dumpsys package` |
| Assistant opens but doesn't process text | Text encoding issue | Check for special characters; use `%s` for spaces in `input text` |
| Command hangs at `am start -W` | Activity never reports "Complete" | Use `am start` without `-W`, add explicit `sleep` instead |
