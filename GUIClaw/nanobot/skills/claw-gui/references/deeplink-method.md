# Android Entry-Point Probing Method

This reference captures the operating model behind the skill. Load it when the task involves OEM apps, preinstalled apps, exported components, or deep-link uncertainty.

## Core idea

Do not treat Android routing as only a deep-link problem.

For system apps and OEM packages, the most reliable path is a chained workflow:

1. Build a package profile with `dumpsys package`
2. Validate explicit and implicit launch paths with `am`
3. Watch runtime state with `dumpsys activity top`, `dumpsys activity services`, and `logcat`
4. Pivot to `content` when `Provider` authorities appear
5. Report permission/export boundaries instead of brute-forcing blocked components

`adb` is the shell entrypoint. `am`, `pm`, `dumpsys`, `content`, and `logcat` are the actual diagnostic lanes.

## The package-profile pass

Start here before broad probing:

```bash
adb shell dumpsys package <PACKAGE>
```

Extract and summarize:

- `Activity`, `Service`, `Receiver`, `Provider`
- `authority`
- `permission`, `readPermission`, `writePermission`
- `exported`
- `processName`
- `VIEW` / `BROWSABLE`
- likely scheme/host/path hints

The goal is to answer four questions:

1. What components are exposed
2. What permissions gate them
3. Whether there is a browsable routing surface
4. Whether multiple processes or providers suggest a non-UI entry path

## Validation lanes

Choose the next lane from the package profile instead of guessing.

### 1. Explicit activity lane

Use when the package profile reveals a likely exported activity:

```bash
adb shell am start -W -n <PACKAGE>/<ACTIVITY>
```

Use this to confirm the entry exists, not to infer the whole business flow.

### 2. Intent and deep-link lane

Use when you see `VIEW`, `BROWSABLE`, scheme, host, or path clues:

```bash
adb shell am start -W -a android.intent.action.VIEW -d '<URI>' <PACKAGE>
```

Prefer compact, high-probability candidates first. If a URI opens the app shell but not the target page, keep the same family and vary one path segment or parameter key at a time.

### 3. Provider lane

Use when `authority` appears:

```bash
adb shell content query --uri content://<AUTHORITY>
adb shell content query --user 0 --uri content://<AUTHORITY>
```

For note, gallery, calendar, and file-manager style apps, `Provider` access often reveals more than UI probing.

### 4. Service and receiver lane

Use when service or receiver behavior seems central:

```bash
adb shell dumpsys activity services
adb shell dumpsys activity service <SERVICE>
```

This is often where OEM apps bind sync, preload, editor, or account logic.

### 5. Runtime observation lane

Pair launch attempts with state inspection:

```bash
adb logcat -c
adb shell am start -W -n <PACKAGE>/<ACTIVITY>
adb shell dumpsys activity top
adb logcat -v time
```

Look for:

- `cmp=`
- `Permission Denial`
- `SecurityException`
- `Start proc`
- provider or resolver activity

## Fast mode

Use this when the goal is to reach a useful state quickly:

1. Build a compact package profile
2. Try a small set of high-probability custom-scheme candidates
3. Try matching `https` links if a host is known
4. Stop as soon as one route reliably reaches the target state or the app shell
5. Return:
   - package profile summary
   - viable URIs
   - partial matches
   - the next few commands worth trying

## Investigate mode

Use this when fast mode cannot explain or reach the target:

1. Expand path variants:
   - `search`
   - `search/`
   - `search/result`
   - `note/123`
   - `profile/<id>`
2. Expand parameter names:
   - `q`
   - `query`
   - `keyword`
   - `id`
   - `url`
3. Extract more from the package profile:
   - exported components
   - provider authorities
   - process names
   - permissions
4. Add targeted next commands for:
   - `am start -n`
   - `am start -W -a VIEW -d`
   - `content query`
   - `dumpsys activity top`
   - `dumpsys activity services`
   - `dumpsys package domain-preferred-apps`

## Reading outcomes

### `unable to resolve Intent`

Usually means:

- the route is not registered
- or the `scheme`, `host`, `path`, or query layout is wrong

### App opens but lands on home or fallback

Usually means:

- the app has a matching entry activity
- but the business route or parameters are not recognized

### `http/https` keeps opening the browser

Usually means:

- it is a web link rather than a validated app link
- or the device has not approved that domain for the package

### Permission denial or security exception

Usually means:

- the component is not exported
- or it requires a permission unavailable to shell or normal apps
- or the OEM added a privileged gate

### Provider query denied

Usually means:

- the provider is internal only
- or a read/write permission blocks external callers

## Multi-user reminder

Check the current user before over-interpreting failures:

```bash
adb shell am get-current-user
adb shell pm list packages --user 0 | grep <PACKAGE>
```

System apps often behave differently across owner, guest, and managed profiles.

## Main-agent hygiene

Do not send raw `dumpsys` or `logcat` output unless the user asks for it. Distill it into a few statements:

- which URI families seem real
- whether the package has a browsable routing surface
- whether a provider or service lane is more promising than UI probing
- whether exported or permission boundaries are the real blocker
