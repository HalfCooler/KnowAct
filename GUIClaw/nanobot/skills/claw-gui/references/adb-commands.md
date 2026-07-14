# ADB Quick Command Catalog

Reference catalog of common `adb shell` commands for Android device control. All commands are prefixed with `adb shell` when run from a host machine.

## Connectivity

| Action | Command |
|--------|---------|
| WiFi on | `svc wifi enable` |
| WiFi off | `svc wifi disable` |
| Bluetooth on | `svc bluetooth enable` |
| Bluetooth off | `svc bluetooth disable` |
| Airplane mode on | `cmd connectivity airplane-mode enable` |
| Airplane mode off | `cmd connectivity airplane-mode disable` |
| Mobile data on | `svc data enable` |
| Mobile data off | `svc data disable` |

Alternative WiFi commands: `cmd wifi set-wifi-enabled enabled/disabled`
Alternative Bluetooth commands: `cmd bluetooth_manager enable/disable`
Alternative mobile data: `cmd phone data enable/disable`

## Display

| Action | Command |
|--------|---------|
| Set brightness (manual) | `settings put system screen_brightness <value>` |
| Auto-brightness on | `settings put system screen_brightness_mode 1` |
| Auto-brightness off | `settings put system screen_brightness_mode 0` |
| Dark mode on | `cmd uimode night yes` |
| Dark mode off | `cmd uimode night no` |
| Night display on | `settings put secure night_display_activated 1` |
| Night display off | `settings put secure night_display_activated 0` |
| Night display schedule | `settings put secure night_display_custom_start_time 79200000` / `night_display_custom_end_time 25200000` |
| Auto-rotate on | `settings put system accelerometer_rotation 1` |
| Auto-rotate off | `settings put system accelerometer_rotation 0` |

Note: brightness range varies by OEM. Stock Android uses 0-255. OPPO/OnePlus may use 0-4096. Disable auto-brightness before setting manual brightness.

## Notifications and Panels

| Action | Command |
|--------|---------|
| Open notification panel | `cmd statusbar expand-notifications` |
| Open control center / quick settings | `cmd statusbar expand-settings` |
| Collapse panels | `cmd statusbar collapse` |
| DND on | `cmd notification set_dnd on` |
| DND off | `cmd notification set_dnd off` |

## Power

| Action | Command |
|--------|---------|
| Battery saver on | `cmd power set-mode 1` |
| Battery saver off | `cmd power set-mode 0` |

## Location

| Action | Command |
|--------|---------|
| Location on | `cmd location set-location-enabled true` |
| Location off | `cmd location set-location-enabled false` |
| Check location status | `cmd location is-location-enabled` |

## Settings Navigation

Open specific settings pages via `am start`:

| Page | Command |
|------|---------|
| WiFi settings panel | `am start -a android.settings.panel.action.WIFI` |
| Internet panel (WiFi + mobile) | `am start -a android.settings.panel.action.INTERNET_CONNECTIVITY` |
| Bluetooth settings | `am start -a android.settings.BLUETOOTH_SETTINGS` |
| Airplane mode settings | `am start -a android.settings.AIRPLANE_MODE_SETTINGS` |
| Battery saver settings | `am start -a android.settings.BATTERY_SAVER_SETTINGS` |

## Other Navigation

| Action | Command |
|--------|---------|
| Open app drawer | `input keyevent 284` (must be on home screen) |
| Global search | `am start -a android.search.action.GLOBAL_SEARCH` |

## Tips

- `cmd statusbar` commands require API level 30+ (Android 11).
- For multi-user devices, check the active user first: `am get-current-user`.
- Some OEM skins rename or gate commands behind additional permissions.
- When targeting a specific device, prefix all commands with `adb -s <serial> shell`.
