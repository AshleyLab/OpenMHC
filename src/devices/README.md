# Devices API

Per-user device-timeline accessor. Returns the iPhone and Apple Watch a
participant was using at any given timestamp.

## Public surface

- `devices.get_devices(health_code, timestamp) -> DeviceSnapshot` — point-in-time lookup
- `devices.get_device_timeline(health_code) -> DeviceTimeline` — full interval history
- `devices.api.configure(data_dir=None)` — select device data and clear cached timelines
- `devices.DeviceInterval` — one `[start, end, name]` row
- `devices.DeviceTimeline` — sorted, non-overlapping intervals for phone and watch
- `devices.DeviceSnapshot` — the resolved phone and watch at one timestamp

## Usage

```python
import pandas as pd
from devices import api as devices_api
from devices import get_devices, get_device_timeline

devices_api.configure(data_dir="/path/to/dataset")
snap = get_devices("user-123", pd.Timestamp("2020-06-01"))
if snap.phone is not None:
    print(snap.phone.name)       # 'iPhone 8 Plus'
if snap.watch is not None:
    print(snap.watch.name)       # 'Apple Watch Series 5 44mm GPS'

timeline = get_device_timeline("user-123")
for iv in timeline.phone:
    print(iv.start.date(), iv.end.date(), iv.name)
```

Either field may be `None` if no interval covers the query timestamp
(before the user's first record, or in a gap between devices). Query
timestamps must be tz-naive.

Query dates must use the dataset's timeline. With shifted data, the returned
device is aligned with the shifted sensor dates; it does not describe device use
on that date in the original calendar.

## Data source

Reads `<dataset_root>/labels/user_device_info.json`. Select the root with
`devices.api.configure(data_dir=...)` or `MHC_DATA_DIR`. An explicit
`USER_DEVICE_INFO_PATH` environment variable overrides that default.

The first query loads and caches the timelines. Call `configure()` after changing
environment overrides or selecting another dataset to clear the cache. Evaluation
entry points configure the device API automatically without reading the file.

The data file ships separately from `last_labels.json` and is not
guaranteed to be in every Dataverse bundle. Releases may intentionally omit it.
Importing and configuring `devices` works without the file; the first lookup
raises `FileNotFoundError` if the configured file is missing. An unset path raises
`ValueError` with configuration guidance.

## Schema

```json
{
  "<healthCode>": {
    "phone": [["2017-12-18", "2023-05-18", "iPhone 8 Plus"], ...],
    "watch": [["2019-04-05", "2023-12-19", "Apple Watch Series 5 44mm GPS"], ...]
  }
}
```

Intervals are inclusive on both ends, contiguous-run (calendar gaps inside
a same-model run are absorbed upstream), non-overlapping, and sorted by
start date. Unrecognised device strings fall through to
`iPhone Unknown Model` / `Apple Watch Unknown Model`; roughly 85% of the
Unknown bucket reflects user-renamed Apple devices in iOS Settings rather
than third-party hardware.

## Running tests

```bash
pytest src/devices/test_api.py
```
