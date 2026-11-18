# Manual Dataset Setup

Use this guide if you cannot use `openmhc.download_dataset()` — for example, when the release requires a guestbook response, on air-gapped clusters, behind a corporate proxy, or when you already have the raw Dataverse bundle on disk.

The OpenMHC dataset is distributed through Harvard Dataverse. Use the [XS deposit](https://doi.org/10.7910/DVN/RIKSK4) or [Full deposit](https://doi.org/10.7910/DVN/XNBITM), subject to its terms and file permissions.

For an air-gapped machine, install on a connected machine first (using this guide
or `openmhc.download_dataset` when the release permits API downloads), then
transfer the complete installed root and verify the transferred files.

## Releases requiring a guestbook response

If the helper raises `openmhc.DatasetGuestbookRequiredError`, open the XS or Full
deposit linked above in your browser. Complete its required guestbook form and
download the original release files through the website. Obtain permission for
any restricted files as well. Use the inventory checks below to confirm that
you have every file, including all multipart segments, before extraction.

The helper does not collect or submit guestbook answers. Completing the form in
a browser does not authorize its subsequent API GET requests, so proceed with
manual installation rather than retrying the helper. Dataverse's
[API guestbook flow](https://guides.dataverse.org/en/latest/api/dataaccess.html#basic-file-access)
requires submitting answers with the download request.

## Obtain a complete release inventory

Dataverse's dataset-ZIP endpoint can omit restricted files or files exceeding the server's ZIP size limit. A successful ZIP request does not establish completeness. Use the latest published inventory, never an editable draft:

```bash
curl -f -H "X-Dataverse-key: <your-token>" \
  "https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest-published?persistentId=doi:10.7910/DVN/RIKSK4" \
  -o release-inventory.json
```

For Full, replace `RIKSK4` with `XNBITM`. Omit the token header when no authentication is needed. For every entry in `data.files`, obtain the original file through the website if a guestbook response is required; otherwise download `/api/access/datafile/<dataFile.id>?format=original`. Place each file at its `directoryLabel/label` path. Check its byte count against `dataFile.filesize` and its checksum against `dataFile.checksum`. All entries must be present and verified, including every multipart segment. Request access if a file is restricted; do not continue with only accessible files.

Keep this inventory for the entire installation. If you already have a ZIP bundle, compare its complete extracted file inventory, sizes and checksums with that same release inventory before unpacking inner archives. Do not forward API tokens to another host when following download redirects.

**One root per version.** Every OpenMHC dataset root must contain a `dataset_version.json` marker declaring which release lives there (`xs` or `full`), and the public eval API cross-checks that marker against the `version=` you pass. The two therefore cannot share a directory — put each under its own root and point `MHC_DATA_DIR` (or `data_dir=`) at whichever one you want to use.

Extract each complete release into a new or empty directory. For replacing an
existing root, see [migration guidance](../DATASET.md#existing-dataset-roots).

Include the release's precomputed imputation masks. Copying `data/imputation/`
below installs them at `<dataset_root>/imputation/masks/`, alongside `processed/`,
`labels/`, and `splits/`. Use the imputation masks included with the dataset release;
see [mask paths and requirements](../DATASET.md#bundled-vs-dataset-root-files).

Recommended layout for keeping both side by side:

```
~/.cache/openmhc/
├── data-xs/      # 593-user development subset
└── data-full/    # 11,894-user paper release
```

---

## Development subset (~1.9 GB, 593 users)

**Source:** [10.7910/DVN/RIKSK4](https://doi.org/10.7910/DVN/RIKSK4)

### Step 1 — Download and verify every release file

Follow the inventory procedure above, preserving directory labels. Keep the packaged `dataset_version.json` with the files.

### Step 2 — Prepare the release tree

```bash
CACHE="$HOME/.cache/openmhc/data-xs"
mkdir -p "$CACHE"
```

Place the verified release files under `$CACHE`, keeping `archives/` and `data/`. If using a previously downloaded and independently verified ZIP, extract it here first. Do not mix releases or overwrite an existing installation.

The release tree contains an `archives/` directory with five tar.gz files and a `data/` tree with metadata.

### Step 3 — Copy metadata into the cache root

```bash
cp -r "$CACHE/data/splits"                    "$CACHE/"
cp -r "$CACHE/data/labels"                    "$CACHE/"
cp -r "$CACHE/data/forecasting_sample_index"  "$CACHE/"
cp -r "$CACHE/data/imputation"               "$CACHE/"
mkdir -p "$CACHE/processed"
cp    "$CACHE/data/processed/"*.json          "$CACHE/processed/"
cp    "$CACHE/data/processed/"*.parquet       "$CACHE/processed/"
```

### Step 4 — Extract the HuggingFace Arrow datasets

The XS archives unpack straight to their target directory names (`daily_hf`,
`daily_hourly_hf`, `hourly_trajectory`, `minute_trajectory`), so no renaming is
needed.

```bash
# Daily 1440-min tensors (Track 2 — imputation)
tar -xzf "$CACHE/archives/daily_hf_xs.tar.gz" -C "$CACHE/processed/"

# Daily 24-hour tensors (Track 1 — outcome prediction)
tar -xzf "$CACHE/archives/daily_hourly_hf_xs.tar.gz" -C "$CACHE/processed/"

# Hourly per-user trajectories (Track 3 — forecasting)
tar -xzf "$CACHE/archives/hourly_trajectory_xs.tar.gz" -C "$CACHE/"

# Per-minute per-user trajectories (Track 3 — optional high-res input)
tar -xzf "$CACHE/archives/minute_trajectory_xs.tar.gz" -C "$CACHE/"
```

### Step 5 — Extract the HDF5 archive

```bash
mkdir -p "$CACHE/hdf5"
tar -xzf "$CACHE/archives/hdf5_sharable_2026_xs.tar.gz" -C "$CACHE/"
```

### Step 6 — Move normalization stats

If `normalization_stats.json` landed at the cache root rather than in `processed/`:

```bash
[ -f "$CACHE/normalization_stats.json" ] && \
  mv "$CACHE/normalization_stats.json" "$CACHE/processed/"
```

### Step 7 — Check the packaged dataset marker

```bash
python -c "import openmhc; print(openmhc.read_dataset_marker('$CACHE'))"
```

The marker must declare `version="xs"`, `n_users=593` and retain its supplied revision. If packaged under `data/`, move it to the root first; reject conflicting root and nested markers. Do not replace a missing release marker with a guessed identity.

---

## Full version (~38 GB, 11,894 users)

**Source:** [10.7910/DVN/XNBITM](https://doi.org/10.7910/DVN/XNBITM)

The full bundle splits large archives into numbered parts (`*.tar.gz.part-NN`) that must be concatenated before extraction. All directories extract to canonical names — no renaming required.

### Step 1 — Download and verify every release file

Follow the inventory procedure above, preserving directory labels. Keep the packaged `dataset_version.json` with the files.

### Step 2 — Prepare the release tree

```bash
CACHE="$HOME/.cache/openmhc/data-full"
mkdir -p "$CACHE"
```

Place the verified release files under `$CACHE`, keeping `archives/` and `data/`. If using a previously downloaded and independently verified ZIP, extract it here first. Do not mix releases or overwrite an existing installation.

### Step 3 — Copy metadata into the cache root

```bash
cp -r "$CACHE/data/splits"                    "$CACHE/"
cp -r "$CACHE/data/labels"                    "$CACHE/"
cp -r "$CACHE/data/forecasting_sample_index"  "$CACHE/"
cp -r "$CACHE/data/imputation"               "$CACHE/"
mkdir -p "$CACHE/processed"
cp    "$CACHE/data/processed/"*.json          "$CACHE/processed/"
cp    "$CACHE/data/processed/"*.parquet       "$CACHE/processed/"
```

### Step 4 — Extract multi-part archives

Each `cat ... | tar -xz` command concatenates the numbered parts and streams them directly into tar — no intermediate file needed.

```bash
# Daily 1440-min tensors — 5 parts → processed/daily_hf/
cat "$CACHE/archives/daily_hf_full.tar.gz.part-"* | tar -xz -C "$CACHE/processed/"

# Daily 24-hour tensors — single archive → processed/daily_hourly_hf/
tar -xzf "$CACHE/archives/daily_hourly_hf_full.tar.gz" -C "$CACHE/processed/"

# HDF5 archives — 5 parts → hdf5/
mkdir -p "$CACHE/hdf5"
cat "$CACHE/archives/hdf5_sharable_2026_full.tar.gz.part-"* | tar -xz -C "$CACHE/hdf5/"

# Hourly trajectories — single archive → hourly_trajectory/
tar -xzf "$CACHE/archives/hourly_trajectory_full.tar.gz" -C "$CACHE/"

# Per-minute trajectories — 10 parts → minute_trajectory/  (optional;
# not used by the public evaluate_* API — safe to skip if disk is tight)
cat "$CACHE/archives/minute_trajectory_full.tar.gz.part-"* | tar -xz -C "$CACHE/"
```

Extracted footprint is roughly 100–500 GB depending on whether you include `minute_trajectory/`. Only `daily_hf/`, `daily_hourly_hf/`, and `hourly_trajectory/` are read by the public eval API; the others mirror the canonical Dataverse layout.

### Step 5 — Check the packaged dataset marker

```bash
python -c "import openmhc; print(openmhc.read_dataset_marker('$CACHE'))"
```

The marker must declare `version="full"`, `n_users=11894` and retain its supplied revision. Promote a marker under `data/` only after checking that it does not conflict with a root marker.

If you want the marker to also pin the row count of `processed/daily_hf` (cheap extra check the resolver will validate):

```bash
python -c "
import openmhc
from datasets import load_from_disk
root = '$CACHE'
n = len(load_from_disk(f'{root}/processed/daily_hf'))
openmhc.write_dataset_marker(root, version='full', daily_hf_rows=n)
"
```

---

## Expected layout after setup

After either version, the chosen cache root should look like:

```
$CACHE/
├── dataset_version.json                       # marker — REQUIRED
├── labels/
│   ├── last_labels.json
│   ├── context_labels.json
│   ├── label_types.json                       # REQUIRED dataset schema
│   ├── ordinal_dictionary.json                # REQUIRED dataset schema
│   ├── validity_config.json                   # REQUIRED dataset schema
│   ├── label_validity.json
│   ├── clip_dates.json
│   └── ...
├── splits/
│   └── sharable_users_seed42_2026.json        # full root
│       (or sharable_users_seed42_2026_xs.json  for xs root)
├── processed/
│   ├── daily_hf/                              # Track 2 — 1440-min tensors (HuggingFace Arrow)
│   ├── daily_hourly_hf/                       # Track 1 — 24-hour tensors (HuggingFace Arrow)
│   ├── normalization_stats.json
│   ├── normalization_stats_hourly.json
│   ├── window_index_w7_s7_d5.parquet
│   ├── daily_labels_lookup.parquet
│   ├── daily_labels_lookup_full_history.parquet
│   └── weekly_labels_lookup_stride7_full_history.parquet
├── imputation/masks/                          # Track 2 — matching release masks
├── forecasting_sample_index/                  # Track 3
│   ├── sample_index_P_24_M_H_7_3_S_100.json   # used by evaluate_forecasting (horizon 24)
│   ├── day_remain_mask.json
│   └── ...
├── hourly_trajectory/                         # Track 3
├── minute_trajectory/                         # Track 3 — optional
└── hdf5/                                      # HDF5 copies of sensor data
```

## Pointing the API at your data directory

Point `MHC_DATA_DIR` at the version you want to run against:

```bash
export MHC_DATA_DIR=~/.cache/openmhc/data-xs    # for xs eval / quickstart
export MHC_DATA_DIR=~/.cache/openmhc/data-full  # for full / leaderboard
```

Or pass it explicitly per call (and remember to pass `version=`, which the API now requires):

```python
import openmhc
results = openmhc.evaluate_imputation(
    MyImputer(),
    version="full",
    data_dir="~/.cache/openmhc/data-full",
)
```

`openmhc.data_dir()` returns the explicit root resolved from `data_dir=` or `MHC_DATA_DIR`, so you can verify the lookup:

```python
import openmhc
print(openmhc.data_dir())                                # → /path/to/your/data
print(openmhc.read_dataset_marker(openmhc.data_dir()))   # → {'version': 'full', ...}
```
