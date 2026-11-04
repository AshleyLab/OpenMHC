"""Extract XS masks from the matching Full dataset without generating new masks.

    MHC_DATA_DIR=/path/to/xs python scripts/precompute_xs_masks.py \
        --full-data-dir /path/to/full

Both datasets must use the same date variant. The output defaults to the XS
root's imputation/masks directory; existing files are never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from migrate_imputation_masks import remap_masks, sample_keys, subset_masks

from imputation_evaluation.masking.generator import ScenarioMasks
from openmhc._constants import MASKING_SCENARIOS
from openmhc._evaluate import _dataset_masks_dir, _DatasetPaths


def main() -> None:
    """Select Full mask rows and map them to the XS dataset's sample indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="XS root (otherwise MHC_DATA_DIR)")
    parser.add_argument("--output", type=Path, help="New directory for the extracted XS masks")
    args = parser.parse_args()
    full = _DatasetPaths.resolve(args.full_data_dir, version="full")
    xs = _DatasetPaths.resolve(args.data_dir, version="xs")
    output = args.output or _dataset_masks_dir(xs.root, "xs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing release masks: {output}")
    full_keys, xs_keys = sample_keys(full.root, "full"), sample_keys(xs.root, "xs")
    prepared = {}
    for split in ("val", "test"):
        if not set(xs_keys[split]) <= set(full_keys[split]):
            raise ValueError("XS samples are not a subset of Full; check the release/date variant")
        for scenario in MASKING_SCENARIOS:
            relative = Path(split) / f"{scenario}.npz"
            source = ScenarioMasks.load(_dataset_masks_dir(full.root, "full") / relative)
            # Check the source identities before extracting any rows.
            source = remap_masks(source, full_keys[split], full_keys[split])
            prepared[relative] = subset_masks(source, xs_keys[split])
    output.mkdir(parents=True)
    for relative, masks in prepared.items():
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        masks.save(path)
    print(f"Extracted {len(prepared)} XS mask files from Full into {output}")


if __name__ == "__main__":
    main()
