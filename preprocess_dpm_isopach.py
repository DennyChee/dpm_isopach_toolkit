#!/usr/bin/env python
# Usage:       python preprocess_dpm_isopach.py --dpm dpm.tif --isopach isopach_thickness_mm.tif \
#                  --edges 0.1 1 5 10 50 100 200 500 --samples_per_class 2000 --output dpm_isopach.csv
# Description: Turns a DPM GeoTIFF (0-1) and an ash-thickness GeoTIFF (mm) into the 2-column CSV
#              that fit_thickness_from_dpm.m expects (column 1 = dpm, column 2 = thickness_mm).
#              The thickness raster is resampled onto the DPM grid, and the same number of pixels
#              is randomly drawn from every isopach class (set by --edges) so that no class
#              dominates the fit. Rasters are read block by block, never whole.
# Date:        2026-10-02

import argparse
import os
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def parse_arguments():
    """Read command-line options. Returns an argparse.Namespace."""
    parser = argparse.ArgumentParser(description="Build the DPM-vs-thickness CSV for fit_thickness_from_dpm.m")
    parser.add_argument("--dpm", required=True, help="DPM GeoTIFF, single band, values 0-1 (see --dpm_divide_by)")
    parser.add_argument("--isopach", required=True, help="Ash thickness GeoTIFF in mm, any grid, CRS must be defined")
    parser.add_argument("--edges", required=True, type=float, nargs="+",
                        help="Isopach class edges in mm (used to balance the sample; pass the SAME edges to the MATLAB "
                             "function), increasing, all > 0, e.g. 0.1 1 5 10 50 100 200 500")
    parser.add_argument("--samples_per_class", type=int, default=2000,
                        help="Pixels drawn per class (classes with fewer pixels keep all of them)")
    parser.add_argument("--dpm_divide_by", type=float, default=1.0,
                        help="Divide DPM values by this to reach 0-1 (e.g. 100 for percent, 255 for 8-bit)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed so the sample is reproducible")
    parser.add_argument("--output", default="dpm_isopach.csv", help="Output CSV path")
    return parser.parse_args()


def read_valid_block(dpm_dataset, thickness_dataset, window, dpm_divide_by, edges_mm):
    """Read one block of both rasters and return (dpm_values, thickness_mm, class_index) for the valid pixels.

    Inputs: open rasterio datasets (thickness already on the DPM grid), a rasterio window,
            the DPM scale divisor (dimensionless), class edges (mm).
    Output: dpm_values (float64, 0-1), thickness_mm (float64) and class_index (int, 0..number_of_classes-1),
            all the same length.
            Pixels that are masked/NaN in either raster, or outside [lowest edge, highest edge], are dropped.
    """
    dpm_block = dpm_dataset.read(1, window=window, masked=True).astype("float64").filled(np.nan) / dpm_divide_by
    thickness_block_mm = thickness_dataset.read(1, window=window, masked=True).astype("float64").filled(np.nan)
    valid_mask = (np.isfinite(dpm_block) & np.isfinite(thickness_block_mm)
                  & (thickness_block_mm >= edges_mm[0]) & (thickness_block_mm <= edges_mm[-1]))   # last edge included, as in MATLAB discretize
    dpm_values = dpm_block[valid_mask]
    thickness_values_mm = thickness_block_mm[valid_mask]
    # digitize returns 1..number_of_classes for values inside the edges; subtract 1 for a 0-based index
    class_index = np.digitize(thickness_block_mm[valid_mask], edges_mm) - 1
    # A value exactly on the last edge would get index number_of_classes; put it in the last class
    class_index = np.minimum(class_index, len(edges_mm) - 2)
    return dpm_values, thickness_values_mm, class_index


def main():
    arguments = parse_arguments()
    edges_mm = np.array(arguments.edges, dtype="float64")

    # --- Validate inputs ---
    print("[1/4] Checking inputs...")
    for input_file in (arguments.dpm, arguments.isopach):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
    if np.any(edges_mm <= 0) or np.any(np.diff(edges_mm) <= 0):
        raise ValueError("--edges must be increasing and all > 0 (use 0.1, not 0, for the lowest edge)")
    number_of_classes = len(edges_mm) - 1
    random_generator = np.random.default_rng(arguments.seed)

    with rasterio.open(arguments.dpm) as dpm_dataset, rasterio.open(arguments.isopach) as isopach_dataset:
        if dpm_dataset.crs is None or isopach_dataset.crs is None:
            raise ValueError("Both rasters need a defined CRS (check with gdalinfo)")
        print(f"  DPM:     shape={dpm_dataset.height}x{dpm_dataset.width}, dtype={dpm_dataset.dtypes[0]}, crs={dpm_dataset.crs}")
        print(f"  Isopach: shape={isopach_dataset.height}x{isopach_dataset.width}, dtype={isopach_dataset.dtypes[0]}, crs={isopach_dataset.crs}")

        # Warp the thickness raster onto the DPM grid lazily (nearest neighbour avoids blending with nodata)
        with WarpedVRT(isopach_dataset, crs=dpm_dataset.crs, transform=dpm_dataset.transform,
                       width=dpm_dataset.width, height=dpm_dataset.height,
                       resampling=Resampling.nearest) as thickness_on_dpm_grid:

            # --- Pass 1: count valid pixels per class and check the DPM range ---
            print("[2/4] Pass 1: counting valid pixels per isopach class...")
            t0 = time.time()
            pixel_count_per_class = np.zeros(number_of_classes, dtype="int64")
            dpm_minimum, dpm_maximum = np.inf, -np.inf
            for _, window in dpm_dataset.block_windows(1):
                dpm_values, thickness_values_mm, class_index = read_valid_block(dpm_dataset, thickness_on_dpm_grid, window,
                                                           arguments.dpm_divide_by, edges_mm)
                if dpm_values.size == 0:
                    continue
                pixel_count_per_class += np.bincount(class_index, minlength=number_of_classes)
                dpm_minimum = min(dpm_minimum, dpm_values.min())
                dpm_maximum = max(dpm_maximum, dpm_values.max())
            print(f"  Done in {time.time() - t0:.1f}s")
            if pixel_count_per_class.sum() == 0:
                raise ValueError("No valid pixels found: check that the rasters overlap and the thickness units are mm")
            if dpm_minimum < 0 or dpm_maximum > 1:
                raise ValueError(f"DPM range after scaling is [{dpm_minimum:.3g}, {dpm_maximum:.3g}], expected 0-1. "
                                 "Use --dpm_divide_by (e.g. 100 or 255).")
            for class_number in range(number_of_classes):
                print(f"  class {edges_mm[class_number]:g}-{edges_mm[class_number + 1]:g} mm: "
                      f"{pixel_count_per_class[class_number]} valid pixels")

            # Probability of keeping a pixel so that each class yields about samples_per_class pixels
            keep_probability_per_class = np.minimum(
                1.0, arguments.samples_per_class / np.maximum(pixel_count_per_class, 1))

            # --- Pass 2: draw the random sample ---
            print("[3/4] Pass 2: drawing random sample...")
            t0 = time.time()
            sampled_dpm, sampled_thickness_mm = [], []
            for _, window in dpm_dataset.block_windows(1):
                dpm_values, thickness_values_mm, class_index = read_valid_block(dpm_dataset, thickness_on_dpm_grid, window,
                                                           arguments.dpm_divide_by, edges_mm)
                if dpm_values.size == 0:
                    continue
                keep_mask = random_generator.random(dpm_values.size) < keep_probability_per_class[class_index]
                sampled_dpm.append(dpm_values[keep_mask])
                sampled_thickness_mm.append(thickness_values_mm[keep_mask])
            print(f"  Done in {time.time() - t0:.1f}s")

    # --- Write CSV ---
    print(f"[4/4] Writing {arguments.output}...")
    sampled_dpm = np.concatenate(sampled_dpm)
    sampled_thickness_mm = np.concatenate(sampled_thickness_mm)
    print(f"  Sampled: {sampled_dpm.size} rows, dtype={sampled_dpm.dtype}")
    output_array = np.column_stack([sampled_dpm, sampled_thickness_mm])   # column 1 = dpm, column 2 = thickness
    np.savetxt(arguments.output, output_array, delimiter=",", fmt=["%.5f", "%.4f"],
               header="dpm,thickness_mm", comments="")
    print(f"Done. Output saved to {arguments.output}")


if __name__ == "__main__":
    main()
