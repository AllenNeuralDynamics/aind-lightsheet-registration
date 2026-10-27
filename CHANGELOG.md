# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.6.1 (2026-10-27)

### Fix

- **register**: read the tissue mask into an array the guard owns

## v0.6.0 (2026-10-27)

### Feat

- **frames**: move pipeline-frame images into the Zarr frame
- **register**: match labeled structures in SyN as extra metrics

## v0.5.0 (2026-10-26)

### BREAKING CHANGE

- pyramid.level_center_offset is removed.

### Fix

- **reader**: let aind-zarr-utils place the pyramid level

## v0.4.0 (2026-10-22)

### Feat

- correct the olfactory-bulb shadow per brain against the template
- divide out the dips at the focus-ramp planes
- fit a z-y shear per illumination side before stitching
- divide out the tiles' common row profile before measuring overlaps
- search tile overlaps over a wide window by FFT and drop contradicted pairs

### Fix

- return a typed array from tile_row_profile

## v0.3.1 (2026-10-21)

## v0.3.0 (2026-10-21)

### Feat

- smooth the total SyN field by one voxel by default
- refuse brains whose metadata names rotated axis directions
- normalize brains to the convention their template declares
- register at 50 um with the template-build schedule by default
- store the seam profile as TSV with a JSON sidecar
- add a calibrated tile-row seam profile
- fit a brightness gain per tile in the sheet illumination model
- skip illumination correction by default and stitch grids with empty tiles
- correct, stitch and fuse raw SmartSPIM tiles before registration
- prepare a brain from its zarr and register it to a template
- threshold brain masks on log intensity by default

### Fix

- keep images alive while reading their arrays
- start N4 from a coarse mesh so the bias field cannot follow anatomy

## v0.2.0 (2026-10-16)

### Feat

- read a light-sheet pyramid level and prepare it for registration
