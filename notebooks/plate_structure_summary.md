# Microscopy Data Plate Structure Summary

## Plates and Channel Structure

Based on our census of all plates, here's a summary of the channel structure:

| Plate | Channel Directory Structure | Available Channels |
|-------|---------------------------|-------------------|
| 20200202_6W-LaC024A | DAPI-GFP-A594-AF750 | DAPI-GFP, A594, AF750 |
| 20200202_6W-LaC024B | A594-AF750 | A594, AF750 |
| 20200202_6W-LaC024C | A594-AF750 | A594, AF750 |
| 20200202_6W-LaC024D | DAPI-GFP-A594-AF750 | DAPI-GFP, A594, AF750 |
| 20200202_6W-LaC024E | DAPI-GFP-A594-AF750 | DAPI-GFP, A594, AF750 |
| 20200202_6W-LaC024F | DAPI-GFP-A594-AF750 | DAPI-GFP, A594, AF750 |
| 20200206_6W-LaC025A | DAPI-GFP-A594-AF750 | DAPI-GFP, A594, AF750 |
| 20200206_6W-LaC025B | DAPI-GFP-A594-AF750 | DAPI-GFP, A594, AF750 |

## File Naming Pattern

There are two main file naming patterns based on the directory structure:

### DAPI-GFP-A594-AF750 structure:
- DAPI-GFP channel: `20X_DAPI-GFP-A594-AF750_[well]_DAPI-GFP_Site-[site].tif`
- A594 channel: `20X_DAPI-GFP-A594-AF750_[well]_A594_Site-[site].tif`
- AF750 channel: `20X_DAPI-GFP-A594-AF750_[well]_AF750_Site-[site].tif`

### A594-AF750 structure:
- A594 channel: `20X_A594-AF750_[well]_A594_Site-[site].tif`
- AF750 channel: `20X_A594-AF750_[well]_AF750_Site-[site].tif`

## Channel File Properties

- **DAPI-GFP** files: ~140MB each, contain two channels stacked vertically in a single file
- **A594** files: ~70MB each, single channel
- **AF750** files: ~70MB each, single channel

## Special Notes

1. **DAPI-GFP combined format**: The DAPI and GFP channels are combined into a single TIFF file, with the image height being twice the normal height (the top half is DAPI, the bottom half is GFP).

2. **Structure prediction**: You can predict the directory structure by looking at the plate name:
   - Plates 20200202_6W-LaC024B and 20200202_6W-LaC024C only have A594-AF750 channels
   - All other plates have all channels (DAPI-GFP, A594, AF750)

3. **Missing files**: Even within a directory structure, some specific site files might be missing. Always check for file existence before attempting to load.

## Using the Utility Functions

The utility functions in `file_structure_utils.py` handle all of these variations automatically. Key functions:

- `analyze_plate_structure()`: Analyze the overall structure of plates
- `get_available_channels(plate, well, site)`: Find which channels are available for a specific location
- `get_image_path(plate, well, site, channel)`: Get the correct file path for any channel
- `extract_cell_from_site(plate, well, site, cell_bounds)`: Extract a cell from any plate structure
