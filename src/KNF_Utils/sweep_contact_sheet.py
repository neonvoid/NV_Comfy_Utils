"""
NV Sweep Contact Sheet Nodes

Two-node system for creating visual contact sheet previews from parameter sweep outputs.

1. NV_SweepContactSheetPlanner - Analyzes sweep plan, generates grid layout JSON
2. NV_SweepContactSheetBuilder - Loads images/videos, assembles grid with headers/labels

Use Case:
- Visually compare all parameter sweep results in an organized grid
- Automatic row/column assignment based on parameter ranges
- Support for 3+ params via pagination (multiple sheets)
- Headers show param values along axes
- Labels below each cell
- Placeholder for missing files
- Supports both images and video files (extracts frame for thumbnail)
"""

import json
import os
import glob
import re
from datetime import datetime
from fractions import Fraction
from itertools import product
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

try:
    import comfy.utils
    HAS_COMFY_UTILS = True
except ImportError:
    HAS_COMFY_UTILS = False

# Video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff'}


def extract_video_frame(video_path: str, position: str = "first") -> Image.Image:
    """
    Extract a frame from a video file.

    Args:
        video_path: Path to video file
        position: "first", "middle", or "last"

    Returns:
        PIL Image of the extracted frame, or None if failed
    """
    try:
        import cv2
    except ImportError:
        print("[NV_SweepContactSheetBuilder] Warning: OpenCV not available, trying imageio...")
        try:
            import imageio.v3 as iio
            # Use imageio as fallback
            try:
                frames = iio.imread(video_path, plugin="pyav")
                if len(frames) == 0:
                    return None

                if position == "first":
                    frame_idx = 0
                elif position == "last":
                    frame_idx = len(frames) - 1
                else:  # middle
                    frame_idx = len(frames) // 2

                frame = frames[frame_idx]
                return Image.fromarray(frame)
            except Exception as e:
                print(f"[NV_SweepContactSheetBuilder] imageio failed: {e}")
                return None
        except ImportError:
            print("[NV_SweepContactSheetBuilder] Warning: Neither OpenCV nor imageio available for video")
            return None

    # Use OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        if position == "first":
            frame_idx = 0
        elif position == "last":
            frame_idx = max(0, total_frames - 1)
        else:  # middle
            frame_idx = total_frames // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            # Fallback to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)

        return None
    finally:
        cap.release()


def is_video_file(path: str) -> bool:
    """Check if file is a video based on extension."""
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def is_image_file(path: str) -> bool:
    """Check if file is an image based on extension."""
    ext = os.path.splitext(path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def get_font(font_size: int):
    """Get a font with fallback options."""
    font_names = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "FreeSans.ttf",
    ]

    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, font_size)
        except (OSError, IOError):
            continue

    # Last resort: default font
    try:
        return ImageFont.load_default()
    except:
        return None


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    elif len(hex_color) == 8:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
    return (128, 128, 128)  # Default gray


def get_unique_path(path: str) -> str:
    """Generate unique path if file exists."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


class NV_SweepContactSheetPlanner:
    """
    Analyzes a sweep plan and generates a grid layout JSON for building contact sheets.

    Automatically assigns parameters to rows/columns based on their ranges:
    - 1 param: Single row with columns
    - 2 params: Larger range → columns, smaller → rows
    - 3+ params: Two largest → grid axes, rest → pagination (multiple sheets)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sweep_json_path": ("STRING", {
                    "default": "sweep_plan.json",
                    "tooltip": "Path to the sweep_plan.json file from NV_SweepPlanner"
                }),
            },
            "optional": {
                "row_parameter": ("STRING", {
                    "default": "auto",
                    "tooltip": "Param name for rows, or 'auto' for best fit"
                }),
                "col_parameter": ("STRING", {
                    "default": "auto",
                    "tooltip": "Param name for columns, or 'auto' for best fit"
                }),
                "max_rows_per_sheet": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "tooltip": "Max rows per contact sheet (0 = unlimited). E.g., 3 for 3xN grids per page."
                }),
                "max_cols_per_sheet": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "tooltip": "Max columns per contact sheet (0 = unlimited). E.g., 3 for Nx3 grids per page."
                }),
                "output_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Where to save layout JSON. Empty = same folder as sweep plan."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("layout_json_path", "grid_summary", "total_cells", "total_sheets")
    FUNCTION = "plan_layout"
    CATEGORY = "NV_Utils/Sweep"
    DESCRIPTION = "Analyzes sweep plan and generates grid layout JSON for contact sheet builder."

    def plan_layout(self, sweep_json_path, row_parameter="auto", col_parameter="auto",
                    max_rows_per_sheet=0, max_cols_per_sheet=0, output_json_path=""):
        """
        Analyze sweep plan and create layout JSON.
        """

        # Load sweep plan
        try:
            with open(sweep_json_path, 'r') as f:
                sweep_plan = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Sweep plan not found: {sweep_json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in sweep plan: {e}")

        # Extract parameters and their values
        parameters = sweep_plan.get("parameters", {})
        iterations = sweep_plan.get("iterations", [])

        if not iterations:
            raise ValueError("Sweep plan has no iterations")

        # Build list of params with their value counts
        param_info = []

        # Numeric params (param_1 through param_8)
        for i in range(1, 9):
            slot_key = f"param_{i}"
            if slot_key in parameters and parameters[slot_key].get("name"):
                param = parameters[slot_key]
                param_info.append({
                    "slot": slot_key,
                    "name": param["name"],
                    "type": param.get("type", "float"),
                    "values": param.get("values", []),
                    "count": len(param.get("values", []))
                })

        # String params (string_param_1 and string_param_2)
        for i in range(1, 3):
            slot_key = f"string_param_{i}"
            if slot_key in parameters and parameters[slot_key].get("name"):
                param = parameters[slot_key]
                param_info.append({
                    "slot": slot_key,
                    "name": param["name"],
                    "type": "string",
                    "values": param.get("values", []),
                    "count": len(param.get("values", []))
                })

        if not param_info:
            raise ValueError("No parameters found in sweep plan")

        # Sort by value count (descending) for auto-layout
        param_info_sorted = sorted(param_info, key=lambda x: x["count"], reverse=True)

        # Determine row/col/page params
        if row_parameter == "auto" and col_parameter == "auto":
            # Auto-assign based on counts
            if len(param_info_sorted) == 1:
                # Single param: all columns, one row
                col_param = param_info_sorted[0]
                row_param = None
                page_params = []
            elif len(param_info_sorted) == 2:
                # Two params: larger → cols, smaller → rows
                col_param = param_info_sorted[0]
                row_param = param_info_sorted[1]
                page_params = []
            else:
                # 3+ params: two largest → grid, rest → pages
                col_param = param_info_sorted[0]
                row_param = param_info_sorted[1]
                page_params = param_info_sorted[2:]
        else:
            # User specified row/col
            row_param = None
            col_param = None
            page_params = []

            for p in param_info:
                if p["name"] == row_parameter:
                    row_param = p
                elif p["name"] == col_parameter:
                    col_param = p
                else:
                    page_params.append(p)

            # Validate
            if col_parameter != "auto" and col_param is None:
                raise ValueError(f"Column parameter '{col_parameter}' not found in sweep")
            if row_parameter != "auto" and row_param is None:
                raise ValueError(f"Row parameter '{row_parameter}' not found in sweep")

            # If only one specified, auto-fill the other
            remaining = [p for p in param_info if p not in [row_param, col_param] and p not in page_params]
            if col_param is None and remaining:
                col_param = remaining.pop(0)
            if row_param is None and remaining:
                row_param = remaining.pop(0)
            page_params.extend(remaining)

        # Get full row/col value lists
        all_row_values = row_param["values"] if row_param else [None]
        all_col_values = col_param["values"] if col_param else [None]
        total_rows = len(all_row_values)
        total_cols = len(all_col_values)

        # Apply grid size limits - chunk row/col values if needed
        rows_per_sheet = max_rows_per_sheet if max_rows_per_sheet > 0 else total_rows
        cols_per_sheet = max_cols_per_sheet if max_cols_per_sheet > 0 else total_cols

        # Split values into chunks
        def chunk_list(lst, chunk_size):
            """Split list into chunks of given size."""
            return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

        row_chunks = chunk_list(all_row_values, rows_per_sheet)
        col_chunks = chunk_list(all_col_values, cols_per_sheet)

        # Calculate page combinations from extra params (3+ params)
        if page_params:
            page_value_lists = [p["values"] for p in page_params]
            page_combinations = list(product(*page_value_lists))
        else:
            page_combinations = [()]

        # Build sheets - one for each combination of:
        # (page_param_combo) × (row_chunk) × (col_chunk)
        sheets = []
        sheet_idx = 0

        for page_combo in page_combinations:
            # Build page label for extra params
            if page_params:
                page_values_dict = {p["name"]: page_combo[i] for i, p in enumerate(page_params)}
                page_label_parts = [f"{p['name']}={page_combo[i]}" for i, p in enumerate(page_params)]
            else:
                page_values_dict = {}
                page_label_parts = []

            for row_chunk_idx, row_chunk in enumerate(row_chunks):
                for col_chunk_idx, col_chunk in enumerate(col_chunks):
                    # Build sheet label
                    sheet_label_parts = list(page_label_parts)

                    # Add row/col chunk info if there are multiple chunks
                    if len(row_chunks) > 1:
                        row_start = row_chunk[0]
                        row_end = row_chunk[-1]
                        if row_param:
                            sheet_label_parts.append(f"{row_param['name']}={row_start}-{row_end}")
                    if len(col_chunks) > 1:
                        col_start = col_chunk[0]
                        col_end = col_chunk[-1]
                        if col_param:
                            sheet_label_parts.append(f"{col_param['name']}={col_start}-{col_end}")

                    sheet_label = ", ".join(sheet_label_parts) if sheet_label_parts else f"Sheet {sheet_idx + 1}"

                    # Build grid for this sheet
                    grid = []
                    for row_idx, row_val in enumerate(row_chunk):
                        grid_row = []
                        for col_idx, col_val in enumerate(col_chunk):
                            # Find matching iteration
                            matching_iter = None
                            for it in iterations:
                                params = it.get("params", {})

                                # Check row param
                                if row_param and params.get(row_param["name"]) != row_val:
                                    continue

                                # Check col param
                                if col_param and params.get(col_param["name"]) != col_val:
                                    continue

                                # Check page params
                                page_match = True
                                for p_idx, p in enumerate(page_params):
                                    if params.get(p["name"]) != page_combo[p_idx]:
                                        page_match = False
                                        break

                                if page_match:
                                    matching_iter = it
                                    break

                            cell = {
                                "row": row_idx,
                                "col": col_idx,
                                "row_value": row_val,
                                "col_value": col_val,
                            }

                            if matching_iter:
                                cell["iter_id"] = matching_iter["id"]
                                cell["suffix"] = matching_iter.get("output_suffix", "")
                                cell["label"] = matching_iter.get("label", "")
                                cell["params"] = matching_iter.get("params", {})
                            else:
                                cell["iter_id"] = None
                                cell["suffix"] = None
                                cell["label"] = "missing"
                                cell["params"] = {}

                            grid_row.append(cell)
                        grid.append(grid_row)

                    sheets.append({
                        "page_id": sheet_idx,
                        "page_values": page_values_dict,
                        "page_label": sheet_label,
                        "row_chunk_idx": row_chunk_idx,
                        "col_chunk_idx": col_chunk_idx,
                        "row_values": row_chunk,
                        "col_values": col_chunk,
                        "grid": grid
                    })
                    sheet_idx += 1

        total_sheets = len(sheets)
        cells_per_sheet = rows_per_sheet * cols_per_sheet
        num_rows = min(total_rows, rows_per_sheet)
        num_cols = min(total_cols, cols_per_sheet)

        # Build layout JSON
        layout = {
            "version": "1.0",
            "source_sweep": sweep_json_path,
            "created_at": datetime.now().isoformat(),
            "layout": {
                "row_param": {
                    "name": row_param["name"] if row_param else None,
                    "values": row_param["values"] if row_param else []
                },
                "col_param": {
                    "name": col_param["name"] if col_param else None,
                    "values": col_param["values"] if col_param else []
                },
                "page_params": [
                    {"name": p["name"], "values": p["values"]}
                    for p in page_params
                ]
            },
            "dimensions": {
                "rows": num_rows,
                "cols": num_cols,
                "cells_per_sheet": cells_per_sheet,
                "total_sheets": total_sheets
            },
            "sheets": sheets
        }

        # Determine output path
        if not output_json_path.strip():
            sweep_dir = os.path.dirname(sweep_json_path) or "."
            sweep_name = sweep_plan.get("sweep_name", "sweep")
            output_json_path = os.path.join(sweep_dir, f"{sweep_name}_contact_layout.json")

        output_json_path = get_unique_path(output_json_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)

        # Save layout JSON
        with open(output_json_path, 'w') as f:
            json.dump(layout, f, indent=2)

        # Build summary
        if row_param and col_param:
            grid_desc = f"{num_rows} rows ({row_param['name']}) x {num_cols} cols ({col_param['name']})"
        elif col_param:
            grid_desc = f"1 row x {num_cols} cols ({col_param['name']})"
        else:
            grid_desc = "1x1 grid"

        if total_sheets > 1:
            page_desc = f" x {total_sheets} sheets"
            if page_params:
                page_names = " x ".join(p["name"] for p in page_params)
                page_desc += f" ({page_names})"
        else:
            page_desc = ""

        summary = f"{grid_desc}{page_desc} = {cells_per_sheet * total_sheets} total cells"

        print(f"[NV_SweepContactSheetPlanner] Layout saved to: {output_json_path}")
        print(f"[NV_SweepContactSheetPlanner] {summary}")

        return (output_json_path, summary, cells_per_sheet, total_sheets)


class NV_SweepContactSheetBuilder:
    """
    Builds contact sheet images from layout JSON and sweep output images.

    Features:
    - Row/column headers showing parameter values
    - Labels below each cell
    - Placeholder for missing images
    - Supports multiple sheets (pagination)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to layout JSON from NV_SweepContactSheetPlanner"
                }),
                "image_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing sweep output images"
                }),
            },
            "optional": {
                # Image sizing
                "cell_width": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 2048,
                    "tooltip": "Width of each image cell"
                }),
                "cell_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "tooltip": "Height of each cell (0 = auto from aspect ratio)"
                }),
                "cell_padding": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "tooltip": "Padding between cells"
                }),
                "outer_padding": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 128,
                    "tooltip": "Padding around entire grid"
                }),
                "background_color": ("STRING", {
                    "default": "#1a1a1a",
                    "tooltip": "Background color (hex)"
                }),

                # Cell labels
                "show_cell_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show parameter labels below each cell"
                }),
                "cell_label_height": ("INT", {
                    "default": 40,
                    "min": 20,
                    "max": 100,
                    "tooltip": "Height of label area below each cell"
                }),
                "cell_label_font_size": ("INT", {
                    "default": 12,
                    "min": 8,
                    "max": 32,
                    "tooltip": "Font size for cell labels"
                }),

                # Row/Column headers
                "show_row_headers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show row headers on left side"
                }),
                "show_col_headers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show column headers on top"
                }),
                "header_width": ("INT", {
                    "default": 80,
                    "min": 40,
                    "max": 200,
                    "tooltip": "Width of row header column"
                }),
                "header_height": ("INT", {
                    "default": 40,
                    "min": 20,
                    "max": 100,
                    "tooltip": "Height of column header row"
                }),
                "header_font_size": ("INT", {
                    "default": 14,
                    "min": 10,
                    "max": 36,
                    "tooltip": "Font size for headers"
                }),

                # Colors
                "text_color": ("STRING", {
                    "default": "#ffffff",
                    "tooltip": "Text color (hex)"
                }),
                "header_bg_color": ("STRING", {
                    "default": "#2a2a2a",
                    "tooltip": "Header background color (hex)"
                }),

                # File matching
                "file_pattern": ("STRING", {
                    "default": "*",
                    "tooltip": "Glob pattern to match files (without extension). Use * to match all."
                }),
                "match_by": (["output_suffix", "iteration_id"], {
                    "default": "output_suffix",
                    "tooltip": "How to match files to iterations"
                }),

                # Video options
                "include_videos": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include video files (.mp4, .webm, .mov) and extract frame"
                }),
                "video_frame_position": (["first", "middle", "last"], {
                    "default": "first",
                    "tooltip": "Which frame to extract from video files"
                }),

                # Filtering
                "exclude_patterns": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated substrings to exclude from file matching (e.g., '-comp,-crop' to skip debug files)"
                }),

                # Output options
                "save_to_path": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save contact sheets. Empty = don't save to disk."
                }),
                "filename_prefix": ("STRING", {
                    "default": "contact_sheet",
                    "tooltip": "Prefix for saved filenames (e.g., 'contact_sheet' → 'contact_sheet_001.png')"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("contact_sheets", "sheet_labels", "build_report", "saved_paths")
    FUNCTION = "build_sheets"
    CATEGORY = "NV_Utils/Sweep"
    DESCRIPTION = "Builds contact sheet images from layout JSON and sweep output images/videos."

    def build_sheets(self, layout_json_path, image_directory,
                     cell_width=256, cell_height=0, cell_padding=8, outer_padding=16,
                     background_color="#1a1a1a",
                     show_cell_labels=True, cell_label_height=40, cell_label_font_size=12,
                     show_row_headers=True, show_col_headers=True,
                     header_width=80, header_height=40, header_font_size=14,
                     text_color="#ffffff", header_bg_color="#2a2a2a",
                     file_pattern="*", match_by="output_suffix",
                     include_videos=True, video_frame_position="first",
                     exclude_patterns="",
                     save_to_path="", filename_prefix="contact_sheet"):
        """
        Build contact sheet images.
        """

        # Load layout JSON
        try:
            with open(layout_json_path, 'r') as f:
                layout = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Layout JSON not found: {layout_json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid layout JSON: {e}")

        # Get dimensions
        dims = layout.get("dimensions", {})
        num_rows = dims.get("rows", 1)
        num_cols = dims.get("cols", 1)
        sheets_data = layout.get("sheets", [])

        if not sheets_data:
            raise ValueError("Layout has no sheets")

        # Build list of extensions to search for
        extensions_to_search = list(IMAGE_EXTENSIONS)
        if include_videos:
            extensions_to_search.extend(VIDEO_EXTENSIONS)

        # Find all matching files in directory
        media_files = []
        for ext in extensions_to_search:
            pattern = f"{file_pattern}{ext}"
            media_files.extend(glob.glob(os.path.join(image_directory, pattern)))
            media_files.extend(glob.glob(os.path.join(image_directory, "**", pattern), recursive=True))
        media_files = list(set(media_files))  # Remove duplicates

        # Parse and apply exclude patterns
        exclude_list = []
        if exclude_patterns and exclude_patterns.strip():
            exclude_list = [p.strip() for p in exclude_patterns.split(",") if p.strip()]

        if exclude_list:
            filtered_files = []
            excluded_count = 0
            for file_path in media_files:
                filename = os.path.basename(file_path)
                should_exclude = False
                for pattern in exclude_list:
                    if pattern in filename:
                        should_exclude = True
                        excluded_count += 1
                        break
                if not should_exclude:
                    filtered_files.append(file_path)
            media_files = filtered_files
            if excluded_count > 0:
                print(f"[NV_SweepContactSheetBuilder] Excluded {excluded_count} files matching patterns: {exclude_list}")

        # Debug: show which files passed filtering
        print(f"[NV_SweepContactSheetBuilder] Files after filtering ({len(media_files)}):")
        for f in media_files[:10]:  # Show first 10
            print(f"  - {os.path.basename(f)}")
        if len(media_files) > 10:
            print(f"  ... and {len(media_files) - 10} more")

        # Build lookup dict
        file_lookup = {}
        for file_path in media_files:
            filename = os.path.basename(file_path)
            name_no_ext = os.path.splitext(filename)[0]
            file_lookup[name_no_ext] = file_path
            file_lookup[filename] = file_path

        # Load fonts
        header_font = get_font(header_font_size)
        label_font = get_font(cell_label_font_size)

        # Convert colors
        bg_rgb = hex_to_rgb(background_color)
        text_rgb = hex_to_rgb(text_color)
        header_bg_rgb = hex_to_rgb(header_bg_color)
        placeholder_rgb = (51, 51, 51)  # #333333

        # Determine cell height if auto
        if cell_height <= 0:
            # Try to get from first file (image or video)
            first_found = False
            for sheet in sheets_data:
                for row in sheet.get("grid", []):
                    for cell in row:
                        suffix = cell.get("suffix")
                        if suffix:
                            file_path = self._find_media(suffix, cell.get("iter_id"), file_lookup, match_by)
                            if file_path:
                                try:
                                    # Load image or extract video frame
                                    if is_video_file(file_path):
                                        img = extract_video_frame(file_path, video_frame_position)
                                    else:
                                        img = Image.open(file_path)

                                    if img:
                                        aspect = img.height / img.width
                                        cell_height = int(cell_width * aspect)
                                        first_found = True
                                        if not is_video_file(file_path):
                                            img.close()
                                        break
                                except:
                                    pass
                    if first_found:
                        break
                if first_found:
                    break

            if cell_height <= 0:
                cell_height = cell_width  # Default to square

        # Pre-calculate shared dimensions
        label_h = cell_label_height if show_cell_labels else 0
        row_header_w = header_width if show_row_headers else 0
        col_header_h = header_height if show_col_headers else 0

        # Count file types
        image_count = sum(1 for f in media_files if is_image_file(f))
        video_count = sum(1 for f in media_files if is_video_file(f))

        # Build each sheet
        result_images = []
        sheet_labels_list = []
        report_lines = [
            f"Building contact sheets from: {layout_json_path}",
            f"Media directory: {image_directory}",
            f"Max grid: {num_rows} rows x {num_cols} cols per sheet",
            f"Cell size: {cell_width}x{cell_height}",
            f"Found {len(media_files)} files ({image_count} images, {video_count} videos)",
            f"Video frame position: {video_frame_position}" if include_videos else "",
            ""
        ]

        row_param_name = layout.get("layout", {}).get("row_param", {}).get("name", "")
        col_param_name = layout.get("layout", {}).get("col_param", {}).get("name", "")

        for sheet_data in sheets_data:
            page_label = sheet_data.get("page_label", "")
            sheet_labels_list.append(page_label if page_label else f"Sheet {sheet_data['page_id']}")

            # Get sheet-specific row/col values (may be chunked subsets)
            sheet_row_values = sheet_data.get("row_values", [])
            sheet_col_values = sheet_data.get("col_values", [])
            sheet_num_rows = len(sheet_row_values)
            sheet_num_cols = len(sheet_col_values)

            # Calculate canvas size for this sheet
            sheet_width = (
                outer_padding +
                row_header_w +
                (cell_width + cell_padding) * sheet_num_cols - cell_padding +
                outer_padding
            )
            sheet_height = (
                outer_padding +
                col_header_h +
                (cell_height + label_h + cell_padding) * sheet_num_rows - cell_padding +
                outer_padding
            )

            report_lines.append(f"Sheet {sheet_data['page_id']}: {page_label} ({sheet_num_rows}x{sheet_num_cols})")

            # Create canvas
            canvas = Image.new("RGB", (sheet_width, sheet_height), bg_rgb)
            draw = ImageDraw.Draw(canvas)

            # Draw column headers
            if show_col_headers and sheet_col_values:
                for col_idx, col_val in enumerate(sheet_col_values):
                    x = outer_padding + row_header_w + col_idx * (cell_width + cell_padding)
                    y = outer_padding

                    # Header background
                    draw.rectangle(
                        [x, y, x + cell_width, y + col_header_h],
                        fill=header_bg_rgb
                    )

                    # Header text
                    header_text = f"{col_param_name}={col_val}" if col_param_name else str(col_val)
                    if header_font:
                        bbox = draw.textbbox((0, 0), header_text, font=header_font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        text_x = x + (cell_width - text_w) // 2
                        text_y = y + (col_header_h - text_h) // 2
                        draw.text((text_x, text_y), header_text, fill=text_rgb, font=header_font)

            # Draw row headers
            if show_row_headers and sheet_row_values:
                for row_idx, row_val in enumerate(sheet_row_values):
                    x = outer_padding
                    y = outer_padding + col_header_h + row_idx * (cell_height + label_h + cell_padding)

                    # Header background
                    draw.rectangle(
                        [x, y, x + row_header_w, y + cell_height + label_h],
                        fill=header_bg_rgb
                    )

                    # Header text (centered vertically)
                    header_text = f"{row_param_name}={row_val}" if row_param_name else str(row_val)
                    if header_font:
                        bbox = draw.textbbox((0, 0), header_text, font=header_font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        text_x = x + (row_header_w - text_w) // 2
                        text_y = y + (cell_height + label_h - text_h) // 2
                        draw.text((text_x, text_y), header_text, fill=text_rgb, font=header_font)

            # Draw cells
            grid = sheet_data.get("grid", [])
            files_found = 0
            files_missing = 0
            videos_loaded = 0
            matched_files = set()  # Track unique files matched
            match_log = []  # Log each match for debugging

            for row_idx, row in enumerate(grid):
                for col_idx, cell in enumerate(row):
                    x = outer_padding + row_header_w + col_idx * (cell_width + cell_padding)
                    y = outer_padding + col_header_h + row_idx * (cell_height + label_h + cell_padding)

                    suffix = cell.get("suffix")
                    iter_id = cell.get("iter_id")

                    # Find and load image or video
                    file_path = self._find_media(suffix, iter_id, file_lookup, match_by)
                    if file_path:
                        matched_files.add(file_path)
                        match_log.append(f"    [{row_idx},{col_idx}] suffix='{suffix}' → {os.path.basename(file_path)}")
                    else:
                        match_log.append(f"    [{row_idx},{col_idx}] suffix='{suffix}' → NO MATCH")

                    if file_path:
                        try:
                            # Load image or extract video frame
                            if is_video_file(file_path):
                                img = extract_video_frame(file_path, video_frame_position)
                                if img is None:
                                    raise ValueError("Failed to extract frame")
                                videos_loaded += 1
                            else:
                                img = Image.open(file_path)

                            # Resize to cell size
                            img_resized = img.convert("RGB").resize(
                                (cell_width, cell_height),
                                Image.Resampling.LANCZOS
                            )
                            canvas.paste(img_resized, (x, y))
                            files_found += 1

                            # Close if it was an opened image file
                            if not is_video_file(file_path):
                                img.close()
                        except Exception as e:
                            self._draw_placeholder(draw, x, y, cell_width, cell_height,
                                                   f"Error: #{iter_id}", placeholder_rgb, text_rgb, label_font)
                            files_missing += 1
                    else:
                        # Draw placeholder
                        self._draw_placeholder(draw, x, y, cell_width, cell_height,
                                               f"Missing: #{iter_id}", placeholder_rgb, text_rgb, label_font)
                        files_missing += 1

                    # Draw cell label
                    if show_cell_labels:
                        label_y = y + cell_height

                        # Build compact label from non-header params
                        params = cell.get("params", {})
                        label_parts = []
                        for k, v in params.items():
                            # Skip row/col params (already in headers)
                            if k == row_param_name or k == col_param_name:
                                continue
                            # Skip page params (already in sheet label)
                            skip = False
                            for pp in layout.get("layout", {}).get("page_params", []):
                                if k == pp.get("name"):
                                    skip = True
                                    break
                            if skip:
                                continue
                            # Format value compactly
                            if isinstance(v, float):
                                if v == int(v):
                                    label_parts.append(f"{k}={int(v)}")
                                else:
                                    label_parts.append(f"{k}={v:.2f}")
                            else:
                                label_parts.append(f"{k}={v}")

                        cell_label = ", ".join(label_parts) if label_parts else f"iter {iter_id}"

                        if label_font:
                            bbox = draw.textbbox((0, 0), cell_label, font=label_font)
                            text_w = bbox[2] - bbox[0]
                            text_h = bbox[3] - bbox[1]
                            text_x = x + (cell_width - text_w) // 2
                            text_y = label_y + (label_h - text_h) // 2
                            draw.text((text_x, text_y), cell_label, fill=text_rgb, font=label_font)

            video_note = f" ({videos_loaded} videos)" if videos_loaded > 0 else ""
            unique_note = f" ({len(matched_files)} unique)" if len(matched_files) != files_found else ""
            report_lines.append(f"  Found: {files_found}{unique_note}{video_note}, Missing: {files_missing}")

            # Debug: show first 5 matches per sheet
            if match_log:
                print(f"[NV_SweepContactSheetBuilder] Sheet {sheet_data['page_id']} matches (first 5):")
                for log_line in match_log[:5]:
                    print(log_line)
                if len(match_log) > 5:
                    print(f"    ... and {len(match_log) - 5} more")

            if len(matched_files) < files_found:
                print(f"[NV_SweepContactSheetBuilder] WARNING: Only {len(matched_files)} unique files matched {files_found} cells!")
                for f in matched_files:
                    print(f"  - {os.path.basename(f)}")

            # Store canvas and its dimensions
            result_images.append((canvas, sheet_width, sheet_height))

        # Save sheets to disk if path provided
        saved_paths_list = []
        if save_to_path and save_to_path.strip():
            os.makedirs(save_to_path, exist_ok=True)
            for idx, (canvas, w, h) in enumerate(result_images):
                # Build filename with sheet index and label
                sheet_label_safe = sheet_labels_list[idx] if idx < len(sheet_labels_list) else ""
                # Make label filename-safe
                for char in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|", ",", "="]:
                    sheet_label_safe = sheet_label_safe.replace(char, "_")
                while "__" in sheet_label_safe:
                    sheet_label_safe = sheet_label_safe.replace("__", "_")
                sheet_label_safe = sheet_label_safe.strip("_")

                if sheet_label_safe:
                    filename = f"{filename_prefix}_{idx:03d}_{sheet_label_safe}.png"
                else:
                    filename = f"{filename_prefix}_{idx:03d}.png"

                save_path = os.path.join(save_to_path, filename)
                canvas.save(save_path, "PNG")
                saved_paths_list.append(save_path)
                print(f"[NV_SweepContactSheetBuilder] Saved: {save_path}")

        # Find max dimensions across all sheets for uniform batching
        if result_images:
            max_width = max(img[1] for img in result_images)
            max_height = max(img[2] for img in result_images)

            # Convert all sheets to tensors with uniform size (pad if needed)
            tensor_list = []
            for canvas, w, h in result_images:
                if w < max_width or h < max_height:
                    # Pad to max size
                    padded = Image.new("RGB", (max_width, max_height), bg_rgb)
                    padded.paste(canvas, (0, 0))
                    img_array = np.array(padded).astype(np.float32) / 255.0
                else:
                    img_array = np.array(canvas).astype(np.float32) / 255.0
                tensor_list.append(torch.from_numpy(img_array))

            result_batch = torch.stack(tensor_list, dim=0)
        else:
            # Return empty tensor if no sheets
            max_width = 256
            max_height = 256
            result_batch = torch.zeros((1, max_height, max_width, 3))

        sheet_labels = ", ".join(sheet_labels_list)
        build_report = "\n".join(report_lines)
        saved_paths = ", ".join(saved_paths_list) if saved_paths_list else ""

        print(f"[NV_SweepContactSheetBuilder] Built {len(result_images)} sheets")
        print(f"[NV_SweepContactSheetBuilder] Output size: {max_width}x{max_height}")
        if saved_paths_list:
            print(f"[NV_SweepContactSheetBuilder] Saved {len(saved_paths_list)} files to: {save_to_path}")

        return (result_batch, sheet_labels, build_report, saved_paths)

    def _find_media(self, suffix, iter_id, file_lookup, match_by, debug=False):
        """Find image or video file matching the cell."""
        if match_by == "output_suffix" and suffix:
            # Look for suffix followed by a delimiter (_, -, ., or end of string)
            # This prevents "0-4" from matching "0-45"
            # Escape special regex chars in suffix, then require delimiter after
            pattern = re.escape(suffix) + r'(?:[_\-\.]|$)'

            for key in file_lookup:
                if re.search(pattern, key):
                    if debug:
                        print(f"  [match] suffix '{suffix}' found in '{key}'")
                    return file_lookup[key]
            if debug:
                print(f"  [no match] suffix '{suffix}' not found in any file")

        if match_by == "iteration_id" and iter_id is not None:
            # Look for iter_id in filename
            patterns = [
                f"iter{iter_id}_",
                f"iter_{iter_id}_",
                f"iteration{iter_id}_",
                f"_{iter_id:04d}_",
                f"_{iter_id:04d}.",
            ]
            for key in file_lookup:
                for pattern in patterns:
                    if pattern in key:
                        return file_lookup[key]

        return None

    def _draw_placeholder(self, draw, x, y, width, height, text, bg_color, text_color, font):
        """Draw a placeholder cell for missing images."""
        # Background
        draw.rectangle([x, y, x + width, y + height], fill=bg_color)

        # X pattern
        line_color = (80, 80, 80)
        draw.line([(x, y), (x + width, y + height)], fill=line_color, width=2)
        draw.line([(x + width, y), (x, y + height)], fill=line_color, width=2)

        # Text
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = x + (width - text_w) // 2
            text_y = y + (height - text_h) // 2

            # Background for text
            padding = 4
            draw.rectangle(
                [text_x - padding, text_y - padding, text_x + text_w + padding, text_y + text_h + padding],
                fill=bg_color
            )
            draw.text((text_x, text_y), text, fill=text_color, font=font)


class VideoStreamManager:
    """
    Context manager for efficient frame-by-frame reading from multiple video files.

    Keeps video handles open for sequential reading, avoiding repeated file opens
    and seeking for each frame.
    """

    def __init__(self, video_info: dict, cell_width: int, cell_height: int):
        """
        Args:
            video_info: Dict mapping cell_key to video info dict with 'path' key
            cell_width: Target width for resized frames
            cell_height: Target height for resized frames
        """
        self.video_info = video_info
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.handles = {}  # {cell_key: cv2.VideoCapture}

    def __enter__(self):
        try:
            import cv2
        except ImportError:
            print("[VideoStreamManager] Warning: OpenCV not available")
            return self

        for cell_key, info in self.video_info.items():
            if info and info.get("path"):
                cap = cv2.VideoCapture(info["path"])
                if cap.isOpened():
                    self.handles[cell_key] = cap
        return self

    def __exit__(self, *args):
        for cap in self.handles.values():
            cap.release()
        self.handles.clear()

    def read_frame(self, cell_key) -> np.ndarray:
        """
        Read next frame from video, resize, and return as RGB numpy array.

        Returns:
            RGB numpy array of shape (cell_height, cell_width, 3), or None if failed
        """
        try:
            import cv2
        except ImportError:
            return None

        cap = self.handles.get(cell_key)
        if cap is None:
            return None

        ret, frame = cap.read()
        if not ret or frame is None:
            return None

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to cell dimensions
        frame_resized = cv2.resize(
            frame_rgb,
            (self.cell_width, self.cell_height),
            interpolation=cv2.INTER_LANCZOS4
        )

        return frame_resized


class NV_SweepVideoContactSheetBuilder:
    """
    Builds animated MP4 contact sheets from layout JSON and sweep output videos.

    Features:
    - Synchronizes all videos by trimming to shortest length
    - Row/column headers showing parameter values
    - Labels below each cell
    - Placeholder for missing videos
    - H.264 MP4 output with configurable quality
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to layout JSON from NV_SweepContactSheetPlanner"
                }),
                "video_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing sweep output videos (.mp4, .webm, etc.)"
                }),
            },
            "optional": {
                # Cell sizing
                "cell_width": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "tooltip": "Width of each video cell"
                }),
                "cell_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,
                    "tooltip": "Height of each cell (0 = auto from aspect ratio)"
                }),
                "cell_padding": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 32,
                    "tooltip": "Padding between cells"
                }),
                "outer_padding": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 64,
                    "tooltip": "Padding around entire grid"
                }),

                # Colors
                "background_color": ("STRING", {
                    "default": "#1a1a1a",
                    "tooltip": "Background color (hex)"
                }),
                "text_color": ("STRING", {
                    "default": "#ffffff",
                    "tooltip": "Text color (hex)"
                }),
                "header_bg_color": ("STRING", {
                    "default": "#2a2a2a",
                    "tooltip": "Header background color (hex)"
                }),

                # Cell labels
                "show_cell_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show parameter labels below each cell"
                }),
                "cell_label_height": ("INT", {
                    "default": 40,
                    "min": 20,
                    "max": 100,
                    "tooltip": "Height of label area below each cell"
                }),
                "cell_label_font_size": ("INT", {
                    "default": 12,
                    "min": 8,
                    "max": 24,
                    "tooltip": "Font size for cell labels"
                }),

                # Row/Column headers
                "show_row_headers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show row headers on left side"
                }),
                "show_col_headers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show column headers on top"
                }),
                "header_width": ("INT", {
                    "default": 80,
                    "min": 40,
                    "max": 200,
                    "tooltip": "Width of row header column"
                }),
                "header_height": ("INT", {
                    "default": 40,
                    "min": 20,
                    "max": 100,
                    "tooltip": "Height of column header row"
                }),
                "header_font_size": ("INT", {
                    "default": 14,
                    "min": 10,
                    "max": 24,
                    "tooltip": "Font size for headers"
                }),

                # Video output
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 0.01,
                    "tooltip": "Output video frame rate"
                }),
                "crf": ("INT", {
                    "default": 23,
                    "min": 0,
                    "max": 51,
                    "tooltip": "H.264 quality (0=lossless, 23=default, 51=worst)"
                }),

                # File matching
                "file_pattern": ("STRING", {
                    "default": "*",
                    "tooltip": "Glob pattern to match files (without extension)"
                }),
                "match_by": (["output_suffix", "iteration_id"], {
                    "default": "output_suffix",
                    "tooltip": "How to match files to iterations"
                }),
                "exclude_patterns": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated substrings to exclude from file matching"
                }),

                # Output
                "save_to_path": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save output MP4 files (required)"
                }),
                "filename_prefix": ("STRING", {
                    "default": "video_contact_sheet",
                    "tooltip": "Prefix for saved filenames"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("saved_paths", "build_report", "total_frames")
    OUTPUT_NODE = True
    FUNCTION = "build_video_sheets"
    CATEGORY = "NV_Utils/Sweep"
    DESCRIPTION = "Builds animated MP4 contact sheets from layout JSON and sweep output videos."

    def build_video_sheets(self, layout_json_path, video_directory,
                           cell_width=256, cell_height=0, cell_padding=8, outer_padding=16,
                           background_color="#1a1a1a", text_color="#ffffff", header_bg_color="#2a2a2a",
                           show_cell_labels=True, cell_label_height=40, cell_label_font_size=12,
                           show_row_headers=True, show_col_headers=True,
                           header_width=80, header_height=40, header_font_size=14,
                           fps=24.0, crf=23,
                           file_pattern="*", match_by="output_suffix", exclude_patterns="",
                           save_to_path="", filename_prefix="video_contact_sheet"):
        """Build animated MP4 contact sheets from sweep output videos."""

        # Check dependencies
        if not HAS_PYAV:
            raise ImportError("PyAV is required for video encoding. Install with: pip install av")

        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for video reading. Install with: pip install opencv-python")

        if not save_to_path or not save_to_path.strip():
            raise ValueError("save_to_path is required for video output")

        # Load layout JSON
        try:
            with open(layout_json_path, 'r') as f:
                layout = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Layout JSON not found: {layout_json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid layout JSON: {e}")

        sheets_data = layout.get("sheets", [])
        if not sheets_data:
            raise ValueError("Layout has no sheets")

        # Find video files
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            pattern = f"{file_pattern}{ext}"
            video_files.extend(glob.glob(os.path.join(video_directory, pattern)))
            video_files.extend(glob.glob(os.path.join(video_directory, "**", pattern), recursive=True))
        video_files = list(set(video_files))

        # Apply exclude patterns
        if exclude_patterns and exclude_patterns.strip():
            exclude_list = [p.strip() for p in exclude_patterns.split(",") if p.strip()]
            video_files = [f for f in video_files
                          if not any(p in os.path.basename(f) for p in exclude_list)]

        print(f"[NV_SweepVideoContactSheetBuilder] Found {len(video_files)} video files")

        # Build lookup dict
        file_lookup = {}
        for file_path in video_files:
            filename = os.path.basename(file_path)
            name_no_ext = os.path.splitext(filename)[0]
            file_lookup[name_no_ext] = file_path
            file_lookup[filename] = file_path

        # Match videos to cells and analyze
        all_video_matches = {}  # {(sheet_idx, row, col): file_path}

        for sheet_data in sheets_data:
            sheet_idx = sheet_data["page_id"]
            for row in sheet_data.get("grid", []):
                for cell in row:
                    cell_key = (sheet_idx, cell["row"], cell["col"])
                    suffix = cell.get("suffix")
                    iter_id = cell.get("iter_id")
                    file_path = self._find_video(suffix, iter_id, file_lookup, match_by)
                    all_video_matches[cell_key] = file_path

        # Analyze all videos
        video_analysis = self._analyze_videos(all_video_matches, cv2)
        target_frames = video_analysis["target_frames"]

        if target_frames == 0:
            raise ValueError("No valid videos found or all videos have 0 frames")

        print(f"[NV_SweepVideoContactSheetBuilder] Target frames (shortest video): {target_frames}")
        print(f"[NV_SweepVideoContactSheetBuilder] Videos found: {len(video_analysis['videos'])}, Missing: {len(video_analysis['missing_cells'])}")

        # Determine cell_height from first video if auto
        if cell_height <= 0:
            for info in video_analysis["videos"].values():
                if info["width"] > 0 and info["height"] > 0:
                    aspect = info["height"] / info["width"]
                    cell_height = int(cell_width * aspect)
                    break
            if cell_height <= 0:
                cell_height = cell_width  # Default to square

        # Load fonts
        header_font = get_font(header_font_size)
        label_font = get_font(cell_label_font_size)

        # Convert colors
        bg_rgb = hex_to_rgb(background_color)
        text_rgb = hex_to_rgb(text_color)
        header_bg_rgb = hex_to_rgb(header_bg_color)
        placeholder_rgb = (51, 51, 51)

        # Get param names for headers
        row_param_name = layout.get("layout", {}).get("row_param", {}).get("name", "")
        col_param_name = layout.get("layout", {}).get("col_param", {}).get("name", "")

        # Pre-calculate dimensions
        label_h = cell_label_height if show_cell_labels else 0
        row_header_w = header_width if show_row_headers else 0
        col_header_h = header_height if show_col_headers else 0

        # Ensure output directory exists
        os.makedirs(save_to_path, exist_ok=True)

        saved_paths_list = []
        report_lines = [
            f"Building video contact sheets from: {layout_json_path}",
            f"Video directory: {video_directory}",
            f"Target frames: {target_frames}",
            f"Output FPS: {fps}",
            f"Cell size: {cell_width}x{cell_height}",
            f"CRF quality: {crf}",
            ""
        ]

        # Process each sheet
        for sheet_data in sheets_data:
            sheet_idx = sheet_data["page_id"]
            page_label = sheet_data.get("page_label", f"Sheet {sheet_idx}")

            sheet_row_values = sheet_data.get("row_values", [])
            sheet_col_values = sheet_data.get("col_values", [])
            sheet_num_rows = len(sheet_row_values) if sheet_row_values else 1
            sheet_num_cols = len(sheet_col_values) if sheet_col_values else 1

            # Calculate canvas size
            canvas_width = (
                outer_padding +
                row_header_w +
                (cell_width + cell_padding) * sheet_num_cols - cell_padding +
                outer_padding
            )
            canvas_height = (
                outer_padding +
                col_header_h +
                (cell_height + label_h + cell_padding) * sheet_num_rows - cell_padding +
                outer_padding
            )

            report_lines.append(f"Sheet {sheet_idx}: {page_label} ({sheet_num_rows}x{sheet_num_cols})")

            # Pre-render static overlay (headers, labels, placeholders)
            static_overlay = self._render_static_overlay(
                sheet_data, layout, canvas_width, canvas_height,
                cell_width, cell_height, cell_padding, outer_padding,
                label_h, row_header_w, col_header_h,
                row_param_name, col_param_name,
                show_row_headers, show_col_headers, show_cell_labels,
                header_font, label_font,
                bg_rgb, text_rgb, header_bg_rgb, placeholder_rgb,
                video_analysis, sheet_idx
            )

            # Build cell position map
            cell_positions = {}
            grid = sheet_data.get("grid", [])
            for row in grid:
                for cell in row:
                    row_idx = cell["row"]
                    col_idx = cell["col"]
                    x = outer_padding + row_header_w + col_idx * (cell_width + cell_padding)
                    y = outer_padding + col_header_h + row_idx * (cell_height + label_h + cell_padding)
                    cell_key = (sheet_idx, row_idx, col_idx)
                    cell_positions[cell_key] = (x, y)

            # Get video info for this sheet's cells
            sheet_video_info = {}
            for row in grid:
                for cell in row:
                    cell_key = (sheet_idx, cell["row"], cell["col"])
                    if cell_key in video_analysis["videos"]:
                        sheet_video_info[cell_key] = video_analysis["videos"][cell_key]

            # Generate output filename
            sheet_label_safe = page_label
            for char in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|", ",", "="]:
                sheet_label_safe = sheet_label_safe.replace(char, "_")
            while "__" in sheet_label_safe:
                sheet_label_safe = sheet_label_safe.replace("__", "_")
            sheet_label_safe = sheet_label_safe.strip("_")

            if sheet_label_safe:
                output_filename = f"{filename_prefix}_{sheet_idx:03d}_{sheet_label_safe}.mp4"
            else:
                output_filename = f"{filename_prefix}_{sheet_idx:03d}.mp4"

            output_path = os.path.join(save_to_path, output_filename)

            # Encode video
            self._encode_video(
                output_path, target_frames,
                canvas_width, canvas_height,
                fps, crf, bg_rgb,
                sheet_video_info, cell_positions, cell_width, cell_height,
                static_overlay
            )

            saved_paths_list.append(output_path)
            report_lines.append(f"  Saved: {output_path}")
            print(f"[NV_SweepVideoContactSheetBuilder] Saved: {output_path}")

        saved_paths = ", ".join(saved_paths_list)
        build_report = "\n".join(report_lines)

        print(f"[NV_SweepVideoContactSheetBuilder] Built {len(saved_paths_list)} video contact sheets")

        return (saved_paths, build_report, target_frames)

    def _find_video(self, suffix, iter_id, file_lookup, match_by):
        """Find video file matching the cell (reuses logic from image builder)."""
        if match_by == "output_suffix" and suffix:
            pattern = re.escape(suffix) + r'(?:[_\-\.]|$)'
            for key in file_lookup:
                if re.search(pattern, key):
                    return file_lookup[key]

        if match_by == "iteration_id" and iter_id is not None:
            patterns = [
                f"iter{iter_id}_",
                f"iter_{iter_id}_",
                f"iteration{iter_id}_",
                f"_{iter_id:04d}_",
                f"_{iter_id:04d}.",
            ]
            for key in file_lookup:
                for pat in patterns:
                    if pat in key:
                        return file_lookup[key]

        return None

    def _analyze_videos(self, video_paths: dict, cv2) -> dict:
        """
        Analyze all matched videos and determine target frame count.

        Returns dict with:
            target_frames: int (shortest video length)
            videos: {cell_key: {path, frame_count, fps, width, height}}
            missing_cells: [cell_key, ...]
        """
        analysis = {"videos": {}, "missing_cells": []}
        min_frames = float('inf')

        for cell_key, path in video_paths.items():
            if path is None:
                analysis["missing_cells"].append(cell_key)
                continue

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                analysis["missing_cells"].append(cell_key)
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if frame_count <= 0:
                analysis["missing_cells"].append(cell_key)
                continue

            analysis["videos"][cell_key] = {
                "path": path,
                "frame_count": frame_count,
                "fps": video_fps,
                "width": width,
                "height": height,
            }

            min_frames = min(min_frames, frame_count)

        analysis["target_frames"] = min_frames if min_frames != float('inf') else 0
        return analysis

    def _render_static_overlay(self, sheet_data, layout, canvas_width, canvas_height,
                                cell_width, cell_height, cell_padding, outer_padding,
                                label_h, row_header_w, col_header_h,
                                row_param_name, col_param_name,
                                show_row_headers, show_col_headers, show_cell_labels,
                                header_font, label_font,
                                bg_rgb, text_rgb, header_bg_rgb, placeholder_rgb,
                                video_analysis, sheet_idx):
        """
        Pre-render static overlay with headers, labels, and placeholders.

        Returns RGBA image where cell areas are transparent.
        """
        # Create RGBA canvas (transparent background for cells)
        overlay = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        sheet_row_values = sheet_data.get("row_values", [])
        sheet_col_values = sheet_data.get("col_values", [])

        # Draw column headers
        if show_col_headers and sheet_col_values:
            for col_idx, col_val in enumerate(sheet_col_values):
                x = outer_padding + row_header_w + col_idx * (cell_width + cell_padding)
                y = outer_padding

                draw.rectangle(
                    [x, y, x + cell_width, y + col_header_h],
                    fill=header_bg_rgb + (255,)
                )

                header_text = f"{col_param_name}={col_val}" if col_param_name else str(col_val)
                if header_font:
                    bbox = draw.textbbox((0, 0), header_text, font=header_font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    text_x = x + (cell_width - text_w) // 2
                    text_y = y + (col_header_h - text_h) // 2
                    draw.text((text_x, text_y), header_text, fill=text_rgb + (255,), font=header_font)

        # Draw row headers
        if show_row_headers and sheet_row_values:
            for row_idx, row_val in enumerate(sheet_row_values):
                x = outer_padding
                y = outer_padding + col_header_h + row_idx * (cell_height + label_h + cell_padding)

                draw.rectangle(
                    [x, y, x + row_header_w, y + cell_height + label_h],
                    fill=header_bg_rgb + (255,)
                )

                header_text = f"{row_param_name}={row_val}" if row_param_name else str(row_val)
                if header_font:
                    bbox = draw.textbbox((0, 0), header_text, font=header_font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    text_x = x + (row_header_w - text_w) // 2
                    text_y = y + (cell_height + label_h - text_h) // 2
                    draw.text((text_x, text_y), header_text, fill=text_rgb + (255,), font=header_font)

        # Draw cell labels and placeholders for missing videos
        grid = sheet_data.get("grid", [])
        for row in grid:
            for cell in row:
                row_idx = cell["row"]
                col_idx = cell["col"]
                cell_key = (sheet_idx, row_idx, col_idx)

                x = outer_padding + row_header_w + col_idx * (cell_width + cell_padding)
                y = outer_padding + col_header_h + row_idx * (cell_height + label_h + cell_padding)

                # Draw placeholder if missing
                if cell_key in video_analysis["missing_cells"]:
                    iter_id = cell.get("iter_id")
                    self._draw_placeholder_rgba(draw, x, y, cell_width, cell_height,
                                                f"Missing: #{iter_id}", placeholder_rgb, text_rgb, label_font)

                # Draw cell label
                if show_cell_labels:
                    label_y = y + cell_height

                    params = cell.get("params", {})
                    label_parts = []
                    for k, v in params.items():
                        if k == row_param_name or k == col_param_name:
                            continue
                        skip = False
                        for pp in layout.get("layout", {}).get("page_params", []):
                            if k == pp.get("name"):
                                skip = True
                                break
                        if skip:
                            continue
                        if isinstance(v, float):
                            if v == int(v):
                                label_parts.append(f"{k}={int(v)}")
                            else:
                                label_parts.append(f"{k}={v:.2f}")
                        else:
                            label_parts.append(f"{k}={v}")

                    cell_label = ", ".join(label_parts) if label_parts else f"iter {cell.get('iter_id')}"

                    # Draw label background
                    draw.rectangle(
                        [x, label_y, x + cell_width, label_y + label_h],
                        fill=bg_rgb + (255,)
                    )

                    if label_font:
                        bbox = draw.textbbox((0, 0), cell_label, font=label_font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        text_x = x + (cell_width - text_w) // 2
                        text_y = label_y + (label_h - text_h) // 2
                        draw.text((text_x, text_y), cell_label, fill=text_rgb + (255,), font=label_font)

        return overlay

    def _draw_placeholder_rgba(self, draw, x, y, width, height, text, bg_color, text_color, font):
        """Draw a placeholder cell for missing videos (RGBA version)."""
        draw.rectangle([x, y, x + width, y + height], fill=bg_color + (255,))

        line_color = (80, 80, 80, 255)
        draw.line([(x, y), (x + width, y + height)], fill=line_color, width=2)
        draw.line([(x + width, y), (x, y + height)], fill=line_color, width=2)

        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = x + (width - text_w) // 2
            text_y = y + (height - text_h) // 2

            padding = 4
            draw.rectangle(
                [text_x - padding, text_y - padding, text_x + text_w + padding, text_y + text_h + padding],
                fill=bg_color + (255,)
            )
            draw.text((text_x, text_y), text, fill=text_color + (255,), font=font)

    def _encode_video(self, output_path, target_frames, canvas_width, canvas_height,
                      fps, crf, bg_rgb, sheet_video_info, cell_positions,
                      cell_width, cell_height, static_overlay):
        """Encode frames to H.264 MP4 using PyAV."""

        # Create output container
        container = av.open(output_path, mode='w')

        # Add video stream
        stream = container.add_stream('h264', rate=Fraction(int(fps * 1000), 1000))
        stream.width = canvas_width
        stream.height = canvas_height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': str(crf)}

        # Progress bar
        pbar = None
        if HAS_COMFY_UTILS:
            pbar = comfy.utils.ProgressBar(target_frames)

        # Open all video streams
        with VideoStreamManager(sheet_video_info, cell_width, cell_height) as streams:
            for frame_idx in range(target_frames):
                # Create canvas with background
                canvas = Image.new("RGB", (canvas_width, canvas_height), bg_rgb)

                # Read and paste each cell's video frame
                for cell_key, (x, y) in cell_positions.items():
                    if cell_key in sheet_video_info:
                        frame_data = streams.read_frame(cell_key)
                        if frame_data is not None:
                            cell_img = Image.fromarray(frame_data)
                            canvas.paste(cell_img, (x, y))

                # Composite static overlay
                canvas.paste(static_overlay, (0, 0), static_overlay)

                # Convert to numpy array
                img_array = np.array(canvas)

                # Create video frame
                frame = av.VideoFrame.from_ndarray(img_array, format='rgb24')
                frame = frame.reformat(format='yuv420p')

                # Encode
                for packet in stream.encode(frame):
                    container.mux(packet)

                if pbar:
                    pbar.update(1)

                if (frame_idx + 1) % 100 == 0:
                    print(f"[NV_SweepVideoContactSheetBuilder] Encoded frame {frame_idx + 1}/{target_frames}")

        # Flush encoder
        for packet in stream.encode(None):
            container.mux(packet)

        container.close()


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_SweepContactSheetPlanner": NV_SweepContactSheetPlanner,
    "NV_SweepContactSheetBuilder": NV_SweepContactSheetBuilder,
    "NV_SweepVideoContactSheetBuilder": NV_SweepVideoContactSheetBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SweepContactSheetPlanner": "NV Sweep Contact Sheet Planner",
    "NV_SweepContactSheetBuilder": "NV Sweep Contact Sheet Builder",
    "NV_SweepVideoContactSheetBuilder": "NV Sweep Video Contact Sheet Builder",
}
