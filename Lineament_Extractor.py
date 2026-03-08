"""
PCI LINE-Style Lineament Extraction from DEM
==============================================

Mimics PCI Geomatica's LINE module output using open-source Python.

Key differences from basic Hough approach:
    - Ridge/valley detection via second-derivative (like PCI's gradient approach)
    - Directional non-maximum suppression for clean edges
    - Aggressive segment extension and collinear merging
    - Multi-pass extraction at different scales
    - Hillshade backdrop rendering (matching PCI Focus display)
    - Bold red lines on gray hillshade (PCI visual style)

Install dependencies:
    pip install numpy scipy matplotlib scikit-image rasterio geopandas shapely fiona

Usage:
    1. Edit DEM_PATH and OUTPUT_DIR
    2. Set MAX_LINEAMENTS (0 = all, or e.g. 1000)
    3. Run:  python lineament_pci_style.py
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.transform
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union
from skimage.feature import canny
from skimage.morphology import skeletonize, thin, binary_dilation, disk
from skimage.transform import probabilistic_hough_line
from scipy.ndimage import (gaussian_filter, sobel, maximum_filter,
                            minimum_filter, uniform_filter, label)
from scipy.spatial import cKDTree
import os
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEM_PATH = r"C:\Users\user\Documents\Dodoma_DEM.tif"
OUTPUT_DIR = r"C:\Users\user\Documents"
OUTPUT_NAME = "lineaments"

# How many lineaments to export (0 = ALL)
MAX_LINEAMENTS = 0

# --- PCI LINE equivalent parameters ---
RADI = 10           # Filter radius (pixels) — like PCI RADI
GTHR = 50           # Gradient threshold (0-255 scale) — like PCI GTHR
LTHR = 15           # Minimum line length (pixels) — like PCI LTHR
FTHR = 3            # Line fitting error tolerance — like PCI FTHR
ATHR = 8            # Angular difference for linking (degrees) — like PCI ATHR
DTHR = 10           # Linking distance (pixels) — like PCI DTHR
MAX_LINE_LENGTH = 120  # Maximum line length (pixels) — splits longer lines

# --- Multi-scale detection ---
SCALES = [
    {"sigma": 1.0, "min_length": 15, "line_gap": 2},   # Fine detail
    {"sigma": 2.0, "min_length": 20, "line_gap": 3},   # Medium features
    {"sigma": 3.0, "min_length": 25, "line_gap": 4},   # Regional structures
]

# --- Hillshade azimuths ---
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
ALTITUDE = 45

# --- Importance weights ---
WEIGHT_LENGTH = 0.35
WEIGHT_STRAIGHTNESS = 0.20
WEIGHT_GRADIENT = 0.25
WEIGHT_DENSITY = 0.20


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_hillshade(dem, azimuth=315, altitude=45):
    """Standard hillshade for display."""
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(dem)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = (np.sin(alt_rad) * np.cos(slope)
          + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    hs = np.clip((hs + 1) / 2, 0, 1)
    return hs


def compute_gradient_magnitude(dem, sigma=1.0):
    """Compute gradient magnitude (like PCI's edge strength)."""
    smoothed = gaussian_filter(dem, sigma=sigma)
    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    return mag, np.arctan2(gy, gx)


def compute_ridges_valleys(dem, sigma=2.0):
    """
    Detect ridges and valleys using eigenvalues of the Hessian matrix.
    This is closer to what PCI LINE does internally.
    """
    smoothed = gaussian_filter(dem, sigma=sigma)
    dy, dx = np.gradient(smoothed)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)

    # Eigenvalues of Hessian
    trace = dxx + dyy
    det = dxx * dyy - dxy * dyx
    discriminant = np.sqrt(np.maximum(trace**2 - 4 * det, 0))

    lambda1 = (trace + discriminant) / 2
    lambda2 = (trace - discriminant) / 2

    # Ridges: large negative eigenvalue, valleys: large positive
    ridge_strength = np.maximum(-lambda2, 0)
    valley_strength = np.maximum(lambda1, 0)

    # Combine
    combined = ridge_strength + valley_strength
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-10)
    return combined


def non_maximum_suppression(magnitude, direction):
    """
    Thin edges by suppressing non-maximum pixels along gradient direction.
    Produces cleaner, single-pixel-wide edges like PCI.
    """
    rows, cols = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i-1, j], magnitude[i+1, j]
            else:
                n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                suppressed[i, j] = magnitude[i, j]

    return suppressed


def multi_azimuth_edges(dem_smooth, azimuths, altitude, sigma, canny_low, canny_high):
    """Edge detection from multiple hillshade directions."""
    h, w = dem_smooth.shape
    combined = np.zeros((h, w), dtype=bool)
    for az in azimuths:
        hs = compute_hillshade(dem_smooth, azimuth=az, altitude=altitude)
        edges = canny(hs, sigma=sigma, low_threshold=canny_low, high_threshold=canny_high)
        combined = combined | edges
    return combined


def extract_lines_multiscale(edge_map, ridge_map, scales):
    """
    Extract lines at multiple scales and combine.
    Coarser scales catch regional structures, finer scales catch detail.
    """
    all_lines = []

    # From edge map
    skeleton = skeletonize(edge_map)
    for scale in scales:
        lines = probabilistic_hough_line(
            skeleton,
            threshold=10,
            line_length=scale["min_length"],
            line_gap=scale["line_gap"]
        )
        all_lines.extend(lines)

    # From ridge/valley map
    ridge_binary = ridge_map > np.percentile(ridge_map, 75)
    ridge_skeleton = skeletonize(ridge_binary)
    for scale in scales:
        lines = probabilistic_hough_line(
            ridge_skeleton,
            threshold=8,
            line_length=scale["min_length"],
            line_gap=scale["line_gap"]
        )
        all_lines.extend(lines)

    return all_lines


def deduplicate_segments(lines_px, dist_threshold=8):
    """Remove near-duplicate segments (common with multi-scale extraction)."""
    if len(lines_px) < 2:
        return lines_px

    midpoints = np.array([
        ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        for p0, p1 in lines_px
    ])
    lengths = np.array([
        np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        for p0, p1 in lines_px
    ])
    angles = np.array([
        np.degrees(np.arctan2(p1[0] - p0[0], p1[1] - p0[1])) % 180
        for p0, p1 in lines_px
    ])

    tree = cKDTree(midpoints)
    keep = np.ones(len(lines_px), dtype=bool)

    for i in range(len(lines_px)):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(midpoints[i], dist_threshold)
        for j in neighbors:
            if j <= i or not keep[j]:
                continue
            angle_diff = abs(angles[i] - angles[j])
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff < 10 and abs(lengths[i] - lengths[j]) < dist_threshold:
                # Keep the longer one
                if lengths[j] > lengths[i]:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return [lines_px[i] for i in range(len(lines_px)) if keep[i]]


def extend_segment(p0, p1, extension_px=5):
    """Extend a line segment by a few pixels on each end."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = np.hypot(dx, dy)
    if length == 0:
        return p0, p1
    ux, uy = dx / length, dy / length
    new_p0 = (p0[0] - ux * extension_px, p0[1] - uy * extension_px)
    new_p1 = (p1[0] + ux * extension_px, p1[1] + uy * extension_px)
    return new_p0, new_p1


def merge_collinear_segments(lines_px, max_dist, angle_tol):
    """
    Aggressively merge collinear segments — key to getting PCI-style
    long continuous lineaments instead of fragments.
    """
    if len(lines_px) == 0:
        return lines_px

    segments = []
    for (c0, r0), (c1, r1) in lines_px:
        angle = np.degrees(np.arctan2(c1 - c0, r1 - r0)) % 180
        segments.append({
            "p0": np.array([c0, r0], dtype=np.float64),
            "p1": np.array([c1, r1], dtype=np.float64),
            "angle": angle,
            "merged": False
        })

    # Single merge pass only — keeps lines short
    for merge_pass in range(1):
        endpoints = []
        ep_to_seg = []
        for i, seg in enumerate(segments):
            if seg["merged"]:
                continue
            endpoints.append(seg["p0"])
            ep_to_seg.append((i, 0))
            endpoints.append(seg["p1"])
            ep_to_seg.append((i, 1))

        if len(endpoints) < 2:
            break

        ep_array = np.array(endpoints)
        tree = cKDTree(ep_array)

        new_segments = []
        used = set()

        for idx in range(len(endpoints)):
            seg_i, end_i = ep_to_seg[idx]
            if seg_i in used or segments[seg_i]["merged"]:
                continue

            neighbors = tree.query_ball_point(ep_array[idx], max_dist)
            best_j = None
            best_dist = max_dist + 1

            for n_idx in neighbors:
                seg_j, end_j = ep_to_seg[n_idx]
                if seg_j == seg_i or seg_j in used or segments[seg_j]["merged"]:
                    continue

                angle_diff = abs(segments[seg_i]["angle"] - segments[seg_j]["angle"])
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                if angle_diff > angle_tol:
                    continue

                d = np.linalg.norm(ep_array[idx] - ep_array[n_idx])
                if d < best_dist:
                    best_dist = d
                    best_j = (seg_j, end_j, n_idx)

            if best_j is not None:
                seg_j, end_j, _ = best_j
                # Merge: take the two farthest endpoints
                all_pts = [
                    segments[seg_i]["p0"], segments[seg_i]["p1"],
                    segments[seg_j]["p0"], segments[seg_j]["p1"]
                ]
                pts = np.array(all_pts)
                dists_matrix = np.sqrt(
                    (pts[:, 0:1] - pts[:, 0:1].T)**2
                    + (pts[:, 1:2] - pts[:, 1:2].T)**2
                )
                i_max, j_max = np.unravel_index(dists_matrix.argmax(), dists_matrix.shape)

                new_p0 = pts[i_max]
                new_p1 = pts[j_max]
                new_angle = np.degrees(np.arctan2(
                    new_p1[0] - new_p0[0], new_p1[1] - new_p0[1]
                )) % 180

                segments[seg_i]["merged"] = True
                segments[seg_j]["merged"] = True
                used.add(seg_i)
                used.add(seg_j)

                new_segments.append({
                    "p0": new_p0,
                    "p1": new_p1,
                    "angle": new_angle,
                    "merged": False
                })

        # Add unmerged segments
        for i, seg in enumerate(segments):
            if not seg["merged"] and i not in used:
                new_segments.append(seg)

        segments = new_segments

    return [
        (tuple(seg["p0"]), tuple(seg["p1"]))
        for seg in segments
        if not seg["merged"]
    ]


def remove_isolated_segments(lines_px, min_neighbors=1, radius=60):
    if len(lines_px) < 3:
        return lines_px
    midpoints = np.array([
        ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        for p0, p1 in lines_px
    ])
    tree = cKDTree(midpoints)
    return [
        lines_px[i] for i, mid in enumerate(midpoints)
        if len(tree.query_ball_point(mid, radius)) - 1 >= min_neighbors
    ]


def split_long_segments(lines_px, max_length):
    """Split any segment longer than max_length into shorter pieces."""
    result = []
    for (c0, r0), (c1, r1) in lines_px:
        length = np.hypot(c1 - c0, r1 - r0)
        if length <= max_length:
            result.append(((c0, r0), (c1, r1)))
        else:
            # Split into N equal parts
            n_parts = int(np.ceil(length / max_length))
            for k in range(n_parts):
                t0 = k / float(n_parts)
                t1 = (k + 1) / float(n_parts)
                sc0 = c0 + (c1 - c0) * t0
                sr0 = r0 + (r1 - r0) * t0
                sc1 = c0 + (c1 - c0) * t1
                sr1 = r0 + (r1 - r0) * t1
                result.append(((sc0, sr0), (sc1, sr1)))
    return result


def pixel_to_coords(lines_px, transform):
    geo_lines = []
    for (col0, row0), (col1, row1) in lines_px:
        x0, y0 = rasterio.transform.xy(transform, int(row0), int(col0))
        x1, y1 = rasterio.transform.xy(transform, int(row1), int(col1))
        geo_lines.append(LineString([(x0, y0), (x1, y1)]))
    return geo_lines


def compute_azimuth(line):
    coords = list(line.coords)
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    return round(np.degrees(np.arctan2(dx, dy)) % 180, 1)


def compute_sinuosity(line):
    coords = list(line.coords)
    straight = np.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
    if straight == 0:
        return 999.0
    return round(line.length / straight, 3)


def sample_gradient_along_line(line_px, gradient_mag):
    (c0, r0), (c1, r1) = line_px
    num = max(int(np.hypot(c1 - c0, r1 - r0)), 2)
    cols = np.linspace(c0, c1, num).astype(int)
    rows = np.linspace(r0, r1, num).astype(int)
    rows = np.clip(rows, 0, gradient_mag.shape[0] - 1)
    cols = np.clip(cols, 0, gradient_mag.shape[1] - 1)
    return np.mean(gradient_mag[rows, cols])


def compute_importance_scores(gdf, lines_px, gradient_mag):
    n = len(gdf)
    if n == 0:
        return gdf

    # Length score
    lengths = gdf["length"].values
    len_score = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-10)

    # Straightness score
    sinuosity = gdf["sinuosity"].values
    straightness = 1.0 / np.clip(sinuosity, 1.0, None)
    str_score = (straightness - straightness.min()) / (straightness.max() - straightness.min() + 1e-10)

    # Gradient score
    grad_vals = np.array([sample_gradient_along_line(lp, gradient_mag) for lp in lines_px])
    grad_score = (grad_vals - grad_vals.min()) / (grad_vals.max() - grad_vals.min() + 1e-10)

    # Density score
    midpoints = np.array([((p0[0]+p1[0])/2, (p0[1]+p1[1])/2) for p0, p1 in lines_px])
    tree = cKDTree(midpoints)
    dens = np.array([len(tree.query_ball_point(m, 80)) - 1 for m in midpoints], dtype=np.float64)
    dens_score = (dens - dens.min()) / (dens.max() - dens.min() + 1e-10)

    importance = (WEIGHT_LENGTH * len_score + WEIGHT_STRAIGHTNESS * str_score
                  + WEIGHT_GRADIENT * grad_score + WEIGHT_DENSITY * dens_score)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10) * 100

    gdf["importance"] = np.round(importance, 1)
    return gdf


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # ----- 1. Load DEM -----
    print("=" * 60)
    print("PCI LINE-Style Lineament Extraction")
    print("=" * 60)
    print("\nLoading DEM: " + DEM_PATH)

    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        pixel_size = src.res

    print("  Shape: %d x %d" % dem.shape)
    print("  CRS:   " + str(crs))
    print("  Pixel: %.4f x %.4f" % (pixel_size[0], pixel_size[1]))

    if nodata is not None:
        mask = dem == nodata
        dem[mask] = np.nanmedian(dem[~mask])

    dem_smooth = gaussian_filter(dem, sigma=1.0)

    # ----- 2. Compute gradient (like PCI RADI + GTHR) -----
    print("\nStep 1: Computing gradient magnitude (RADI=%d)..." % RADI)
    grad_mag, grad_dir = compute_gradient_magnitude(dem, sigma=float(RADI) / 3.0)
    grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-10)

    # Apply gradient threshold (like PCI GTHR)
    gthr_normalized = GTHR / 255.0
    strong_edges = grad_norm > gthr_normalized
    print("  Gradient pixels above threshold: %d" % strong_edges.sum())

    # Non-maximum suppression for thin edges
    print("  Applying non-maximum suppression...")
    nms = non_maximum_suppression(grad_norm, grad_dir)
    nms_binary = nms > gthr_normalized

    # ----- 3. Ridge/valley detection (Hessian-based) -----
    print("\nStep 2: Detecting ridges and valleys (Hessian eigenvalues)...")
    ridge_map = compute_ridges_valleys(dem_smooth, sigma=2.0)

    # ----- 4. Multi-azimuth hillshade edges -----
    print("\nStep 3: Multi-azimuth edge detection (%d directions)..." % len(AZIMUTHS))
    hs_edges = multi_azimuth_edges(dem_smooth, AZIMUTHS, ALTITUDE,
                                    sigma=1.5, canny_low=0.04, canny_high=0.12)

    # ----- 5. Combine all edge sources -----
    print("\nStep 4: Combining edge sources...")
    combined_edges = nms_binary | hs_edges
    print("  Combined edge pixels: %d" % combined_edges.sum())

    # Density filter
    density = uniform_filter(combined_edges.astype(np.float64), size=31)
    density_thresh = np.percentile(density[combined_edges], 8)
    combined_edges = combined_edges & (density >= density_thresh)
    print("  After density filter: %d" % combined_edges.sum())

    # Skeletonize
    skeleton = skeletonize(combined_edges)

    # ----- 6. Multi-scale line extraction -----
    print("\nStep 5: Multi-scale line extraction (%d scales)..." % len(SCALES))
    all_lines = extract_lines_multiscale(skeleton, ridge_map, SCALES)
    print("  Raw segments (all scales): %d" % len(all_lines))

    if len(all_lines) == 0:
        print("No lineaments found. Try lowering GTHR.")
        return

    # ----- 7. Deduplicate -----
    print("\nStep 6: Deduplicating overlapping segments...")
    all_lines = deduplicate_segments(all_lines, dist_threshold=8)
    print("  After deduplication: %d" % len(all_lines))

    # ----- 8. Remove isolated -----
    print("  Removing isolated segments...")
    all_lines = remove_isolated_segments(all_lines, min_neighbors=1, radius=60)
    print("  After isolation filter: %d" % len(all_lines))

    # ----- 9. Extend segments slightly -----
    print("  Extending segment endpoints...")
    all_lines = [extend_segment(p0, p1, extension_px=1) for p0, p1 in all_lines]

    # ----- 10. Aggressive collinear merging (like PCI ATHR + DTHR) -----
    print("\nStep 7: Merging collinear segments (ATHR=%d, DTHR=%d)..." % (ATHR, DTHR))
    all_lines = merge_collinear_segments(all_lines, max_dist=DTHR, angle_tol=ATHR)
    print("  After merging: %d" % len(all_lines))

    # ----- 11. Length filter (like PCI LTHR) -----
    all_lines = [
        (p0, p1) for p0, p1 in all_lines
        if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= LTHR
    ]
    print("  After length filter (LTHR=%d): %d" % (LTHR, len(all_lines)))

    # Split any segments that are too long
    print("  Splitting segments longer than %d px..." % MAX_LINE_LENGTH)
    all_lines = split_long_segments(all_lines, max_length=MAX_LINE_LENGTH)
    print("  After splitting: %d" % len(all_lines))

    if len(all_lines) == 0:
        print("No lineaments survived filtering.")
        return

    # ----- 12. Convert to geographic coordinates -----
    print("\nStep 8: Converting to geographic coordinates...")
    geo_lines = pixel_to_coords(all_lines, transform)

    # ----- 13. Build GeoDataFrame -----
    gdf = gpd.GeoDataFrame({
        "id": range(1, len(geo_lines) + 1),
        "length": [round(l.length, 4) for l in geo_lines],
        "azimuth": [compute_azimuth(l) for l in geo_lines],
        "sinuosity": [compute_sinuosity(l) for l in geo_lines],
        "geometry": geo_lines
    }, crs=crs)

    # ----- 14. Importance scoring -----
    print("\nStep 9: Computing importance scores...")
    gdf = compute_importance_scores(gdf, all_lines, grad_norm)
    gdf = gdf.sort_values("importance", ascending=False).reset_index(drop=True)
    gdf["id"] = range(1, len(gdf) + 1)
    gdf["rank"] = range(1, len(gdf) + 1)

    total_found = len(gdf)
    print("  Total lineaments: %d" % total_found)

    # ----- 15. Apply limit with SPATIAL BALANCE -----
    # Instead of just taking top N (which clusters in one area),
    # divide the map into a grid and pick top lineaments from EACH cell.
    # This ensures lineaments are spread across the entire DEM.
    if MAX_LINEAMENTS > 0 and MAX_LINEAMENTS < total_found:
        print("  Selecting %d lineaments with spatial balance..." % MAX_LINEAMENTS)

        # Compute midpoint of each lineament in pixel space
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        mid_x = gdf.geometry.apply(lambda g: (g.coords[0][0] + g.coords[-1][0]) / 2)
        mid_y = gdf.geometry.apply(lambda g: (g.coords[0][1] + g.coords[-1][1]) / 2)

        # Create grid — auto-size to get roughly even distribution
        # More cells = more even spread, fewer per cell
        n_grid = max(int(np.sqrt(MAX_LINEAMENTS / 3)), 4)
        x_bins = np.linspace(bounds[0], bounds[2], n_grid + 1)
        y_bins = np.linspace(bounds[1], bounds[3], n_grid + 1)

        # Assign each lineament to a grid cell
        gdf["_grid_x"] = np.digitize(mid_x, x_bins) - 1
        gdf["_grid_y"] = np.digitize(mid_y, y_bins) - 1
        gdf["_grid_x"] = gdf["_grid_x"].clip(0, n_grid - 1)
        gdf["_grid_y"] = gdf["_grid_y"].clip(0, n_grid - 1)
        gdf["_cell"] = gdf["_grid_x"].astype(str) + "_" + gdf["_grid_y"].astype(str)

        # Count occupied cells
        occupied_cells = gdf["_cell"].nunique()
        base_per_cell = MAX_LINEAMENTS // max(occupied_cells, 1)
        remainder = MAX_LINEAMENTS - (base_per_cell * occupied_cells)

        print("    Grid: %d x %d = %d cells, %d occupied" % (n_grid, n_grid, n_grid * n_grid, occupied_cells))
        print("    Base per cell: %d, remainder: %d" % (base_per_cell, remainder))

        # From each cell, pick top N by importance (already sorted)
        selected_indices = []
        cell_counts = {}

        for cell_id, cell_gdf in gdf.groupby("_cell"):
            # Sort within cell by importance
            cell_sorted = cell_gdf.sort_values("importance", ascending=False)
            n_take = base_per_cell
            cell_counts[cell_id] = min(n_take, len(cell_sorted))
            selected_indices.extend(cell_sorted.index[:n_take].tolist())

        # Distribute remainder to cells that still have more lineaments
        if remainder > 0:
            remaining_gdf = gdf.drop(index=selected_indices)
            if len(remaining_gdf) > 0:
                remaining_sorted = remaining_gdf.sort_values("importance", ascending=False)
                extra = remaining_sorted.head(remainder)
                selected_indices.extend(extra.index.tolist())

        gdf = gdf.loc[selected_indices].copy()
        gdf = gdf.drop(columns=["_grid_x", "_grid_y", "_cell"])
        gdf = gdf.sort_values("importance", ascending=False).reset_index(drop=True)
        gdf["id"] = range(1, len(gdf) + 1)
        gdf["rank"] = range(1, len(gdf) + 1)
        print("  Exported: %d lineaments (spatially balanced)" % len(gdf))
    else:
        print("  Exporting ALL %d lineaments" % total_found)

    # ----- 16. Save shapefile -----
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + ".shp")
    gdf.to_file(out_path)
    print("\nShapefile saved: " + out_path)
    print("  Attributes: id, rank, length, azimuth, sinuosity, importance")

    # ----- 17. Summary -----
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print("  Total exported:   %d" % len(gdf))
    print("  Mean length:      %.4f" % gdf["length"].mean())
    print("  Max length:       %.4f" % gdf["length"].max())
    print("  Mean azimuth:     %.1f deg" % gdf["azimuth"].mean())
    print("  Mean importance:  %.1f / 100" % gdf["importance"].mean())

    print("\n  Importance distribution:")
    for t in [90, 75, 50, 25]:
        print("    Score >= %d:  %d lineaments" % (t, (gdf["importance"] >= t).sum()))

    # ----- 18. PCI-style preview (hillshade + bold red lines) -----
    print("\nGenerating PCI-style preview...")
    hillshade_display = compute_hillshade(dem, azimuth=315, altitude=45)

    # Full extent view
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(hillshade_display, cmap="gray", vmin=0, vmax=1)
    gdf.plot(ax=ax, color="red", linewidth=1.2, alpha=0.85)
    ax.set_title("Lineaments (%d) — PCI LINE Style" % len(gdf), fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    preview_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + "_preview.png")
    plt.savefig(preview_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()
    print("  Preview saved: " + preview_path)

    # Zoomed detail view (center 25% of image)
    h, w = dem.shape
    r0, r1 = int(h * 0.375), int(h * 0.625)
    c0, c1 = int(w * 0.375), int(w * 0.625)

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(hillshade_display, cmap="gray", vmin=0, vmax=1)
    gdf.plot(ax=ax2, color="red", linewidth=1.8, alpha=0.9)
    # Convert pixel bounds to geographic for zoom
    x_min, y_max = rasterio.transform.xy(transform, r0, c0)
    x_max, y_min = rasterio.transform.xy(transform, r1, c1)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("Zoomed Detail — Center Region", fontsize=13)
    ax2.axis("off")
    plt.tight_layout()

    zoom_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + "_zoom.png")
    plt.savefig(zoom_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()
    print("  Zoom saved: " + zoom_path)

    # ----- 19. Rose diagram -----
    fig3, ax3 = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    az_vals = np.radians(gdf["azimuth"].values)
    az_full = np.concatenate([az_vals, az_vals + np.pi])
    ax3.hist(az_full, bins=72, color="red", edgecolor="darkred", alpha=0.7)
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)
    ax3.set_title("Lineament Rose Diagram (n=%d)" % len(gdf), pad=20, fontsize=13)

    rose_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + "_rose.png")
    plt.savefig(rose_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print("  Rose diagram saved: " + rose_path)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
