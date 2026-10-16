"""
gcode_generator.py
Generates machine-ready CNC G-code with Traveling Salesman Problem (TSP) rapid path optimization.
Runs 100% offline.
"""

import math
from typing import List, Dict, Tuple

def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

def optimize_toolpath_sequence_tsp(placed_parts: List[Dict]) -> List[Dict]:
    """
    Optimizes the cutting sequence using the Nearest-Neighbor Traveling Salesman (TSP) heuristic.
    Minimizes non-productive laser/spindle rapid travel ('air moves' G00) across parts.
    """
    if not placed_parts:
        return []

    unvisited = list(placed_parts)
    current_pos = (0.0, 0.0) # Machine home / zero position
    ordered = []

    while unvisited:
        # Find nearest unvisited part starting corner
        nearest_part = min(
            unvisited,
            key=lambda p: euclidean_distance(current_pos, (p['position'][0], p['position'][1]))
        )
        ordered.append(nearest_part)
        unvisited.remove(nearest_part)
        # Update current position to part end
        coords = list(nearest_part['polygon'].exterior.coords)
        current_pos = coords[-1]

    return ordered

def generate_cnc_gcode(
    placed_parts: List[Dict],
    feedrate_cut: float = 1200.0,
    feedrate_rapid: float = 3000.0,
    laser_power: int = 800,
    sheet_thickness: float = 3.0
) -> Tuple[str, float, float]:
    """
    Generates standard RS-274D / ISO CNC G-code program with toolpath statistics.
    Returns: (gcode_string, total_cut_distance_mm, total_rapid_distance_mm)
    """
    ordered_parts = optimize_toolpath_sequence_tsp(placed_parts)
    
    gcode = []
    gcode.append("( ========================================================== )")
    gcode.append("(  OptiCut AI - Autonomous CNC G-Code Toolpath Program       )")
    gcode.append("(  Machine Target: 2D Laser / Plasma / Router CNC           )")
    gcode.append(f"(  Total Nested Parts: {len(placed_parts)}                            )")
    gcode.append(f"(  Sheet Thickness: {sheet_thickness} mm | Feed: {feedrate_cut} mm/min       )")
    gcode.append("( ========================================================== )\n")
    
    # Initialization
    gcode.append("G21          ; Set metric units (millimeters)")
    gcode.append("G90          ; Absolute positioning mode")
    gcode.append("G17          ; Select XY plane")
    gcode.append(f"F{feedrate_cut:.0f}        ; Set cutting feedrate (mm/min)")
    gcode.append("G00 Z5.0     ; Rapid move to safety clearance height\n")

    total_cut_dist = 0.0
    total_rapid_dist = 0.0
    current_pos = (0.0, 0.0)

    for idx, part in enumerate(ordered_parts):
        gcode.append(f"; --- Part #{part['id']} : {part['name']} (Rot: {part['rotation']}°) ---")
        coords = list(part['polygon'].exterior.coords)
        if not coords:
            continue

        start_pt = coords[0]
        # 1. Rapid move to start of part
        rapid_move = euclidean_distance(current_pos, start_pt)
        total_rapid_dist += rapid_move
        gcode.append(f"G00 X{start_pt[0]:.2f} Y{start_pt[1]:.2f}  ; Rapid traverse to part start")
        gcode.append("G00 Z1.0                 ; Lower tool to pierce height")
        gcode.append(f"M03 S{laser_power}             ; Laser / Spindle ON (Power: {laser_power})")
        gcode.append("G04 P0.2                 ; Dwell 0.2s for pierce\n")

        # 2. Linear cutting moves around polygon perimeter
        for next_pt in coords[1:]:
            cut_move = euclidean_distance(start_pt, next_pt)
            total_cut_dist += cut_move
            gcode.append(f"G01 X{next_pt[0]:.2f} Y{next_pt[1]:.2f}")
            start_pt = next_pt

        # 3. Retract tool at end of part
        gcode.append("M05                      ; Laser / Spindle OFF")
        gcode.append("G00 Z5.0                 ; Retract to safety height\n")
        current_pos = coords[-1]

    # Return to machine origin
    return_dist = euclidean_distance(current_pos, (0.0, 0.0))
    total_rapid_dist += return_dist
    gcode.append("; --- End of Program ---")
    gcode.append("G00 X0.0 Y0.0            ; Return to machine home (origin)")
    gcode.append("M30                      ; Program stop and rewind")

    full_gcode = "\n".join(gcode)
    return full_gcode, round(total_cut_dist, 1), round(total_rapid_dist, 1)
