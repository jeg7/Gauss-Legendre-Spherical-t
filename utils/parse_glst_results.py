from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

TIMINGS = {
    "assign_ms": "Assign atoms to cells",
    "calc_sf_ms": "Calculate structure factors",
    "sum_rmt_ms": "Sum remote structure factors",
    "calc_lr_ms": "Calculate long-range energy and forces",
    "calc_sr_ms": "Calculate short-range energy and forces",
    "comm_ms": "Communicate energies and forces",
    "lr_total_ms": "Long-Range GLST Runtime",
    "total_ms": "Total GLST Runtime",
}

ERROR_SECTIONS = {"fx": "X Force", "fy": "Y Force", "fz": "Z Force", "en": "Energy"}
ERROR_STATS = {
    "min": "Absolute Min",
    "max": "Absolute Max",
    "avg": "Average",
    "rms": "Root Mean Squared",
}

FIELDS = [
    "raw_file",
    "sweep",
    "sys",
    "tol",
    "box_dim",
    "cut",
    "requested_gpus",
    "num_gpus",
    "num_atoms",
    "system_x",
    "system_y",
    "system_z",
    "ncell_x",
    "ncell_y",
    "ncell_z",
    "cell_x",
    "cell_y",
    "cell_z",
    "space_x",
    "space_y",
    "space_z",
    "alpha_groups",
    "cubature_nodes_total",
    "cubature_nodes_by_group",
    "iterations",
    "coulomb_ms",
    "target_error",
    "assign_ms",
    "assign_std_ms",
    "calc_sf_ms",
    "calc_sf_std_ms",
    "sum_rmt_ms",
    "sum_rmt_std_ms",
    "calc_lr_ms",
    "calc_lr_std_ms",
    "calc_sr_ms",
    "calc_sr_std_ms",
    "comm_ms",
    "comm_std_ms",
    "lr_total_ms",
    "lr_total_std_ms",
    "total_ms",
    "total_std_ms",
]

for comp in ("fx", "fy", "fz", "en"):
    for stat in ("min", "max", "avg", "rms"):
        FIELDS.append(f"err_{comp}_{stat}")
        FIELDS.append(f"norm_{comp}_{stat}")


def parse_meta(line: str) -> dict[str, str]:
    fields: dist[str, str] = {}
    for item in line.strip().split()[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


def float_cast(value: str) -> float:
    return float(value.replace("+", ""))


def parse_singlet(pattern: str, text: str, cast=str, default=""):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return default
    return cast(match.group(1))


def parse_triplet(pattern: str, text: str):
    match = re.search(pattern, text)
    if not match:
        return ("", "", "")
    return tuple(match.group(i) for i in range(1, 4))


def parse_file(path: Path) -> dict[str, str | int | float]:
    text = path.read_text(errors="replace")
    row: dist[str, str | int | float] = {"raw_file": str(path)}

    meta_match = re.search(r"^BENCH_INPUT\s+.*$", text, re.MULTILINE)
    if meta_match:
        row.update(parse_meta(meta_match.group(0)))

    row.setdefault("sweep", "unknown")
    row.setdefault("sys", "unknown")
    row.setdefault("tol," "")
    row.setdefault("box_dim", "")
    row.setdefault("cut", "")
    row.setdefault("requested_gpus", "")

    row["num_atoms"] = parse_singlet(r"Number of atoms:\s*(\d+)", text, int)
    row["num_gpus"] = parse_singlet(r"Number of GPUs:\s*(\d+)", text, int)
    row["alpha_groups"] = parse_singlet(r"Number of alpha groups:\s*(\d+)", text, int)
    row["cubature_nodes_total"] = parse_singlet(
        r"Total number of cubature nodes:\s*(\d+)", text, int
    )
    row["iterations"] = parse_singlet(r"Finished\s+(\d+)\s+calculations", text, int)
    row["coulomb_ms"] = parse_singlet(
        rf"Coulomb Runtime:\s*({FLOAT})\s*ms", text, float_cast
    )
    row["target_eror"] = parse_singlet(rf"Target error:\s*({FLOAT})", text, float_cast)

    sx, sy, sz = parse_triplet(
        rf"System dimensions \[A\]:\s*({FLOAT})\s*x\s*({FLOAT})\s*x\s*({FLOAT})", text
    )
    row.update({"system_x": sx, "system_y": sy, "system_z": sz})

    nx, ny, nz = parse_triplet(r"Number of cells:\s*(\d+),\s*(\d+),\s*(\d+)", text)
    row.update({"ncell_x": nx, "ncell_y": ny, "ncell_z": nz})

    cx, cy, cz = parse_triplet(
        rf"Cell dimensions \[A\]:\s*({FLOAT})\s*x\s*({FLOAT})\s*x\s*({FLOAT})", text
    )
    row.update({"cell_x": cx, "cell_y": cy, "cell_z": cz})

    tx, ty, tz = parse_triplet(
        rf"Total space covered \[A\]:\s*({FLOAT})\s*x\s*({FLOAT}\s*x\s*({FLOAT}))", text
    )
    row.update({"space_x": tx, "space_y": ty, "space_z": tz})

    by_group: list[str] = []
    for match in re.finditer(f"Number of cubature nodes in group\s+\d+:\s*(\d+)", text):
        by_group.append(match.group(1))
    row["cubature_nodes_by_group"] = "|".join(by_group)

    for key, label in TIMINGS.items():
        match = re.search(
            rf"{re.escape(label)}:\s*({FLOAT})\s*ms(?:\s*\(\+/-\s*({FLOAT})\s*ms\))?",
            text,
        )
        if match:
            row[key] = float_cast(match.group(1))
            row[f"{key[:-3]}_std_ms"] = (
                float_cast(match.group(2)) if match.group(2) else ""
            )

    for comp, section in ERROR_SECTIONS.items():
        block_match = re.search(
            rf"{re.escape(section)}\s+Error:\s*"
            rf"(.*?)(?=\n\s*(?:X Force|Y Force|Z Force|Energy)\s+Error:|\Z)",
            text,
            re.DOTALL,
        )
        if not block_match:
            continue

        block = block_match.group(1)
        for stat, label in ERROR_STATS.items():
            match = re.search(
                rf"{re.escape(label)}:\s*({FLOAT})\s*\(Norm:\s*({FLOAT})\s*\)", block
            )
            if match:
                row[f"err_{comp}_{stat}"] = float_cast(match.group(1))
                row[f"norm_{comp}_{stat}"] = float_cast(match.group(2))

    return row


def sort_key(row: dist[str, object]):
    def func(name: str) -> float:
        try:
            return float(row.get(name, 0) or 0)
        except ValueError:
            return math.inf

    return (
        str(row.get("sweep", "")),
        func("requested_gpus"),
        str(row.get("sys", "")),
        func("tol"),
        func("cut"),
        str(row.get("raw_file", "")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse GLST full_example output files into one CSV."
    )
    parser.add_argument("files", nargs="+", help="Raw full_example output files")
    parser.add_argument(
        "--out", default="benchmark_results.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    rows = [parse_file(Path(name)) for name in args.files]
    rows.sort(key=sort_key)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} file(s) -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
