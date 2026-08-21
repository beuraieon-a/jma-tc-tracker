"""Fetch JMA tropical cyclone data and maintain a cumulative archive.

Data source: Japan Meteorological Agency "bosai" typhoon JSON API
    https://www.jma.go.jp/bosai/typhoon/data/

Designed to run repeatedly (e.g. hourly via GitHub Actions). For every active
system it maintains, under <outdir>/<base>/:
    1. besttrack.csv          -> cumulative operational best track, one row per
                                 analysis time (deduplicated by valid_time_utc)
    2. forecast/<base>_<issue>_forecast.json             -> every issuance (pretty)
    3. specifications/<base>_<issue>_specifications.json -> every issuance (pretty)
    plus meta.json (per system) and a top-level index.json manifest that marks
    each system active or terminated for the dashboard.

<base> follows the official typhoon number, 'T{number}_{NAME}' (e.g.
T2618_SAUDEL); an unnamed depression uses '{tc_id}_TD_{number}'
(e.g. TC2622_TD_b) and is renamed automatically once it is upgraded and named.

Usage:
    python fetch_jma_tc.py --all              # fetch every active system
    python fetch_jma_tc.py --tc TC2615        # one system by id / number
    python fetch_jma_tc.py --list             # list active systems and exit
    python fetch_jma_tc.py --reindex          # rebuild index.json only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE_URL = "https://www.jma.go.jp/bosai/typhoon/data"
TARGET_URL = f"{BASE_URL}/targetTc.json"
DEFAULT_OUTDIR = "data"
USER_AGENT = "Mozilla/5.0 (JMA-TC-fetch; research use)"
# Maps a stable tropicalCyclone id to its current folder base name so an
# upgraded depression can be detected and renamed on a later run.
REGISTRY_NAME = ".tc_registry.json"


# --------------------------------------------------------------------------- #
# Networking
# --------------------------------------------------------------------------- #
def fetch_json(url: str, attempts: int = 3, backoff: float = 2.0):
    """Return parsed JSON for a URL, retrying transient network errors."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
    raise last  # type: ignore[misc]


def get_active_systems() -> list[dict]:
    """List of currently tracked cyclones from targetTc.json."""
    data = fetch_json(TARGET_URL)
    return data if isinstance(data, list) else []


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def match_system(system: dict, selection: str) -> bool:
    """True if a user selection string identifies this system.

    Matches (case-insensitive) against the tropicalCyclone id (with or without
    the leading "TC"), the typhoon number, and the category. Matching is kept
    lenient so tropical-depression designations (e.g. "TD b") can be caught
    once JMA publishes them.
    """
    sel = selection.strip().lower()
    tc_id = str(system.get("tropicalCyclone", "")).lower()
    number = str(system.get("typhoonNumber", "")).lower()
    category = str(system.get("category", "")).lower()
    candidates = {
        tc_id,
        tc_id.removeprefix("tc"),
        number,
        category,
        f"{category} {number}".strip(),
    }
    if sel in candidates and sel:
        return True
    # Fallback: substring against the raw record (covers unexpected TD labels).
    return sel in json.dumps(system, ensure_ascii=False).lower()


def resolve_targets(active: list[dict], selection: str | None) -> list[dict]:
    if selection is None or selection.strip().lower() == "all":
        return active
    matches = [s for s in active if match_system(s, selection)]
    return matches


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def part_name(entry: dict) -> str:
    """English label of a data 'part' (handles both str and {jp,en} forms)."""
    part = entry.get("part")
    if isinstance(part, dict):
        return part.get("en", "")
    return part or ""


def split_forecast(forecast: list[dict]) -> tuple[dict, dict | None, list[dict]]:
    """Return (title, analysis_entry, forecast_entries) from forecast.json."""
    title: dict = {}
    analysis: dict | None = None
    forecasts: list[dict] = []
    for entry in forecast:
        if entry.get("part") == "title":
            title = entry
        elif entry.get("advancedHours") == 0 or part_name(entry) == "Analysis":
            analysis = entry
        elif "center" in entry or "probabilityCircle" in entry:
            forecasts.append(entry)
    return title, analysis, forecasts


def index_specs(specs: list[dict]) -> tuple[dict | None, dict]:
    """Return (analysis_spec, {validtime_UTC: spec}) from specifications.json."""
    analysis_spec: dict | None = None
    by_time: dict[str, dict] = {}
    for spec in specs:
        if spec.get("part") == "title":
            continue
        vt = (spec.get("validtime") or {}).get("UTC")
        if vt:
            by_time[vt] = spec
        if part_name(spec) == "Analysis":
            analysis_spec = spec
    return analysis_spec, by_time


COMPASS = {
    "北": "N", "北北東": "NNE", "北東": "NE", "東北東": "ENE",
    "東": "E", "東南東": "ESE", "南東": "SE", "南南東": "SSE",
    "南": "S", "南南西": "SSW", "南西": "SW", "西南西": "WSW",
    "西": "W", "西北西": "WNW", "北西": "NW", "北北西": "NNW",
}


def compass_dir(kanji) -> str:
    """JMA 16-point compass kanji -> English abbreviation ('' if unknown)."""
    return COMPASS.get(str(kanji).strip(), "")


def intensity_category(category_en, intensity_jp) -> str:
    """Merge JMA category + intensity into one international-scale label."""
    cat = (category_en or "").strip()
    base = {
        "TD": "tropical depression",
        "TS": "tropical storm",
        "STS": "severe tropical storm",
        "LOW": "extratropical",
    }
    if cat in base:
        return base[cat]
    if cat == "TY":
        return {
            "強い": "typhoon",
            "非常に強い": "very strong typhoon",
            "猛烈な": "violent typhoon",
        }.get((intensity_jp or "").strip(), "typhoon")
    return cat


def size_category(scale_jp) -> str:
    """JMA scale kanji -> size label ('' when small/medium or not applicable)."""
    s = (scale_jp or "").strip()
    if s in ("大型", "大"):
        return "large"
    if s in ("超大型", "超大"):
        return "very_large"
    return ""


def area_label(area) -> str:
    """Wind-field sector -> compass abbreviation, or 'symmetric' for 全域/All."""
    if isinstance(area, dict):
        if area.get("en") == "All" or area.get("jp") == "全域":
            return "symmetric"
        return compass_dir(area.get("jp", "")) or area.get("en", "")
    return compass_dir(area)


def parse_wind_radii(warnings) -> dict:
    """Split a wind-radius list into longest/shortest direction and length.

    Keys: long_dir, long_km, long_nm, short_dir, short_km, short_nm. A single
    '全域'/'All' entry is symmetric, so long and short radii are equal.
    """
    empty = {"long_dir": "", "long_km": "", "long_nm": "",
             "short_dir": "", "short_km": "", "short_nm": ""}
    if not warnings:
        return empty

    def pack(entry):
        rng = entry.get("range", {}) or {}
        return area_label(entry.get("area")), rng.get("km", ""), rng.get("nm", "")

    if len(warnings) == 1:
        d, km, nm = pack(warnings[0])
        return {"long_dir": d, "long_km": km, "long_nm": nm,
                "short_dir": d, "short_km": km, "short_nm": nm}

    ordered = sorted(warnings,
                     key=lambda w: (w.get("range", {}) or {}).get("km") or 0,
                     reverse=True)
    l_dir, l_km, l_nm = pack(ordered[0])
    s_dir, s_km, s_nm = pack(ordered[-1])
    return {"long_dir": l_dir, "long_km": l_km, "long_nm": l_nm,
            "short_dir": s_dir, "short_km": s_km, "short_nm": s_nm}


def meta_ids(meta: dict) -> tuple[str, str, str]:
    """(tc_id_monitoring, tc_id_official, tc_name) from a system's metadata.

    A named system uses 'T{number}' and its uppercase name. An unnamed
    depression (letter designation) leaves the official id blank and carries
    the letter as the name.
    """
    tc_id = str(meta.get("tc_id", "")).strip()
    number = str(meta.get("number", "")).strip()
    name_en = str(meta.get("name_en", "")).strip().upper()
    if number.isdigit():
        return tc_id, f"T{number}", name_en
    return tc_id, "", (name_en or number)


def spec_fields(spec: dict | None) -> dict:
    """Scalar, already-mapped parameters shared by both CSV outputs."""
    if not spec:
        return {}
    mw = spec.get("maximumWind", {})
    sustained = mw.get("sustained", {})
    gust = mw.get("gust", {})
    speed = spec.get("speed", {})
    category = spec.get("category", {})
    category_en = category.get("en") if isinstance(category, dict) else category
    course = spec.get("course", "")
    stationary = str(course).strip() == "不定"
    speed_kmh = speed.get("km/h", "")
    speed_kt = speed.get("kt", "")
    if stationary:
        # Quasi-stationary: no direction, no speed.
        movement_direction, speed_kmh, speed_kt = "almost stationary", "", ""
    else:
        movement_direction = compass_dir(course) or course
        if not speed_kmh and not speed_kt:
            # Directional but only a qualitative note (e.g. ゆっくり / Slow).
            speed_kmh = speed_kt = "slowly"
    return {
        "tc_intensity_category": intensity_category(category_en, spec.get("intensity", "")),
        "tc_size_category": size_category(spec.get("scale", "")),
        "central_pressure": spec.get("pressure", ""),
        "msw_ms": sustained.get("m/s", ""),
        "msw_kt": sustained.get("kt", ""),
        "gustiness_ms": gust.get("m/s", ""),
        "gustiness_kt": gust.get("kt", ""),
        "movement_direction": movement_direction,
        "movement_speed_kmh": speed_kmh,
        "movement_speed_kt": speed_kt,
    }


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pretty_dump(obj, path: str) -> None:
    """Write JSON indented and UTF-8 (readable / 'pretty-print' in browsers)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: str):
    """Parsed JSON at path, or None if missing/unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


BESTTRACK_HEADER = [
    "tc_id_monitoring", "tc_id_official", "tc_name",
    "valid_time_utc", "center_latitude", "center_longitude",
    "tc_intensity_category", "tc_size_category", "central_pressure",
    "msw_ms", "msw_kt", "gustiness_ms", "gustiness_kt",
    "movement_direction", "movement_speed_kmh", "movement_speed_kt",
    "30kt_wind_longrad_dir", "30kt_wind_longrad_length_km", "30kt_wind_longrad_length_nm",
    "30kt_wind_shortrad_dir", "30kt_wind_shortrad_length_km", "30kt_wind_shortrad_length_nm",
    "50kt_wind_longrad_dir", "50kt_wind_longrad_length_km", "50kt_wind_longrad_length_nm",
    "50kt_wind_shortrad_dir", "50kt_wind_shortrad_length_km", "50kt_wind_shortrad_length_nm",
]


def analysis_position(analysis_entry, analysis_spec):
    """Best available (lat, lon) for the analysis time."""
    if analysis_spec:
        pos = analysis_spec.get("position", {}).get("deg")
        if pos:
            return pos[0], pos[1]
    if analysis_entry:
        center = analysis_entry.get("center")
        if isinstance(center, list) and len(center) == 2 and not isinstance(center[0], list):
            return center[0], center[1]
        track = analysis_entry.get("track", {}) or {}
        for seg in ("typhoon", "preTyphoon"):
            pts = track.get(seg) or []
            if pts:
                return pts[-1]
    return "", ""


def best_track_row(meta, analysis_entry, analysis_spec):
    """One cumulative best-track record from the current analysis part."""
    valid = ((analysis_spec or {}).get("validtime")
             or (analysis_entry or {}).get("validtime") or {})
    vt = valid.get("UTC")
    if not vt:
        return None
    tc_mon, tc_off, tc_name = meta_ids(meta)
    lat, lon = analysis_position(analysis_entry, analysis_spec)
    sf = spec_fields(analysis_spec)
    gale = parse_wind_radii((analysis_spec or {}).get("galeWarning"))
    storm = parse_wind_radii((analysis_spec or {}).get("stormWarning"))
    return {
        "tc_id_monitoring": tc_mon, "tc_id_official": tc_off, "tc_name": tc_name,
        "valid_time_utc": vt, "center_latitude": lat, "center_longitude": lon,
        "tc_intensity_category": sf.get("tc_intensity_category", ""),
        "tc_size_category": sf.get("tc_size_category", ""),
        "central_pressure": sf.get("central_pressure", ""),
        "msw_ms": sf.get("msw_ms", ""), "msw_kt": sf.get("msw_kt", ""),
        "gustiness_ms": sf.get("gustiness_ms", ""), "gustiness_kt": sf.get("gustiness_kt", ""),
        "movement_direction": sf.get("movement_direction", ""),
        "movement_speed_kmh": sf.get("movement_speed_kmh", ""),
        "movement_speed_kt": sf.get("movement_speed_kt", ""),
        "30kt_wind_longrad_dir": gale["long_dir"],
        "30kt_wind_longrad_length_km": gale["long_km"],
        "30kt_wind_longrad_length_nm": gale["long_nm"],
        "30kt_wind_shortrad_dir": gale["short_dir"],
        "30kt_wind_shortrad_length_km": gale["short_km"],
        "30kt_wind_shortrad_length_nm": gale["short_nm"],
        "50kt_wind_longrad_dir": storm["long_dir"],
        "50kt_wind_longrad_length_km": storm["long_km"],
        "50kt_wind_longrad_length_nm": storm["long_nm"],
        "50kt_wind_shortrad_dir": storm["short_dir"],
        "50kt_wind_shortrad_length_km": storm["short_km"],
        "50kt_wind_shortrad_length_nm": storm["short_nm"],
    }


def update_besttrack(path: str, row: dict) -> int:
    """Merge an analysis record into the cumulative best track (dedup by time)."""
    rows: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for existing in csv.DictReader(f):
                rows[existing.get("valid_time_utc", "")] = existing
    rows[row["valid_time_utc"]] = row  # newest wins (analysis revisions update)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BESTTRACK_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))
    return len(rows)


# --------------------------------------------------------------------------- #
# Per-system processing
# --------------------------------------------------------------------------- #
def sanitize_issue(issue_utc: str) -> str:
    """'2026-08-09T15:45:00Z' -> '20260809T1545Z' for use in filenames."""
    digits = "".join(c for c in issue_utc if c.isdigit())
    if len(digits) >= 12:
        return f"{digits[:8]}T{digits[8:12]}Z"
    return digits or "unknown"


def build_base_name(meta: dict) -> str:
    """Folder/file base name for a system.

    Named systems (numeric typhoon number, including a named cyclone that has
    weakened back to a depression) use 'T{number}_{NAME}'. An unnamed tropical
    depression, whose typhoon number is a lowercase letter, uses
    '{tc_id}_TD_{number}'.
    """
    number = str(meta.get("number", "")).strip()
    name_en = str(meta.get("name_en", "")).strip().upper()
    tc_id = str(meta.get("tc_id", "")).strip()
    if number.isdigit():
        return f"T{number}_{name_en}" if name_en else f"T{number}"
    return f"{tc_id}_TD_{number}" if number else f"{tc_id}_TD"


def load_registry(outdir: str) -> dict:
    path = os.path.join(outdir, REGISTRY_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_registry(outdir: str, registry: dict) -> None:
    path = os.path.join(outdir, REGISTRY_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def find_stale_bases(outdir: str, tc_id: str, target_base: str,
                     registry: dict) -> list[str]:
    """Existing folders that belong to a tc_id but use a different base name.

    Covers the last registered name, the old '{tc_id}' scheme, and any
    '{tc_id}_TD_*' depression folder, so earlier data is consolidated under
    the current name.
    """
    candidates: set[str] = set()
    if registry.get(tc_id):
        candidates.add(registry[tc_id])
    td_prefix = f"{tc_id}_TD_"
    if os.path.isdir(outdir):
        for entry in os.listdir(outdir):
            if not os.path.isdir(os.path.join(outdir, entry)):
                continue
            if entry == tc_id or entry.startswith(td_prefix):
                candidates.add(entry)
    candidates.discard(target_base)
    return [c for c in candidates if os.path.isdir(os.path.join(outdir, c))]


def rename_system(outdir: str, old_base: str, new_base: str) -> None:
    """Move a system's folder tree to a new base name, re-prefixing its files."""
    old_dir = os.path.join(outdir, old_base)
    new_dir = os.path.join(outdir, new_base)
    if old_base == new_base or not os.path.isdir(old_dir):
        return
    if not os.path.isdir(new_dir):
        os.rename(old_dir, new_dir)
    else:  # merge into an existing target
        for root, _dirs, files in os.walk(old_dir):
            dest_root = os.path.join(new_dir, os.path.relpath(root, old_dir))
            os.makedirs(dest_root, exist_ok=True)
            for fn in files:
                os.replace(os.path.join(root, fn), os.path.join(dest_root, fn))
        shutil.rmtree(old_dir, ignore_errors=True)
    # Re-prefix any archived file still carrying the old base name.
    for root, _dirs, files in os.walk(new_dir):
        for fn in files:
            if fn.startswith(old_base):
                os.replace(os.path.join(root, fn),
                           os.path.join(root, new_base + fn[len(old_base):]))


def process_system(system: dict, outdir: str, registry: dict) -> None:
    tc_id = system.get("tropicalCyclone", "")
    print(f"\n=== {tc_id} (No. {system.get('typhoonNumber', '?')}, "
          f"{system.get('category', '?')}) ===")

    forecast = fetch_json(f"{BASE_URL}/{tc_id}/forecast.json")
    specs = fetch_json(f"{BASE_URL}/{tc_id}/specifications.json")

    title, analysis_entry, _forecasts = split_forecast(forecast)
    analysis_spec, _spec_by_time = index_specs(specs)

    name = title.get("name", {}) if isinstance(title.get("name"), dict) else {}
    meta = {
        "tc_id": tc_id,
        "number": title.get("typhoonNumber", system.get("typhoonNumber", "")),
        "name_en": name.get("en", ""),
        "name_jp": name.get("jp", ""),
    }
    issue = (title.get("issue", {}) or {}).get("UTC", "")
    stamp = sanitize_issue(issue)

    # Rename/consolidate earlier folders if this system's name has changed
    # (e.g. a depression that has since been upgraded and named).
    base = build_base_name(meta)
    for old_base in find_stale_bases(outdir, tc_id, base, registry):
        rename_system(outdir, old_base, base)
        print(f"  renamed: {old_base} -> {base}")
    registry[tc_id] = base

    sys_dir = os.path.join(outdir, base)
    fdir = os.path.join(sys_dir, "forecast")
    sdir = os.path.join(sys_dir, "specifications")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(sdir, exist_ok=True)

    # Outputs 2 & 3: archive every issuance, pretty-printed for readability.
    pretty_dump(forecast, os.path.join(fdir, f"{base}_{stamp}_forecast.json"))
    pretty_dump(specs, os.path.join(sdir, f"{base}_{stamp}_specifications.json"))

    # Output 1: cumulative operational best track from the analysis part.
    row = best_track_row(meta, analysis_entry, analysis_spec)
    n_bt = update_besttrack(os.path.join(sys_dir, "besttrack.csv"), row) if row else 0

    # Per-system metadata used by the dashboard manifest.
    tc_mon, tc_off, tc_name = meta_ids(meta)
    meta_path = os.path.join(sys_dir, "meta.json")
    existing = read_json(meta_path) or {}
    pretty_dump({
        "base": base,
        "tc_id_monitoring": tc_mon,
        "tc_id_official": tc_off,
        "tc_name": tc_name,
        "status": "active",
        "first_analysis_utc": existing.get("first_analysis_utc")
        or (row or {}).get("valid_time_utc", ""),
        "last_analysis_utc": (row or {}).get("valid_time_utc", ""),
        "last_issue_utc": issue,
    }, meta_path)

    label = tc_name or tc_off or tc_id
    print(f"  {label}: issue {issue}, best-track points = {n_bt}")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def build_index(outdir: str, active_ids) -> dict:
    """Rebuild index.json, marking each stored system active or terminated."""
    active_ids = set(active_ids)
    systems: list[dict] = []
    if os.path.isdir(outdir):
        for base in sorted(os.listdir(outdir)):
            sys_dir = os.path.join(outdir, base)
            meta = read_json(os.path.join(sys_dir, "meta.json"))
            if not os.path.isdir(sys_dir) or not meta:
                continue
            meta["status"] = ("active" if meta.get("tc_id_monitoring") in active_ids
                              else "terminated")
            pretty_dump(meta, os.path.join(sys_dir, "meta.json"))

            def _listing(sub: str, _dir=sys_dir, _base=base) -> list[str]:
                sub_dir = os.path.join(_dir, sub)
                if not os.path.isdir(sub_dir):
                    return []
                return [f"{_base}/{sub}/{n}" for n in sorted(os.listdir(sub_dir))]

            entry = dict(meta)
            entry["forecast_files"] = _listing("forecast")
            entry["specifications_files"] = _listing("specifications")
            bt = os.path.join(sys_dir, "besttrack.csv")
            entry["besttrack"] = f"{base}/besttrack.csv" if os.path.exists(bt) else ""
            systems.append(entry)
    index = {"generated_utc": _utcnow_iso(), "active": sorted(active_ids),
             "systems": systems}
    pretty_dump(index, os.path.join(outdir, "index.json"))
    return index


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def print_active(active: list[dict]) -> None:
    if not active:
        print("No active tropical cyclones are listed by JMA right now.")
        return
    print("Active tropical cyclones:")
    for s in active:
        print(f"  {s.get('tropicalCyclone', '?'):10} "
              f"No.{s.get('typhoonNumber', '?'):6} "
              f"{s.get('category', '?'):4} "
              f"issued {s.get('issue', '?')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tc", help="tropicalCyclone id (e.g. TC2615) or "
                                     "typhoon number (e.g. 2613). Omit to be prompted.")
    parser.add_argument("--all", action="store_true",
                        help="fetch every currently active system")
    parser.add_argument("--list", action="store_true",
                        help="list active systems and exit")
    parser.add_argument("--reindex", action="store_true",
                        help="rebuild index.json from existing data and exit")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help=f"output folder (default: {DEFAULT_OUTDIR})")
    args = parser.parse_args(argv)

    try:
        active = get_active_systems()
    except urllib.error.URLError as exc:
        print(f"Could not reach JMA: {exc}", file=sys.stderr)
        return 2

    if args.list:
        print_active(active)
        return 0

    active_ids = {s.get("tropicalCyclone", "") for s in active}

    if args.reindex:
        os.makedirs(args.outdir, exist_ok=True)
        build_index(args.outdir, active_ids)
        print("reindexed")
        return 0

    if not active:
        print("No active tropical cyclones are listed by JMA right now.")
        return 0

    if args.all:
        selection = "all"
    elif args.tc:
        selection = args.tc
    else:
        print_active(active)
        selection = input(
            "\nEnter a tropicalCyclone id / typhoon number, or 'all': ").strip()

    targets = resolve_targets(active, selection)
    if not targets:
        print(f"No active system matched '{selection}'.", file=sys.stderr)
        print_active(active)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    registry = load_registry(args.outdir)
    for system in targets:
        try:
            process_system(system, args.outdir, registry)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"  ! failed for {system.get('tropicalCyclone')}: {exc}",
                  file=sys.stderr)
    save_registry(args.outdir, registry)
    build_index(args.outdir, active_ids)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
