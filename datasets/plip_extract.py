import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PLIP_CACHE_SCHEMA_VERSION = 2

# Water bridges are intentionally omitted.
DEFAULT_INTERACTION_TYPES: Sequence[str] = (
    "hydrophobic",
    "hbond",
    "saltbridge",
    "pistacking",
    "pication",
    "halogen",
    "metal",
)


@dataclass
class PlipInteraction:
    lig_atom_idx: int
    rec_residue_idx: int
    interaction_type: str
    distance: float
    angle: Optional[float] = None
    direction: Optional[Tuple[float, float, float]] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        payload = asdict(self)
        if self.direction is not None:
            payload["direction"] = list(self.direction)
        return payload


def _coerce_int(value: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _split_atom_list(token: str) -> List[int]:
    if token is None:
        return []
    return [int(x) for x in str(token).split(",") if str(x).strip().isdigit()]


def _parse_rst_table(lines: List[str]) -> List[Dict[str, str]]:
    """Parse a simple reStructuredText table emitted by PLIP txt reporter."""
    rows: List[List[str]] = []
    for line in lines:
        line = line.rstrip("\n")
        if not line.startswith("|"):
            continue
        # Drop first and last empty split from leading/trailing |
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return []
    headers = rows[0]
    parsed: List[Dict[str, str]] = []
    for row in rows[1:]:
        # Pad/truncate to header length
        adjusted = row + [""] * (len(headers) - len(row))
        parsed.append(dict(zip(headers, adjusted)))
    return parsed


def _extract_section(lines: List[str], section_name: str) -> List[str]:
    """Return table lines belonging to a section named by the bold heading."""
    start_token = f"**{section_name}**"
    collecting = False
    buffer: List[str] = []
    for line in lines:
        if line.strip().startswith("**") and start_token not in line:
            if collecting:
                break
        if start_token in line:
            collecting = True
            continue
        if collecting:
            buffer.append(line)
    return buffer


def _build_interactions(parsed_rows: List[Dict[str, str]], interaction_type: str) -> List[PlipInteraction]:
    interactions: List[PlipInteraction] = []
    for row in parsed_rows:
        if interaction_type == "hydrophobic":
            lig_idx = _coerce_int(row.get("LIGCARBONIDX"), -1)
            rec_res = _coerce_int(row.get("RESNR"), -1)
            distance = _coerce_float(row.get("DIST"), 0.0)
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=lig_idx,
                    rec_residue_idx=rec_res,
                    interaction_type=interaction_type,
                    distance=distance,
                    angle=0.0,
                    direction=(0.0, 0.0, 0.0),
                )
            )
        elif interaction_type == "hbond":
            prot_is_don = row.get("PROTISDON", "False").lower() in ("true", "1")
            donor_idx = _coerce_int(row.get("DONORIDX"), -1)
            acceptor_idx = _coerce_int(row.get("ACCEPTORIDX"), -1)
            lig_idx = acceptor_idx if prot_is_don else donor_idx
            rec_res = _coerce_int(row.get("RESNR"), -1)
            distance = _coerce_float(row.get("DIST_H-A"), 0.0)
            angle = _coerce_float(row.get("DON_ANGLE"), 0.0)
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=lig_idx,
                    rec_residue_idx=rec_res,
                    interaction_type=interaction_type,
                    distance=distance,
                    angle=angle,
                    direction=(0.0, 0.0, 0.0),
                )
            )
        elif interaction_type == "saltbridge":
            lig_list = _split_atom_list(row.get("LIG_IDX_LIST"))
            lig_idx = lig_list[0] if lig_list else -1
            rec_res = _coerce_int(row.get("RESNR"), -1)
            distance = _coerce_float(row.get("DIST"), 0.0)
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=lig_idx,
                    rec_residue_idx=rec_res,
                    interaction_type=interaction_type,
                    distance=distance,
                    angle=0.0,
                    direction=(0.0, 0.0, 0.0),
                )
            )
        elif interaction_type == "pistacking":
            lig_list = _split_atom_list(row.get("LIG_IDX_LIST"))
            lig_idx = lig_list[0] if lig_list else -1
            rec_res = _coerce_int(row.get("RESNR"), -1)
            distance = _coerce_float(row.get("CENTDIST"), 0.0)
            angle = _coerce_float(row.get("ANGLE"), 0.0)
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=lig_idx,
                    rec_residue_idx=rec_res,
                    interaction_type=interaction_type,
                    distance=distance,
                    angle=angle,
                    direction=(0.0, 0.0, 0.0),
                )
            )
        elif interaction_type == "pication":
            lig_list = _split_atom_list(row.get("LIG_IDX_LIST"))
            lig_idx = lig_list[0] if lig_list else -1
            rec_res = _coerce_int(row.get("RESNR"), -1)
            distance = _coerce_float(row.get("DIST"), 0.0)
            angle = 0.0
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=lig_idx,
                    rec_residue_idx=rec_res,
                    interaction_type=interaction_type,
                    distance=distance,
                    angle=angle,
                    direction=(0.0, 0.0, 0.0),
                )
            )
        elif interaction_type == "halogen":
            lig_idx = _coerce_int(row.get("DON_IDX") or row.get("ACC_IDX"), -1)
            rec_res = _coerce_int(row.get("RESNR"), -1)
            distance = _coerce_float(row.get("DIST"), 0.0)
            angle = _coerce_float(row.get("DON_ANGLE") or row.get("ACC_ANGLE"), 0.0)
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=lig_idx,
                    rec_residue_idx=rec_res,
                    interaction_type=interaction_type,
                    distance=distance,
                    angle=angle,
                    direction=(0.0, 0.0, 0.0),
                )
            )
        elif interaction_type == "metal":
            lig_idx = _coerce_int(row.get("TARGET_IDX"), -1)
            rec_res = _coerce_int(row.get("RESNR"), -1)
            distance = _coerce_float(row.get("DIST"), 0.0)
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=lig_idx,
                    rec_residue_idx=rec_res,
                    interaction_type=interaction_type,
                    distance=distance,
                    angle=0.0,
                    direction=(0.0, 0.0, 0.0),
                )
            )
    return interactions


def parse_plip_txt(txt_path: str, pdbid: str) -> List[PlipInteraction]:
    """Parse a PLIP-generated txt report into a list of PlipInteraction."""
    with open(txt_path, "r") as f:
        lines = f.readlines()

    interactions: List[PlipInteraction] = []
    section_mapping = {
        "Hydrophobic Interactions": "hydrophobic",
        "Hydrogen Bonds": "hbond",
        "Salt Bridges": "saltbridge",
        "pi-Stacking": "pistacking",
        "pi-Cation Interactions": "pication",
        "Halogen Bonds": "halogen",
        "Metal Complexes": "metal",
    }

    for section_label, interaction_type in section_mapping.items():
        table_lines = _extract_section(lines, section_label)
        parsed_rows = _parse_rst_table(table_lines)
        if not parsed_rows:
            continue
        interactions.extend(_build_interactions(parsed_rows, interaction_type))

    return interactions


def _resolve_txt_path(structure_path: str, pdbid: str) -> Optional[str]:
    """Infer the txt report path from a provided path (txt or pdb)."""
    if os.path.isfile(structure_path) and structure_path.lower().endswith(".txt"):
        return structure_path
    candidate_dir = structure_path
    if os.path.isfile(structure_path):
        candidate_dir = os.path.dirname(structure_path)
    for filename in (f"{pdbid}.txt", f"{pdbid}_report.txt", f"{pdbid}_plip.txt"):
        candidate = os.path.join(candidate_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def extract_plip_interactions(
    structure_path: str,
    pdbid: str,
    output_dir: str,
    overwrite: bool = False,
) -> str:
    """Parse a PLIP txt report and cache the interaction JSON.

    Parameters
    ----------
    structure_path : str
        Path to a PLIP txt report or a directory containing ``<pdbid>.txt``.
    pdbid : str
        Identifier used to name the cache file.
    output_dir : str
        Directory used to store the cached interaction JSON files.
    overwrite : bool
        If True, re-parse even when a cache file exists.

    Returns
    -------
    str
        Path to the written cache file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdbid}_interactions.json")
    if os.path.exists(output_path) and not overwrite:
        return output_path

    txt_path = _resolve_txt_path(structure_path, pdbid)
    if txt_path is None:
        raise FileNotFoundError(f"Could not find PLIP txt report for {pdbid} near {structure_path}")

    interactions = parse_plip_txt(txt_path, pdbid)
    payload = {
        "pdbid": pdbid,
        "num_interactions": len(interactions),
        "interactions": [inter.to_dict() for inter in interactions],
        "schema_version": PLIP_CACHE_SCHEMA_VERSION,
        "interaction_types": list(DEFAULT_INTERACTION_TYPES),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)
    return output_path


def load_plip_cache(pdbid: str, cache_dir: str) -> Optional[Dict]:
    """Load a cached PLIP result."""
    cache_path_json = os.path.join(cache_dir, f"{pdbid}_interactions.json")
    cache_path_pkl = os.path.join(cache_dir, f"{pdbid}_interactions.pkl")
    if os.path.exists(cache_path_json):
        with open(cache_path_json, "r") as f:
            payload = json.load(f)
    elif os.path.exists(cache_path_pkl):  # pragma: no cover - backwards compatibility
        import pickle

        with open(cache_path_pkl, "rb") as f:
            payload = pickle.load(f)
    else:
        return None

    if not isinstance(payload, dict):  # pragma: no cover - defensive guard
        return None

    payload.setdefault("schema_version", 0)
    payload.setdefault("interaction_types", list(DEFAULT_INTERACTION_TYPES))
    if payload["schema_version"] != PLIP_CACHE_SCHEMA_VERSION:
        warnings.warn(
            f"PLIP cache schema mismatch for {pdbid}: found {payload['schema_version']}, "
            f"expected {PLIP_CACHE_SCHEMA_VERSION}. Falling back to legacy compatibility.",
            stacklevel=2,
        )
        payload["_schema_mismatch"] = True
    return payload


def batch_extract(
    txt_dir: str,
    pdb_ids: Iterable[str],
    output_dir: str = "data/cache_plip",
    overwrite: bool = False,
) -> None:
    """Batch parse PLIP txt reports for a collection of complexes."""
    failures: List[str] = []
    for pdbid in pdb_ids:
        txt_path = os.path.join(txt_dir, f"{pdbid}.txt")
        if not os.path.exists(txt_path):
            warnings.warn(f"Skipping {pdbid}: txt report not found at {txt_path}")
            failures.append(pdbid)
            continue
        try:
            extract_plip_interactions(txt_path, pdbid, output_dir, overwrite=overwrite)
        except Exception as exc:  # pragma: no cover - logging only
            warnings.warn(f"PLIP txt parsing failed on {pdbid}: {exc}")
            failures.append(pdbid)
    if failures:
        sys.stderr.write(f"PLIP txt parsing failed for {len(failures)} complexes: {', '.join(failures)}\n")


__all__ = [
    "PLIP_CACHE_SCHEMA_VERSION",
    "DEFAULT_INTERACTION_TYPES",
    "PlipInteraction",
    "batch_extract",
    "extract_plip_interactions",
    "load_plip_cache",
    "parse_plip_txt",
]
