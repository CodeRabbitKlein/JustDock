import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_INTERACTION_TYPES: Sequence[str] = (
    "hydrophobic",
    "hbond",
    "waterbridge",
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


def _try_import_plip():
    try:
        from plip.structure.preparation import PDBComplex  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PLIP is required to extract interactions. Install with `pip install plip`."
        ) from exc
    return PDBComplex


def _extract_interactions_from_complex(complex_obj, pdbid: str) -> List[PlipInteraction]:
    """Extract interactions from an analyzed PLIP complex."""
    interactions: List[PlipInteraction] = []
    for ligand in complex_obj.ligands:
        # Skip ligands that are not part of the requested pdbid
        if ligand.hetid != pdbid and ligand.resnr != pdbid:
            # PLIP stores the pdbid on each ligand for multi-ligand structures. If none
            # match, we still continue because PDBBind entries are unique per structure.
            pass
        interaction_set = complex_obj.interaction_sets[ligand.uid]
        # hydrophobic
        for hphob in interaction_set.hydrophobic_contacts:
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=hphob.ligatom.idx,
                    rec_residue_idx=hphob.resnr,
                    interaction_type="hydrophobic",
                    distance=float(hphob.dist),
                )
            )
        # hydrogen bonds
        for hbond in interaction_set.hbonds_pdonor + interaction_set.hbonds_ldonor:
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=hbond.ligatom.idx,
                    rec_residue_idx=hbond.resnr,
                    interaction_type="hbond",
                    distance=float(hbond.dist_h_a),
                    angle=float(hbond.angle if hasattr(hbond, "angle") else 0.0),
                    direction=tuple(np.array(hbond.normvec) if hasattr(hbond, "normvec") else np.zeros(3)),
                )
            )
        # salt bridges
        for salt in interaction_set.saltbridge_lneg + interaction_set.saltbridge_lpos:
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=salt.ligatom.idx,
                    rec_residue_idx=salt.resnr,
                    interaction_type="saltbridge",
                    distance=float(salt.dist),
                )
            )
        # pi interactions
        for stack in interaction_set.pistacking:
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=stack.ligatom.idx,
                    rec_residue_idx=stack.resnr,
                    interaction_type="pistacking",
                    distance=float(stack.distance),
                    angle=float(stack.angle if hasattr(stack, "angle") else 0.0),
                    direction=tuple(np.array(stack.nvec) if hasattr(stack, "nvec") else np.zeros(3)),
                )
            )
        for cat in interaction_set.pication_laro + interaction_set.pication_paro:
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=cat.ligatom.idx,
                    rec_residue_idx=cat.resnr,
                    interaction_type="pication",
                    distance=float(cat.distance),
                    angle=float(cat.angle if hasattr(cat, "angle") else 0.0),
                    direction=tuple(np.array(cat.vec) if hasattr(cat, "vec") else np.zeros(3)),
                )
            )
        # halogens
        for hal in interaction_set.halogen_bonds:
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=hal.ligatom.idx,
                    rec_residue_idx=hal.resnr,
                    interaction_type="halogen",
                    distance=float(hal.dist),
                    angle=float(hal.angle if hasattr(hal, "angle") else 0.0),
                    direction=tuple(np.array(hal.vec) if hasattr(hal, "vec") else np.zeros(3)),
                )
            )
        # metal
        for metal in interaction_set.metals:
            interactions.append(
                PlipInteraction(
                    lig_atom_idx=metal.ligatom.idx,
                    rec_residue_idx=metal.resnr,
                    interaction_type="metal",
                    distance=float(metal.dist),
                )
            )
    return interactions


def extract_plip_interactions(
    pdb_path: str,
    pdbid: str,
    output_dir: str,
    overwrite: bool = False,
) -> str:
    """Run PLIP on a complex file and cache the interaction JSON.

    Parameters
    ----------
    pdb_path : str
        Path to a prepared complex structure containing both receptor and ligand.
    pdbid : str
        Identifier used to name the cache file.
    output_dir : str
        Directory used to store the cached interaction JSON files.
    overwrite : bool
        If True, rerun PLIP even when a cache file exists.

    Returns
    -------
    str
        Path to the written cache file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdbid}_interactions.json")
    if os.path.exists(output_path) and not overwrite:
        return output_path

    PDBComplex = _try_import_plip()
    complex_obj = PDBComplex()
    complex_obj.load_pdb(pdb_path)
    complex_obj.analyze()
    interactions = _extract_interactions_from_complex(complex_obj, pdbid)

    with open(output_path, "w") as f:
        json.dump(
            {
                "pdbid": pdbid,
                "num_interactions": len(interactions),
                "interactions": [inter.to_dict() for inter in interactions],
            },
            f,
        )
    return output_path


def load_plip_cache(pdbid: str, cache_dir: str) -> Optional[Dict]:
    """Load a cached PLIP result."""
    cache_path_json = os.path.join(cache_dir, f"{pdbid}_interactions.json")
    cache_path_pkl = os.path.join(cache_dir, f"{pdbid}_interactions.pkl")
    if os.path.exists(cache_path_json):
        with open(cache_path_json, "r") as f:
            return json.load(f)
    if os.path.exists(cache_path_pkl):  # pragma: no cover - backwards compatibility
        import pickle

        with open(cache_path_pkl, "rb") as f:
            return pickle.load(f)
    return None


def batch_extract(
    pdb_dir: str,
    pdb_ids: Iterable[str],
    output_dir: str = "data/cache_plip",
    overwrite: bool = False,
) -> None:
    """Batch run PLIP over a collection of complexes."""
    failures: List[str] = []
    for pdbid in pdb_ids:
        pdb_path = os.path.join(pdb_dir, pdbid, f"{pdbid}.pdb")
        if not os.path.exists(pdb_path):
            warnings.warn(f"Skipping {pdbid}: structure file not found at {pdb_path}")
            failures.append(pdbid)
            continue
        try:
            extract_plip_interactions(pdb_path, pdbid, output_dir, overwrite=overwrite)
        except Exception as exc:  # pragma: no cover - logging only
            warnings.warn(f"PLIP failed on {pdbid}: {exc}")
            failures.append(pdbid)
    if failures:
        sys.stderr.write(f"PLIP extraction failed for {len(failures)} complexes: {', '.join(failures)}\n")


__all__ = [
    "DEFAULT_INTERACTION_TYPES",
    "PlipInteraction",
    "batch_extract",
    "extract_plip_interactions",
    "load_plip_cache",
]
