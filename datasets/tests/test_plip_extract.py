import json
import pickle

import pytest

pytest.importorskip("numpy")

from datasets.plip_extract import (
    DEFAULT_INTERACTION_TYPES,
    PLIP_CACHE_SCHEMA_VERSION,
    load_plip_cache,
)


def test_load_plip_cache_new_schema(tmp_path):
    pdbid = "1abc"
    payload = {
        "pdbid": pdbid,
        "num_interactions": 1,
        "interactions": [
            {
                "lig_atom_idx": 0,
                "rec_residue_idx": 1,
                "interaction_type": "hydrophobic",
                "distance": 1.1,
                "angle": None,
                "direction": None,
                "confidence": 0.75,
            }
        ],
        "schema_version": PLIP_CACHE_SCHEMA_VERSION,
        "interaction_types": list(DEFAULT_INTERACTION_TYPES),
    }
    cache_path = tmp_path / f"{pdbid}_interactions.json"
    cache_path.write_text(json.dumps(payload))

    result = load_plip_cache(pdbid, tmp_path)

    assert result["schema_version"] == PLIP_CACHE_SCHEMA_VERSION
    assert result["interaction_types"] == list(DEFAULT_INTERACTION_TYPES)
    assert result["interactions"] == payload["interactions"]
    assert result["pdbid"] == pdbid


def test_load_plip_cache_legacy_pickle(tmp_path):
    pdbid = "2xyz"
    payload = {
        "interactions": [
            {
                "lig_atom_idx": 2,
                "rec_residue_idx": 5,
                "interaction_type": "hbond",
                "distance": 2.0,
            }
        ],
        "num_interactions": 1,
    }
    cache_path = tmp_path / f"{pdbid}_interactions.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)

    result = load_plip_cache(pdbid, tmp_path)

    assert result["schema_version"] == 0
    assert result["interaction_types"] == list(DEFAULT_INTERACTION_TYPES)
    assert result["interactions"] == payload["interactions"]
