import binascii
import glob
import hashlib
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
import math
import pandas as pd
import numpy as np
# from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser,MMCIFParser
import torch
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch.utils.data import WeightedRandomSampler

from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from datasets.process_mols import read_molecule, get_rec_graph, generate_conformer, \
    get_lig_graph_with_matching, extract_receptor_structure, parse_receptor, parse_pdb_from_path, \
    lig_feature_dims, rec_residue_feature_dims
from datasets.plip_extract import (
    DEFAULT_INTERACTION_TYPES,
    PLIP_CACHE_SCHEMA_VERSION,
    extract_plip_interactions,
    load_plip_cache,
)
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt
from utils import so3, torus

CACHE_SCHEMA_VERSION = 1




class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom

    def __call__(self, data):
        t = np.random.uniform()
        t_tr, t_rot, t_tor, t_res_tr, t_res_tor, t_res_chi = t, t, t, t, t, t
        return self.apply_noise(data, t_tr, t_rot, t_tor, t_res_tr, t_res_tor, t_res_chi)

    def apply_noise(self, data, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, tr_update = None, rot_update=None, torsion_updates=None, res_tr_update = None, res_rot_update=None, res_chi_update=None):
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        tr_sigma, rot_sigma, tor_sigma, res_tr_sigma, res_rot_sigma, res_chi_sigma = self.t_to_sigma(t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi)
        set_time(data, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, 1, self.all_atom, device=None)

        pocket_center = data['ligand'].pos.mean(dim=0,keepdim=True)
        res_distance = (data['receptor'].pos-pocket_center).norm(dim=-1,keepdim=True)
        data.res_decay_weight = torch.exp(-torch.nn.ReLU()((res_distance-6.)/10.))
        orig_ca_lig_cross_distances = (data['ligand'].pos[None,...] - data['receptor'].pos[:,None,...]).norm(dim=-1)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)).float() if tr_update is None else tr_update
        tr_update_norm = tr_update.norm(dim=-1).item()
        # x = np.random.randn(3)
        # x /= np.linalg.norm(x)
        # rot_update = torch.tensor(x).float()[None,...] * torch.clamp(torch.normal(mean=0., std=rot_sigma, size=(1,)).float(),min=-np.pi, max=np.pi)[0] if rot_update is None else rot_update
        if rot_update is None:
            # rot_update = torch.normal(mean=0, std=rot_sigma, size=(1, 3))
            rot_update = torch.tensor(so3.sample_vec(eps=rot_sigma)).float()[None,...]

        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
        torsion_updates = None if self.no_torsion else torsion_updates

        res_sigma = torch.clamp(res_tr_sigma + torch.normal(mean=0., std=0.2, size=(1,)).float(),min=0., max=1.)[0]
        res_tr_update = data['receptor'].af2_trans * res_sigma #+ torch.normal(mean=0, std=0.5, size=(data['receptor'].pos.shape[0], 3))#torch.normal(mean=0, std=res_tr_sigma, size=(data['receptor'].pos.shape[0], 1)).abs() if res_tr_update is None else res_tr_update
        res_rot_update = data['receptor'].af2_rotvecs * res_sigma #torch.normal(mean=0, std=res_rot_sigma, size=(data['receptor'].pos.shape[0], 1)).abs() if res_rot_update is None else res_rot_update
        res_chi_update = (data['receptor'].af2_chis[:,[0,2,4,5,6]] * res_sigma + torch.normal(mean=0, std=0.3, size=(data['receptor'].pos.shape[0], 5))) * data['receptor'].chi_masks[:,[0,2,4,5,6]] if res_chi_update is None else res_chi_update
        modify_conformer(data, tr_update, rot_update, torsion_updates, res_tr_update, res_rot_update, res_chi_update)
        data.tr_score = -tr_update
        data.rot_score = -rot_update#torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(-torsion_updates).float()
        # data.rot_loss_weight = torch.ones(1,1).float() * (tr_update_norm<10.)
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma
        # data.tor_loss_weight = None if self.no_torsion else torch.from_numpy(np.ones(data['ligand'].edge_mask.sum()) * (tr_update_norm<10.)).float()
        data.res_tr_score = -res_tr_update
        data.res_rot_score = -res_rot_update#torch.from_numpy(so3.score_res_vec(vec=res_rot_update, eps=res_rot_sigma.cpu().numpy())).float()
        data.res_chi_score = -res_chi_update
        data.res_loss_weight = torch.ones(data['receptor'].pos.shape[0],1).float() * (tr_update_norm<6.)
        ca_lig_cross_distances = (data['ligand'].pos[None,...] - data['receptor'].pos[:,None,...]).norm(dim=-1)
        ca_lig_cross_distances_diff = (orig_ca_lig_cross_distances - ca_lig_cross_distances).abs()

        cutoff_mask = (orig_ca_lig_cross_distances < 15.0).float()
        score = 0.25 * ((ca_lig_cross_distances_diff<0.5).float()
                        + (ca_lig_cross_distances_diff<1.0).float()
                        + (ca_lig_cross_distances_diff<2.0).float()
                        + (ca_lig_cross_distances_diff<4.0).float())
        ca_lig_cross_lddt =  (score * cutoff_mask).sum() / cutoff_mask.sum()
        data.lddt = ca_lig_cross_lddt[None,None]
        # print(data.name,data.lddt)
        data.affinity = data.affinity[:,None] #* data.lddt
        return data


class PDBBind(Dataset):
    def __init__(self, root, transform=None, info=None, cache_path='data/cache', split_path='data/', limit_complexes=0,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, center_ligand=False, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False, require_receptor=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, name_list=None, keep_local_structures=False, use_existing_cache=True,
                 use_plip=False, plip_cache_dir='data/cache_plip', plip_confidence_threshold=0.5, plip_interaction_types=None,
                 plip_auto_rebuild=False, plip_skip_invalid=True, plip_max_interactions_per_residue=None,
                 plip_max_interactions_per_ligand_atom=None, plip_distance_max=None, plip_min_confidence=None):

        super(PDBBind, self).__init__(root, transform)
        self.pdbbind_dir = root
        self.info = info
        self.parallel_count = 1000
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.require_ligand = require_ligand
        self.require_receptor = require_receptor
        self.protein_path_list = protein_path_list
        self.name_list = name_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.use_plip = use_plip
        self.plip_cache_dir = plip_cache_dir
        self.plip_confidence_threshold = plip_confidence_threshold
        self.plip_interaction_types = plip_interaction_types if plip_interaction_types is not None else list(DEFAULT_INTERACTION_TYPES)
        self.cache_failures = defaultdict(list)
        self._cache_summary_logged = False
        self.plip_index_anomalies = []
        self._rebuilt_cache = False
        self.plip_auto_rebuild = plip_auto_rebuild
        self.plip_skip_invalid = plip_skip_invalid
        self.plip_max_interactions_per_residue = plip_max_interactions_per_residue
        self.plip_max_interactions_per_ligand_atom = plip_max_interactions_per_ligand_atom
        self.plip_distance_max = plip_distance_max
        self.plip_min_confidence = plip_min_confidence
        self._plip_disabled = False
        self._plip_disable_reason = None
        self._plip_rebuild_attempted = set()
        self.plip_bad_samples = []
        if matching or protein_path_list is not None and ligand_descriptions is not None:
            cache_path += '_torsion'
        if all_atoms:
            cache_path += '_allatoms'
        self.full_cache_path = os.path.join(cache_path, f'limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                            + ('' if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + ('' if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode()))))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.center_ligand = center_ligand
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        hetero_path = os.path.join(self.full_cache_path, "heterographs.pkl")
        rdkit_path = os.path.join(self.full_cache_path, "rdkit_ligands.pkl")
        receptor_path = os.path.join(self.full_cache_path, "receptor_pdbs.pkl")
        if (not use_existing_cache) or (not os.path.exists(hetero_path)\
                or (require_ligand and not os.path.exists(rdkit_path))):
            os.makedirs(self.full_cache_path, exist_ok=True)
            if protein_path_list is None or ligand_descriptions is None:
                self.preprocessing()
            else:
                self.inference_preprocessing()

        print('loading data from memory: ', hetero_path)
        try:
            self.complex_graphs, hetero_meta = self._load_cache(hetero_path, 'heterographs')
        except Exception as e:
            self._maybe_rebuild_cache(f'heterograph load failure: {e}')
            self.complex_graphs, hetero_meta = self._load_cache(hetero_path, 'heterographs')
        for g in self.complex_graphs:
            self._ensure_anchor_fields(g)
            if self.use_plip and hasattr(g, 'name'):
                self._apply_plip_interactions(g, g.name)
        if require_ligand:
            try:
                self.rdkit_ligands, _ = self._load_cache(rdkit_path, 'rdkit_ligands')
            except Exception as e:
                self._maybe_rebuild_cache(f'ligand cache load failure: {e}')
                self.rdkit_ligands, _ = self._load_cache(rdkit_path, 'rdkit_ligands')
        if require_receptor:
            try:
                self.receptor_pdbs, _ = self._load_cache(receptor_path, 'receptor_pdbs')
            except Exception as e:
                self._maybe_rebuild_cache(f'receptor cache load failure: {e}')
                self.receptor_pdbs, _ = self._load_cache(receptor_path, 'receptor_pdbs')
        print_statistics(self.complex_graphs)

    def len(self):
        return len(self.complex_graphs)

    # Cache helpers
    def _cache_metadata(self):
        return {
            'schema_version': CACHE_SCHEMA_VERSION,
            'use_plip': self.use_plip,
            'plip_interaction_types': list(self.plip_interaction_types),
        }

    def _save_cache(self, path, payload):
        wrapped = {'metadata': self._cache_metadata(), 'data': payload}
        with open(path, 'wb') as f:
            pickle.dump(wrapped, f)

    def _load_cache(self, path, description):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        except FileNotFoundError as e:
            self.cache_failures['missing'].append({'description': description, 'path': path})
            raise
        except Exception as e:
            self.cache_failures['io'].append({'description': description, 'path': path, 'error': str(e)})
            raise
        metadata = {}
        data = obj
        if isinstance(obj, dict) and 'data' in obj:
            data = obj.get('data')
            metadata = obj.get('metadata', {})
        self._validate_cache_metadata(metadata, description)
        return data, metadata

    def _validate_cache_metadata(self, metadata, description):
        if not metadata:
            # legacy cache
            self.cache_failures['legacy'].append(description)
            return
        schema_version = metadata.get('schema_version')
        if schema_version != CACHE_SCHEMA_VERSION:
            self.cache_failures['schema_mismatch'].append({'description': description, 'found': schema_version})
            raise ValueError(f'{description} cache schema mismatch: expected {CACHE_SCHEMA_VERSION}, got {schema_version}')
        if self.use_plip:
            cached_types = metadata.get('plip_interaction_types')
            if cached_types is None or set(cached_types) != set(self.plip_interaction_types):
                self.cache_failures['plip_type_mismatch'].append({'description': description, 'cached': cached_types})
                raise ValueError(f'{description} cache PLIP types mismatch: expected {self.plip_interaction_types}, got {cached_types}')

    def _maybe_rebuild_cache(self, reason):
        if self._rebuilt_cache:
            return
        print(f'Attempting to rebuild cache due to: {reason}')
        os.makedirs(self.full_cache_path, exist_ok=True)
        if self.protein_path_list is None or self.ligand_descriptions is None:
            self.preprocessing()
        else:
            self.inference_preprocessing()
        self._rebuilt_cache = True
        self._log_cache_anomalies()

    def _log_cache_anomalies(self):
        if self._cache_summary_logged:
            return
        missing_entries = self.cache_failures.get('missing', []) + self.cache_failures.get('legacy', [])
        damaged_entries = self.cache_failures.get('io', []) + self.cache_failures.get('schema_mismatch', [])
        plip_mismatch_entries = self.cache_failures.get('plip_type_mismatch', [])

        def _format(entries):
            formatted = []
            for entry in entries:
                if isinstance(entry, dict):
                    label = entry.get('description') or entry.get('name') or 'unknown'
                    details = []
                    for key in ('path', 'error', 'found', 'cached'):
                        if key in entry and entry[key] is not None:
                            details.append(f'{key}={entry[key]}')
                    formatted.append(f'{label} ({", ".join(details)})' if details else label)
                else:
                    formatted.append(str(entry))
            return ', '.join(formatted)

        if not (missing_entries or damaged_entries or plip_mismatch_entries):
            return
        print('Cache anomaly summary (triggered rebuild):')
        if missing_entries:
            print(f'  Missing caches: {_format(missing_entries)}')
        if damaged_entries:
            print(f'  Damaged caches: {_format(damaged_entries)}')
        if plip_mismatch_entries:
            print(f'  PLIP type mismatches: {_format(plip_mismatch_entries)}')
        if self.plip_bad_samples:
            bad_summary = ', '.join([f"{e['name']} ({e['reason']})" for e in self.plip_bad_samples])
            print(f'  PLIP validation failures: {bad_summary}')
        self._cache_summary_logged = True

    def _disable_plip(self, reason):
        if self._plip_disabled:
            return
        self._plip_disabled = True
        self._plip_disable_reason = reason
        self.use_plip = False
        print(f'Disabling PLIP features: {reason}. Falling back to non-PLIP mode.')

    def _maybe_rebuild_plip_cache(self, name, reason):
        if not self.plip_auto_rebuild or name in self._plip_rebuild_attempted:
            return False
        self._plip_rebuild_attempted.add(name)
        pdb_path = os.path.join(self.pdbbind_dir, name, f"{name}.pdb")
        if not os.path.exists(pdb_path):
            self.plip_bad_samples.append({'name': name, 'reason': f'{reason}; missing pdb at {pdb_path}'})
            return False
        try:
            print(f'Rebuilding PLIP cache for {name} due to {reason}...')
            extract_plip_interactions(pdb_path, name, self.plip_cache_dir, overwrite=True)
            return True
        except Exception as exc:
            self.plip_bad_samples.append({'name': name, 'reason': f'{reason}; rebuild failed: {exc}'})
            return False

    def _validate_plip_entries(self, cache, name, lig_count, rec_count):
        normalized = []
        invalid_entries = []
        for entry in cache.get('interactions', []):
            try:
                lig_idx = int(entry.get('lig_atom_idx'))
                rec_idx = int(entry.get('rec_residue_idx'))
            except Exception:
                invalid_entries.append(entry)
                continue
            if lig_idx >= lig_count or rec_idx >= rec_count or lig_idx < 0 or rec_idx < 0:
                invalid_entries.append({'lig': lig_idx, 'rec': rec_idx})
                continue
            normalized.append({
                'lig_atom_idx': lig_idx,
                'rec_residue_idx': rec_idx,
                'interaction_type': entry.get('interaction_type', 'misc'),
                'distance': float(entry.get('distance', 0.0)),
                'angle': float(entry.get('angle', 0.0)) if entry.get('angle') is not None else 0.0,
                'direction': entry.get('direction', [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0],
                'confidence': float(entry.get('confidence', 1.0)),
            })
        if invalid_entries:
            self.plip_index_anomalies.append({'name': name, 'invalid': len(invalid_entries), 'total': len(cache.get('interactions', [])), 'lig_nodes': lig_count, 'rec_nodes': rec_count})
        return normalized, invalid_entries

    def _filter_plip_interactions(self, interactions):
        stats = defaultdict(int)
        filtered = []
        clipped_flags = []
        res_counts = defaultdict(int)
        lig_counts = defaultdict(int)
        max_per_res = self.plip_max_interactions_per_residue
        max_per_lig = self.plip_max_interactions_per_ligand_atom
        distance_cap = self.plip_distance_max
        min_conf = self.plip_min_confidence

        for entry in interactions:
            stats['total_before'] += 1
            if min_conf is not None and entry['confidence'] < min_conf:
                stats['dropped_low_conf'] += 1
                continue
            if distance_cap is not None and entry['distance'] > distance_cap:
                stats['dropped_distance'] += 1
                continue
            if max_per_res is not None and res_counts[entry['rec_residue_idx']] >= max_per_res:
                stats['dropped_max_residue'] += 1
                continue
            if max_per_lig is not None and lig_counts[entry['lig_atom_idx']] >= max_per_lig:
                stats['dropped_max_ligand'] += 1
                continue

            clipped = False
            if distance_cap is not None:
                if entry['distance'] > distance_cap:
                    clipped = True
                entry['distance'] = min(entry['distance'], distance_cap)
            entry['angle'] = max(0.0, min(float(entry['angle']), math.pi))
            direction = np.asarray(entry.get('direction', [0.0, 0.0, 0.0]), dtype=float)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / max(norm, 1.0)
            direction = np.clip(direction, -1.0, 1.0)
            if norm > 1.0 or np.any((direction < -1.0) | (direction > 1.0)):
                clipped = True
            entry['direction'] = direction.tolist()

            filtered.append(entry)
            clipped_flags.append(clipped)
            res_counts[entry['rec_residue_idx']] += 1
            lig_counts[entry['lig_atom_idx']] += 1

        stats['kept'] = len(filtered)
        stats['clipped'] = int(sum(clipped_flags))
        return filtered, stats, clipped_flags

    def _load_validated_plip_cache(self, name, lig_count, rec_count):
        if self._plip_disabled:
            return None
        cache = load_plip_cache(name, self.plip_cache_dir)
        if cache is None:
            if self._maybe_rebuild_plip_cache(name, 'missing cache'):
                cache = load_plip_cache(name, self.plip_cache_dir)
        if cache is None:
            print(f'PLIP cache missing for {name}, skipping PLIP features.')
            return None
        if cache.get('_schema_mismatch'):
            self._disable_plip(f'cache schema version {cache.get("schema_version")} != {PLIP_CACHE_SCHEMA_VERSION}')
            return None
        cached_types = cache.get('interaction_types')
        if cached_types is None or set(cached_types) != set(self.plip_interaction_types):
            self.cache_failures['plip_type_mismatch'].append({'name': name, 'cached': cached_types})
            if self._maybe_rebuild_plip_cache(name, f'type mismatch: {cached_types} vs {self.plip_interaction_types}'):
                cache = load_plip_cache(name, self.plip_cache_dir)
                cached_types = cache.get('interaction_types') if cache else None
            if cached_types is None or set(cached_types) != set(self.plip_interaction_types):
                self.plip_bad_samples.append({'name': name, 'reason': 'interaction_types mismatch'})
                return None

        normalized, invalid_entries = self._validate_plip_entries(cache, name, lig_count, rec_count)
        if invalid_entries:
            print(f'PLIP cache index mismatch for {name}: {len(invalid_entries)} invalid interactions skipped (lig_nodes={lig_count}, rec_nodes={rec_count}).')
            if self._maybe_rebuild_plip_cache(name, 'index mismatch'):
                return self._load_validated_plip_cache(name, lig_count, rec_count)
            if self.plip_skip_invalid:
                self.plip_bad_samples.append({'name': name, 'reason': 'index mismatch'})
                return None
        cache = dict(cache)
        cache['interactions'] = normalized
        return cache

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graphs[idx])
        # if self.protein_path_list is None or self.ligand_descriptions is None:
        #     af2_trans_sigma = torch.maximum(complex_graph['receptor'].af2_trans_sigma,torch.ones_like(complex_graph['receptor'].af2_trans_sigma))
        #     complex_graph['receptor'].af2_trans = complex_graph['receptor'].af2_trans / complex_graph['receptor'].af2_trans_sigma.unsqueeze(-1) * af2_trans_sigma.unsqueeze(-1)
        #     complex_graph['receptor'].af2_trans_sigma = af2_trans_sigma
        #     af2_rotvecs_sigma = torch.maximum(complex_graph['receptor'].af2_rotvecs_sigma,torch.ones_like(complex_graph['receptor'].af2_rotvecs_sigma)*0.3)
        #     complex_graph['receptor'].af2_rotvecs = complex_graph['receptor'].af2_rotvecs / complex_graph['receptor'].af2_rotvecs_sigma.unsqueeze(-1) * af2_rotvecs_sigma.unsqueeze(-1)
        #     complex_graph['receptor'].af2_rotvecs_sigma = af2_rotvecs_sigma
        if self.require_ligand:
            complex_graph.mol = copy.deepcopy(self.rdkit_ligands[idx])
        if self.require_receptor:
            complex_graph.rec_pdb = copy.deepcopy(self.receptor_pdbs[idx])
        # complex_graph['receptor'].acc_pred_chis = complex_graph['receptor'].acc_pred_chis[:,:5]
        return complex_graph

    def _apply_plip_interactions(self, complex_graph, name):
        if not self.use_plip or self._plip_disabled:
            return
        lig_count = complex_graph['ligand'].num_nodes
        rec_count = complex_graph['receptor'].num_nodes
        cache = self._load_validated_plip_cache(name, lig_count, rec_count)
        if cache is None or cache.get('interactions') is None:
            return
        filtered, clip_stats, clipped_flags = self._filter_plip_interactions(cache.get('interactions', []))
        if len(filtered) == 0:
            self.plip_bad_samples.append({'name': name, 'reason': 'no valid interactions after filtering'})
            return
        lig_idxs, rec_idxs, types, distances, angles, directions, confidences = [], [], [], [], [], [], []
        for entry in filtered:
            interaction_type = entry.get('interaction_type', 'misc')
            if interaction_type not in self.plip_interaction_types:
                continue
            lig_idxs.append(entry['lig_atom_idx'])
            rec_idxs.append(entry['rec_residue_idx'])
            types.append(self.plip_interaction_types.index(interaction_type))
            distances.append(float(entry.get('distance', 0.0)))
            angles.append(float(entry.get('angle', 0.0)))
            directions.append(entry.get('direction', [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
            confidences.append(float(entry.get('confidence', 1.0)))
        if len(lig_idxs) == 0:
            self.plip_bad_samples.append({'name': name, 'reason': 'no valid interactions'})
            return
        edge_index = torch.tensor([lig_idxs, rec_idxs], dtype=torch.long)
        complex_graph.plip_edge_index = edge_index
        complex_graph.plip_type = torch.tensor(types, dtype=torch.long)
        complex_graph.plip_distance = torch.tensor(distances, dtype=torch.float32)
        complex_graph.plip_angle = torch.tensor(angles, dtype=torch.float32)
        complex_graph.plip_direction = torch.tensor(directions, dtype=torch.float32)
        complex_graph.plip_confidence = torch.tensor(confidences, dtype=torch.float32)
        complex_graph.plip_clipped_mask = torch.tensor(clipped_flags, dtype=torch.bool)
        complex_graph.plip_interactions = {
            'edge_index': edge_index,
            'type': complex_graph.plip_type,
            'distance': complex_graph.plip_distance,
            'angle': complex_graph.plip_angle,
            'direction': complex_graph.plip_direction,
            'confidence': complex_graph.plip_confidence,
            'clipped_mask': complex_graph.plip_clipped_mask,
        }
        complex_graph.plip_clip_info = dict(clip_stats)
        # anchor masks
        lig_anchor = complex_graph['ligand'].anchor_mask.clone()
        rec_anchor = complex_graph['receptor'].anchor_mask.clone()
        confidence_tensor = complex_graph.plip_interactions['confidence']
        high_conf = confidence_tensor >= self.plip_confidence_threshold
        for lig_idx, rec_idx, mask in zip(lig_idxs, rec_idxs, high_conf):
            if mask:
                lig_anchor[lig_idx] = 1.0
                rec_anchor[rec_idx] = 1.0
        complex_graph['ligand'].anchor_mask = lig_anchor
        complex_graph['ligand'].x[:, -1] = lig_anchor
        complex_graph['receptor'].anchor_mask = rec_anchor
        complex_graph['receptor'].x[:, 1] = rec_anchor
        complex_graph.plip_pair_to_index = {(int(l), int(r)): idx for idx, (l, r) in enumerate(zip(lig_idxs, rec_idxs))}

    def _ensure_anchor_fields(self, complex_graph):
        # ligand anchors
        if 'anchor_mask' not in complex_graph['ligand']:
            lig_anchor = torch.zeros(complex_graph['ligand'].num_nodes, dtype=torch.float32)
            complex_graph['ligand'].anchor_mask = lig_anchor
            if complex_graph['ligand'].x.shape[1] == len(lig_feature_dims[0]):
                complex_graph['ligand'].x = torch.cat([complex_graph['ligand'].x.float(), lig_anchor.unsqueeze(-1)], dim=1)
        elif complex_graph['ligand'].x.shape[1] == len(lig_feature_dims[0]):
            complex_graph['ligand'].x = torch.cat([complex_graph['ligand'].x.float(), complex_graph['ligand'].anchor_mask.unsqueeze(-1)], dim=1)
        # receptor anchors
        if 'anchor_mask' not in complex_graph['receptor']:
            rec_anchor = torch.zeros(complex_graph['receptor'].num_nodes, dtype=torch.float32)
            complex_graph['receptor'].anchor_mask = rec_anchor
        else:
            rec_anchor = complex_graph['receptor'].anchor_mask
        if complex_graph['receptor'].x.shape[1] in (1, 1281):
            complex_graph['receptor'].x = torch.cat([complex_graph['receptor'].x.float(), rec_anchor.unsqueeze(-1)], dim=1)

    def process_one_batch(self,param):
        complex_names,lm_embeddings_chains,i = param
        if isinstance(lm_embeddings_chains, str):
            with open(lm_embeddings_chains,'rb') as f:
                lm_embeddings_chains = pickle.load(f)
        print(len(lm_embeddings_chains))
        complex_graphs, rdkit_ligands = [], []
        # with tqdm(total=len(complex_names), desc=f'loading complexes {i}') as pbar:
        for idx in tqdm(range(len(complex_names))):
            t = self.get_complex((complex_names[idx],None,lm_embeddings_chains[idx],None,None,None))
            complex_graphs.extend(t[0])
            rdkit_ligands.extend(t[1])
        # for t in map(self.get_complex, zip(complex_names, [None] * len(complex_names), lm_embeddings_chains, [None] * len(complex_names), [None] * len(complex_names), [None] * len(complex_names))):
        #     complex_graphs.extend(t[0])
        #     rdkit_ligands.extend(t[1])
            # pbar.update()

        self._save_cache(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), complex_graphs)
        self._save_cache(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), rdkit_ligands)

    def preprocessing(self):
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} complexes.')

        if self.esm_embeddings_path is not None:
            id_to_embeddings = torch.load(self.esm_embeddings_path)
            chain_embeddings_dictlist = defaultdict(list)
            for key, embedding in id_to_embeddings.items():
                key_name = key.split('_chain_')[0]
                if key_name in complex_names_all:
                    chain_embeddings_dictlist[key_name].append(embedding)
            lm_embeddings_chains_all = []
            for name in complex_names_all:
                lm_embeddings_chains_all.append(chain_embeddings_dictlist[name])
        else:
            lm_embeddings_chains_all = [None] * len(complex_names_all)
        # lm_embeddings_chains_all = None
        if self.num_workers > 1:
            # running preprocessing in parallel on multiple workers and saving the progress every self.parallel_count complexes
            lm_embeddings_chains = None
            for i in range(len(complex_names_all)//self.parallel_count+1):
                if os.path.exists(os.path.join(self.full_cache_path, f'lm_embeddings_chains_{i}.pkl')):
                    continue
                lm_embeddings_chains = lm_embeddings_chains_all[self.parallel_count*i:self.parallel_count*(i+1)]
                with open(os.path.join(self.full_cache_path, f'lm_embeddings_chains_{i}.pkl'), 'wb') as f:
                    pickle.dump(lm_embeddings_chains, f)
            del lm_embeddings_chains_all, lm_embeddings_chains
            params = []
            for i in range(len(complex_names_all)//self.parallel_count+1):
                if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                    continue
                complex_names = complex_names_all[self.parallel_count*i:self.parallel_count*(i+1)]
                params.append((complex_names,os.path.join(self.full_cache_path, f'lm_embeddings_chains_{i}.pkl'),i))
            print('params',len(params))
            p = Pool(self.num_workers)
            p.map(self.process_one_batch,params)
            p.close()

            complex_graphs_all = []
            for i in range(len(complex_names_all)//self.parallel_count+1):
                l, _ = self._load_cache(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), f"heterographs_split_{i}")
                complex_graphs_all.extend(l)
            self._save_cache(os.path.join(self.full_cache_path, f"heterographs.pkl"), complex_graphs_all)

            rdkit_ligands_all = []
            for i in range(len(complex_names_all) // self.parallel_count + 1):
                l, _ = self._load_cache(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), f"rdkit_split_{i}")
                rdkit_ligands_all.extend(l)
            self._save_cache(os.path.join(self.full_cache_path, f"rdkit_ligands.pkl"), rdkit_ligands_all)
        else:
            complex_graphs, rdkit_ligands = [], []
            with tqdm(total=len(complex_names_all), desc='loading complexes') as pbar:
                for t in map(self.get_complex, zip(complex_names_all, [None] * len(complex_names_all), lm_embeddings_chains_all, [None] * len(complex_names_all), [None] * len(complex_names_all), [None] * len(complex_names_all))):
                    complex_graphs.extend(t[0])
                    rdkit_ligands.extend(t[1])
                    pbar.update()
            self._save_cache(os.path.join(self.full_cache_path, "heterographs.pkl"), complex_graphs)
            self._save_cache(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), rdkit_ligands)

    def inference_preprocessing(self):
        ligands_list = []
        receptors_list = []
        print('Reading molecules and generating local structures with RDKit')
        failed_ligand_indices = []
        for idx, ligand_description in tqdm(enumerate(self.ligand_descriptions)):
            try:
                mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
                if mol is not None:
                    mol = AddHs(mol)
                    generate_conformer(mol)
                    ligands_list.append(mol)
                else:
                    mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                    if mol is None:
                        raise Exception('RDKit could not read the molecule ', ligand_description)
                    if not self.keep_local_structures:
                        mol.RemoveAllConformers()
                        mol = AddHs(mol)
                        generate_conformer(mol)
                    ligands_list.append(mol)
            except Exception as e:
                print('Failed to read molecule ', ligand_description, ' We are skipping it. The reason is the exception: ', e)
                failed_ligand_indices.append(idx)
                continue
            if '.pdb' in self.protein_path_list[idx]:
                receptor_pdb = PDBParser(QUIET=True).get_structure('pdb', self.protein_path_list[idx])
            elif 'cif' in self.protein_path_list[idx]:
                receptor_pdb = MMCIFParser().get_structure('cif', self.protein_path_list[idx])
            receptors_list.append(receptor_pdb)
        for index in sorted(failed_ligand_indices, reverse=True):
            del self.protein_path_list[index]
            del self.ligand_descriptions[index]
            del self.name_list[index]

        if self.esm_embeddings_path is not None:
            print('Reading language model embeddings.')
            lm_embeddings_chains_all = []
            if not os.path.exists(self.esm_embeddings_path): raise Exception('ESM embeddings path does not exist: ',self.esm_embeddings_path)
            for protein_path in self.protein_path_list:
                embeddings_paths = sorted(glob.glob(os.path.join(self.esm_embeddings_path, os.path.basename(protein_path)) + '*'))
                lm_embeddings_chains = []
                for embeddings_path in embeddings_paths:
                    lm_embeddings_chains.append(torch.load(embeddings_path)['representations'][33])
                lm_embeddings_chains_all.append(lm_embeddings_chains)
        else:
            lm_embeddings_chains_all = [None] * len(self.protein_path_list)
        print('Generating graphs for ligands and proteins')
        if self.num_workers > 1:
            # running preprocessing in parallel on multiple workers and saving the progress every self.parallel_count complexes
            for i in range(len(self.protein_path_list)//self.parallel_count+1):
                if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                    continue
                protein_paths_chunk = self.protein_path_list[self.parallel_count*i:self.parallel_count*(i+1)]
                ligand_description_chunk = self.ligand_descriptions[self.parallel_count*i:self.parallel_count*(i+1)]
                ligands_chunk = ligands_list[self.parallel_count * i:self.parallel_count * (i + 1)]
                lm_embeddings_chains = lm_embeddings_chains_all[self.parallel_count*i:self.parallel_count*(i+1)]
                complex_graphs, rdkit_ligands = [], []
                if self.num_workers > 1:
                    p = Pool(self.num_workers, maxtasksperchild=1)
                    p.__enter__()
                with tqdm(total=len(protein_paths_chunk), desc=f'loading complexes {i}/{len(protein_paths_chunk)//self.parallel_count+1}') as pbar:
                    map_fn = p.imap_unordered if self.num_workers > 1 else map
                    for t in map_fn(self.get_complex, zip(protein_paths_chunk, lm_embeddings_chains, ligands_chunk,ligand_description_chunk)):
                        complex_graphs.extend(t[0])
                        rdkit_ligands.extend(t[1])
                        pbar.update()
                if self.num_workers > 1: p.__exit__(None, None, None)

                self._save_cache(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), complex_graphs)
                self._save_cache(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), rdkit_ligands)

            complex_graphs_all = []
            for i in range(len(self.protein_path_list)//self.parallel_count+1):
                l, _ = self._load_cache(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), f"heterographs_split_{i}")
                complex_graphs_all.extend(l)
            self._save_cache(os.path.join(self.full_cache_path, f"heterographs.pkl"), complex_graphs_all)

            rdkit_ligands_all = []
            for i in range(len(self.protein_path_list) // self.parallel_count + 1):
                l, _ = self._load_cache(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), f"rdkit_split_{i}")
                rdkit_ligands_all.extend(l)
            self._save_cache(os.path.join(self.full_cache_path, f"rdkit_ligands.pkl"), rdkit_ligands_all)
        else:
            complex_graphs, rdkit_ligands, receptor_pdbs = [], [], []
            with tqdm(total=len(self.protein_path_list), desc='loading complexes') as pbar:
                for t in map(self.get_complex, zip(self.name_list, self.protein_path_list, lm_embeddings_chains_all, ligands_list, receptors_list, self.ligand_descriptions)):
                    complex_graphs.extend(t[0])
                    rdkit_ligands.extend(t[1])
                    receptor_pdbs.extend(t[2])
                    pbar.update()
            self._save_cache(os.path.join(self.full_cache_path, "heterographs.pkl"), complex_graphs)
            self._save_cache(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), rdkit_ligands)
            self._save_cache(os.path.join(self.full_cache_path, "receptor_pdbs.pkl"), receptor_pdbs)

    def get_complex(self, par):
        name, protein_path, lm_embedding_chains, ligand, receptor_pdb, ligand_description = par
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
            print("Folder not found", name)
            return [], []

        if ligand is not None:
            rec_model = parse_pdb_from_path(protein_path)
            af2_rec_model = None
            ligs = [ligand]
        else:
            try:
                rec_model, af2_rec_model = parse_receptor(name, self.pdbbind_dir)
            except Exception as e:
                print(f'Skipping {name} because of the error:')
                print(e)
                return [], []
            ligs = read_mols(self.pdbbind_dir, name, remove_hs=False)
        rec_pdbs = [receptor_pdb]
        complex_graphs = []
        failed_indices = []
        for i, lig in enumerate(ligs):
            if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                continue
            complex_graph = HeteroData()
            complex_graph.name = name
            if self.info is not None:
                complex_graph.affinity = torch.tensor(self.info.loc[self.info['name']==name,'affinity'].values[[0]]).float()
                complex_graph.gap_masks = torch.tensor([[int(x)] for x in self.info.loc[self.info['name']==name,'gap_mask'].values[0]]).float()
            try:
                get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                            self.num_conformers, remove_hs=self.remove_hs)
                rec, rec_coords, c_alpha_coords, n_coords, c_coords, chis, chi_masks, lm_embeddings = extract_receptor_structure(copy.deepcopy(rec_model), lig, lm_embedding_chains=lm_embedding_chains)
                rec_pdbs = [rec]
                if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                    print(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
                    print(len(c_alpha_coords),len(lm_embeddings))
                    failed_indices.append(i)
                    continue

                get_rec_graph(name,rec, af2_rec_model, rec_coords, c_alpha_coords, n_coords, c_coords, chis, chi_masks, complex_graph, rec_radius=self.receptor_radius,
                              c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                              atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)

                self._apply_plip_interactions(complex_graph, name)

            except Exception as e:
                print(f'Skipping {name} because of the error:')
                print(e)
                failed_indices.append(i)
                continue

            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
            complex_graph['receptor'].pos -= protein_center
            complex_graph['receptor'].lf_3pts -= protein_center[None,...]
            if self.all_atoms:
                complex_graph['atom'].pos -= protein_center

            if not self.center_ligand:
                if (not self.matching) or self.num_conformers == 1:
                    complex_graph['ligand'].pos -= protein_center
                else:
                    for p in complex_graph['ligand'].pos:
                        p -= protein_center
            else:
                if (not self.matching) or self.num_conformers == 1:
                    complex_graph['ligand'].pos -= complex_graph['ligand'].pos.mean(0,keepdim=True)
                else:
                    for p in complex_graph['ligand'].pos:
                        p -= p.mean(0,keepdim=True)

            complex_graph.original_center = protein_center
            complex_graphs.append(complex_graph)
        for idx_to_delete in sorted(failed_indices, reverse=True):
            del ligs[idx_to_delete]
            del rec_pdbs[idx_to_delete]
        return complex_graphs, ligs, rec_pdbs


class PDBBindScoring(Dataset):
    def __init__(self, root, transform=None, info=None, cache_path='data/cache', split_path='data/', limit_complexes=0,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, center_ligand=False, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False, require_receptor=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, name_list=None, keep_local_structures=False, use_existing_cache=True,
                 use_plip=False, plip_cache_dir='data/cache_plip', plip_confidence_threshold=0.5, plip_interaction_types=None,
                 plip_auto_rebuild=False, plip_skip_invalid=True):

        super().__init__(root, transform)
        self.pdbbind_dir = root
        self.info = info
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.require_ligand = require_ligand
        self.require_receptor = require_receptor
        self.protein_path_list = protein_path_list
        self.name_list = name_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.use_plip = use_plip
        self.plip_cache_dir = plip_cache_dir
        self.plip_confidence_threshold = plip_confidence_threshold
        self.plip_interaction_types = plip_interaction_types if plip_interaction_types is not None else list(DEFAULT_INTERACTION_TYPES)
        self.cache_failures = defaultdict(list)
        self.plip_auto_rebuild = plip_auto_rebuild
        self.plip_skip_invalid = plip_skip_invalid
        if matching or protein_path_list is not None and ligand_descriptions is not None:
            cache_path += '_torsion'
        if all_atoms:
            cache_path += '_allatoms'
        self.full_cache_path = os.path.join(cache_path, f'limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                            + ('' if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + ('' if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode()))))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.center_ligand = center_ligand
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        # if not os.path.exists(os.path.join(self.full_cache_path, "ligand_graphs.pkl")) or not os.path.exists(os.path.join(self.full_cache_path, "receptor_graphs.pkl"))\
        #         or (require_ligand and not os.path.exists(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"))):
        os.makedirs(self.full_cache_path, exist_ok=True)
        # if protein_path_list is None or ligand_descriptions is None:
        #     self.preprocessing()
        # else:
        self.ligand_graphs, self.receptor_graphs, self.rdkit_ligands, self.receptor_pdbs = self.inference_preprocessing()

        # print('loading data from memory: ', os.path.join(self.full_cache_path, "ligand_graphs.pkl"))


        # with open(os.path.join(self.full_cache_path, "ligand_graphs.pkl"), 'rb') as f:
        #     self.ligand_graphs = pickle.load(f)
        # with open(os.path.join(self.full_cache_path, "receptor_graphs.pkl"), 'rb') as f:
        #     self.receptor_graphs = pickle.load(f)
        #
        # if require_ligand:
        #     with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
        #         self.rdkit_ligands = pickle.load(f)
        # if require_receptor:
        #     with open(os.path.join(self.full_cache_path, "receptor_pdbs.pkl"), 'rb') as f:
        #         self.receptor_pdbs = pickle.load(f)


    def len(self):
        return len(self.ligand_graphs)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.ligand_graphs[idx])
        receptor_graph = self.receptor_graphs[complex_graph.protein_path]
        for type in receptor_graph.node_types + receptor_graph.edge_types:
            for key, value in receptor_graph[type].items():
                complex_graph[type][key] = value

        protein_center = receptor_graph.original_center
        complex_graph.original_center = protein_center
        if not self.center_ligand:
            if (not self.matching) or self.num_conformers == 1:
                complex_graph['ligand'].pos -= protein_center
            else:
                for p in complex_graph['ligand'].pos:
                    p -= protein_center
        else:
            if (not self.matching) or self.num_conformers == 1:
                complex_graph['ligand'].pos -= complex_graph['ligand'].pos.mean(0,keepdim=True)
            else:
                for p in complex_graph['ligand'].pos:
                    p -= p.mean(0,keepdim=True)

        if self.require_ligand:
            complex_graph.mol = copy.deepcopy(self.rdkit_ligands[idx])
        if self.require_receptor:
            complex_graph.rec_pdb = copy.deepcopy(self.receptor_pdbs[complex_graph.protein_path])
        return complex_graph

    def _cache_metadata(self):
        return {
            'schema_version': CACHE_SCHEMA_VERSION,
            'use_plip': self.use_plip,
            'plip_interaction_types': list(self.plip_interaction_types),
        }

    def _save_cache(self, path, payload):
        wrapped = {'metadata': self._cache_metadata(), 'data': payload}
        with open(path, 'wb') as f:
            pickle.dump(wrapped, f)

    def _load_cache(self, path, description):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        metadata = {}
        data = obj
        if isinstance(obj, dict) and 'data' in obj:
            data = obj.get('data')
            metadata = obj.get('metadata', {})
        if metadata.get('schema_version') not in (None, CACHE_SCHEMA_VERSION):
            raise ValueError(f'{description} cache schema mismatch: {metadata.get("schema_version")}')
        return data, metadata

    def preprocessing(self):
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} complexes.')

        if self.esm_embeddings_path is not None:
            id_to_embeddings = torch.load(self.esm_embeddings_path)
            chain_embeddings_dictlist = defaultdict(list)
            for key, embedding in id_to_embeddings.items():
                key_name = key.split('_chain_')[0]
                if key_name in complex_names_all:
                    chain_embeddings_dictlist[key_name].append(embedding)
            lm_embeddings_chains_all = []
            for name in complex_names_all:
                lm_embeddings_chains_all.append(chain_embeddings_dictlist[name])
        else:
            lm_embeddings_chains_all = [None] * len(complex_names_all)

        if self.num_workers > 1:
            # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
            for i in range(len(complex_names_all)//1000+1):
                if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                    continue
                complex_names = complex_names_all[1000*i:1000*(i+1)]
                lm_embeddings_chains = lm_embeddings_chains_all[1000*i:1000*(i+1)]
                complex_graphs, rdkit_ligands = [], []
                if self.num_workers > 1:
                    p = Pool(self.num_workers, maxtasksperchild=1)
                    p.__enter__()
                with tqdm(total=len(complex_names), desc=f'loading complexes {i}/{len(complex_names_all)//1000+1}') as pbar:
                    map_fn = p.imap_unordered if self.num_workers > 1 else map
                    for t in map_fn(self.get_complex, zip(complex_names, lm_embeddings_chains, [None] * len(complex_names), [None] * len(complex_names))):
                        complex_graphs.extend(t[0])
                        rdkit_ligands.extend(t[1])
                        pbar.update()
                if self.num_workers > 1: p.__exit__(None, None, None)

                self._save_cache(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), complex_graphs)
                self._save_cache(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), rdkit_ligands)

            complex_graphs_all = []
            for i in range(len(complex_names_all)//1000+1):
                l, _ = self._load_cache(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), f"heterographs_split_{i}")
                complex_graphs_all.extend(l)
            self._save_cache(os.path.join(self.full_cache_path, f"heterographs.pkl"), complex_graphs_all)

            rdkit_ligands_all = []
            for i in range(len(complex_names_all) // 1000 + 1):
                l, _ = self._load_cache(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), f"rdkit_split_{i}")
                rdkit_ligands_all.extend(l)
            self._save_cache(os.path.join(self.full_cache_path, f"rdkit_ligands.pkl"), rdkit_ligands_all)
        else:
            complex_graphs, rdkit_ligands = [], []
            with tqdm(total=len(complex_names_all), desc='loading complexes') as pbar:
                for t in map(self.get_complex, zip(complex_names_all, [None] * len(complex_names_all), lm_embeddings_chains_all, [None] * len(complex_names_all), [None] * len(complex_names_all), [None] * len(complex_names_all))):
                    complex_graphs.extend(t[0])
                    rdkit_ligands.extend(t[1])
                    pbar.update()
            self._save_cache(os.path.join(self.full_cache_path, "heterographs.pkl"), complex_graphs)
            self._save_cache(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), rdkit_ligands)

    def inference_preprocessing(self):
        receptor_pdbs = {}
        print('Reading molecules')
        if self.esm_embeddings_path is not None:
            print('Reading language model embeddings.')
            lm_embeddings_chains_all = []
            if not os.path.exists(self.esm_embeddings_path): raise Exception('ESM embeddings path does not exist: ',self.esm_embeddings_path)
            unique_protein_path_list = set(self.protein_path_list)
            for protein_path in unique_protein_path_list:
                embeddings_paths = sorted(glob.glob(os.path.join(self.esm_embeddings_path, os.path.basename(protein_path)) + '*'))
                lm_embeddings_chains = []
                for embeddings_path in embeddings_paths:
                    lm_embeddings_chains.append(torch.load(embeddings_path)['representations'][33])
                lm_embeddings_chains_all.append(lm_embeddings_chains)
        else:
            lm_embeddings_chains_all = [None] * len(unique_protein_path_list)
        print('Generating graphs for ligands and proteins')
        receptor_graphs = {}

        for i,protein_path in tqdm(enumerate(unique_protein_path_list), desc='parse receptor', total=len(unique_protein_path_list)):
            receptor_graph = HeteroData()

            rec_model = parse_pdb_from_path(protein_path)
            rec, rec_coords, c_alpha_coords, n_coords, c_coords, chis, chi_masks, lm_embeddings = extract_receptor_structure(copy.deepcopy(rec_model), None, lm_embedding_chains=lm_embeddings_chains_all[i])

            if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                assert 1 == 0,f'LM embeddings for {protein_path} did not have the right length for the protein.'

            get_rec_graph(protein_path, rec, None, rec_coords, c_alpha_coords, n_coords, c_coords, chis, chi_masks, receptor_graph, rec_radius=self.receptor_radius,
                              c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                              atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)

            protein_center = torch.mean(receptor_graph['receptor'].pos, dim=0, keepdim=True)
            receptor_graph['receptor'].pos -= protein_center
            if self.all_atoms:
                receptor_graph['atom'].pos -= protein_center
            receptor_graph.original_center = protein_center
            receptor_graphs[protein_path] = receptor_graph

            if '.pdb' in protein_path:
                receptor_pdb = PDBParser(QUIET=True).get_structure('pdb', protein_path)
            elif 'cif' in protein_path:
                receptor_pdb = MMCIFParser().get_structure('cif', protein_path)
            receptor_pdbs[protein_path] = receptor_pdb

        # with open(os.path.join(self.full_cache_path, "receptor_graphs.pkl"), 'wb') as f:
        #     pickle.dump((receptor_graphs), f)
        # with open(os.path.join(self.full_cache_path, "receptor_pdbs.pkl"), 'wb') as f:
        #     pickle.dump((receptors_list), f)

        if self.num_workers > 1:
            # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
            for i in range(len(self.protein_path_list)//1000+1):
                if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                    continue
                protein_paths_chunk = self.protein_path_list[1000*i:1000*(i+1)]
                ligand_description_chunk = self.ligand_descriptions[1000*i:1000*(i+1)]
                ligands_chunk = ligands_list[1000 * i:1000 * (i + 1)]
                lm_embeddings_chains = lm_embeddings_chains_all[1000*i:1000*(i+1)]
                complex_graphs, rdkit_ligands = [], []
                if self.num_workers > 1:
                    p = Pool(self.num_workers, maxtasksperchild=1)
                    p.__enter__()
                with tqdm(total=len(protein_paths_chunk), desc=f'loading complexes {i}/{len(protein_paths_chunk)//1000+1}') as pbar:
                    map_fn = p.imap_unordered if self.num_workers > 1 else map
                    for t in map_fn(self.get_complex, zip(protein_paths_chunk, lm_embeddings_chains, ligands_chunk,ligand_description_chunk)):
                        complex_graphs.extend(t[0])
                        rdkit_ligands.extend(t[1])
                        pbar.update()
                if self.num_workers > 1: p.__exit__(None, None, None)

                self._save_cache(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), complex_graphs)
                self._save_cache(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), rdkit_ligands)

            complex_graphs_all = []
            for i in range(len(self.protein_path_list)//1000+1):
                l, _ = self._load_cache(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), f"heterographs_split_{i}")
                complex_graphs_all.extend(l)
            self._save_cache(os.path.join(self.full_cache_path, f"heterographs.pkl"), complex_graphs_all)

            rdkit_ligands_all = []
            for i in range(len(self.protein_path_list) // 1000 + 1):
                l, _ = self._load_cache(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), f"rdkit_split_{i}")
                rdkit_ligands_all.extend(l)
            self._save_cache(os.path.join(self.full_cache_path, f"rdkit_ligands.pkl"), rdkit_ligands_all)
        else:

            ligand_graphs, rdkit_ligands = [], []
            with tqdm(total=len(self.protein_path_list), desc='loading complexes') as pbar:
                for t in map(self.get_complex, zip(self.name_list, self.protein_path_list, self.ligand_descriptions)):
                    if t[0] is None:continue
                    ligand_graphs.append(t[0])
                    rdkit_ligands.append(t[1])
                    pbar.update()
            # with open(os.path.join(self.full_cache_path, "ligand_graphs.pkl"), 'wb') as f:
            #     pickle.dump((ligand_graphs), f)
            # with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'wb') as f:
            #     pickle.dump((rdkit_ligands), f)
        return ligand_graphs, receptor_graphs, rdkit_ligands, receptor_pdbs
    def get_complex(self, par):
        name, protein_path, ligand_description = par
        try:
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    raise Exception('RDKit could not read the molecule ', ligand_description)
                if not self.keep_local_structures:
                    mol.RemoveAllConformers()
                    mol = AddHs(mol)
                    generate_conformer(mol)
        except Exception as e:
            print('Failed to read molecule ', ligand_description, ' We are skipping it. The reason is the exception: ', e)
            return None, None
        lig = mol
        ligand_graph = HeteroData()
        ligand_graph['name'] = name
        ligand_graph['protein_path'] = protein_path
        try:
            get_lig_graph_with_matching(lig, ligand_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                        self.num_conformers, remove_hs=self.remove_hs)
        except Exception as e:
            print(f'get_complex, Skipping {name} because of the error:')
            print(e)
            return None,None

        return ligand_graph, lig


def print_statistics(complex_graphs):
    statistics = ([], [], [], [])

    for complex_graph in complex_graphs:
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)

    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching']
    print('Number of complexes: ', len(complex_graphs))
    for i in range(4):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")


def construct_loader(args, t_to_sigma):
    transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms)

    common_args = {'transform': transform, 'root': args.data_dir, 'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                   'esm_embeddings_path': args.esm_embeddings_path,
                   'use_plip': args.use_plip,
                   'plip_cache_dir': args.plip_cache_dir,
                   'plip_confidence_threshold': args.plip_confidence_threshold,
                   'plip_interaction_types': args.plip_interaction_types.split(',') if args.plip_interaction_types else None,
                   'plip_auto_rebuild': getattr(args, 'plip_auto_rebuild', False),
                   'plip_skip_invalid': getattr(args, 'plip_skip_invalid', True),
                   'plip_max_interactions_per_residue': getattr(args, 'plip_max_interactions_per_residue', None),
                   'plip_max_interactions_per_ligand_atom': getattr(args, 'plip_max_interactions_per_ligand_atom', None),
                   'plip_distance_max': getattr(args, 'plip_distance_max', None),
                   'plip_min_confidence': getattr(args, 'plip_min_confidence', None)}
    info=pd.read_csv(args.info_path,dtype={'gap_mask':str})
    train_dataset = PDBBind(info=info,cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                            num_conformers=args.num_conformers, **common_args)
    ligand_names = []
    for x in train_dataset.complex_graphs:
        name = x.name
        if '_' in name:
            ligand_names.append(name.split('_')[1])
        else:
            ligand_names.append(name)
    ligand_names = pd.Series(ligand_names)
    ligand_count = ligand_names.map(ligand_names.value_counts())
    sample_weight = (1./ligand_count).tolist()
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
    val_dataset = PDBBind(info=info,cache_path=args.cache_path, split_path=args.split_val, keep_original=True, **common_args)

    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_loader = loader_class(dataset=train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, drop_last=True, pin_memory=args.pin_memory)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=False, pin_memory=args.pin_memory)

    return train_loader, val_loader


def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = None
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf"):
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
        if lig is not None:
            break
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
            if '.' in MolToSmiles(lig):
                continue
            if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs
