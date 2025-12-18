import os
import subprocess
import warnings
from datetime import datetime
import signal
from contextlib import contextmanager
import numpy as np
from numpy import dot, transpose, sqrt
from numpy.linalg import svd, det
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import RemoveHs, MolToPDBFile
from torch_geometric.nn.data_parallel import DataParallel

from models.all_atom_score_model import TensorProductScoreModel as AAScoreModel
from models.score_model import TensorProductScoreModel as CGScoreModel
from datasets.plip_extract import DEFAULT_INTERACTION_TYPES
from utils.diffusion_utils import get_timestep_embedding
from spyrmsd import rmsd, molecule


def get_obrmsd(mol1_path, mol2_path, cache_name=None):
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f') if cache_name is None else cache_name
    os.makedirs(".openbabel_cache", exist_ok=True)
    if not isinstance(mol1_path, str):
        MolToPDBFile(mol1_path, '.openbabel_cache/obrmsd_mol1_cache.pdb')
        mol1_path = '.openbabel_cache/obrmsd_mol1_cache.pdb'
    if not isinstance(mol2_path, str):
        MolToPDBFile(mol2_path, '.openbabel_cache/obrmsd_mol2_cache.pdb')
        mol2_path = '.openbabel_cache/obrmsd_mol2_cache.pdb'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return_code = subprocess.run(f"obrms {mol1_path} {mol2_path} > .openbabel_cache/obrmsd_{cache_name}.rmsd",
                                     shell=True)
        print(return_code)
    obrms_output = read_strings_from_txt(f".openbabel_cache/obrmsd_{cache_name}.rmsd")
    rmsds = [line.split(" ")[-1] for line in obrms_output]
    return np.array(rmsds, dtype=np.float)


def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def get_optimizer_and_scheduler(args, model, scheduler_mode='min'):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False):
    if 'all_atoms' in args and args.all_atoms:
        model_class = AAScoreModel
    else:
        model_class = CGScoreModel

    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type,
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale)

    lm_embedding_type = None
    if getattr(args, 'esm_embeddings_path', None) is not None: lm_embedding_type = 'esm'
    plip_interaction_types = getattr(args, 'plip_interaction_types', None)
    if isinstance(plip_interaction_types, str):
        plip_types = plip_interaction_types.split(',') if plip_interaction_types else list(DEFAULT_INTERACTION_TYPES)
    elif plip_interaction_types is None:
        plip_types = list(DEFAULT_INTERACTION_TYPES)
    else:
        plip_types = list(plip_interaction_types)
    plip_feat_dims = {'distance': 16, 'angle': 8}
    parsed_plip_feat_dims = getattr(args, 'plip_feat_dims', None)
    if parsed_plip_feat_dims:
        for token in parsed_plip_feat_dims.split(','):
            if '=' not in token:
                continue
            k, v = token.split('=')
            if k.strip() in plip_feat_dims:
                try:
                    plip_feat_dims[k.strip()] = int(v)
                except ValueError:
                    print(f'Could not parse plip_feat_dims token {token}, keeping default.')
    use_plip = getattr(args, 'use_plip', False)
    use_plip_features_requested = getattr(args, 'use_plip_features', False)
    if use_plip_features_requested and not use_plip:
        warnings.warn('use_plip_features requested but use_plip is False; disabling PLIP features for backward compatibility.', stacklevel=2)
    use_plip_features = use_plip_features_requested and use_plip
    plip_num_types = len(plip_types) if use_plip_features else 0

    model = model_class(t_to_sigma=t_to_sigma,
                        device=device,
                        no_torsion=args.no_torsion,
                        timestep_emb_func=timestep_emb_func,
                        num_conv_layers=args.num_conv_layers,
                        lig_max_radius=args.max_radius,
                        scale_by_sigma=args.scale_by_sigma,
                        sigma_embed_dim=args.sigma_embed_dim,
                        ns=args.ns, nv=args.nv,
                        distance_embed_dim=args.distance_embed_dim,
                        cross_distance_embed_dim=args.cross_distance_embed_dim,
                        batch_norm=not args.no_batch_norm,
                        dropout=args.dropout,
                        use_second_order_repr=args.use_second_order_repr,
                        cross_max_distance=args.cross_max_distance,
                        dynamic_max_cross=args.dynamic_max_cross,
                        lm_embedding_type=lm_embedding_type,
                        confidence_mode=confidence_mode,
                        num_confidence_outputs=len(
                            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                            args.rmsd_classification_cutoff, list) else 1,
                        use_plip=use_plip,
                        use_plip_features=use_plip_features,
                        plip_num_types=plip_num_types,
                        plip_distance_embed_dim=plip_feat_dims['distance'],
                        plip_angle_embed_dim=plip_feat_dims['angle'])

    if device.type == 'cuda' and not no_parallel:
        model = DataParallel(model)
    model.to(device)
    return model


def load_state_dict_flexible(model, state_dict, strict=True, log_path=None):
    """
    A flexible loader that tolerates shape mismatches and module prefixes while surfacing
    any dropped keys. Missing parameters keep their initialized values, which means EMA
    tracking will simply start from those initial weights.
    """
    parallel_wrappers = (torch.nn.DataParallel, DataParallel)
    target_model = model.module if isinstance(model, parallel_wrappers) else model
    current_state = target_model.state_dict()

    filtered = {}
    dropped = []
    renamed = []
    for k, v in state_dict.items():
        candidate_keys = [k]
        if k.startswith('module.'):
            candidate_keys.append(k[len('module.'):])

        matched = False
        for candidate in candidate_keys:
            if candidate in current_state and current_state[candidate].shape == v.shape:
                filtered[candidate] = v
                if candidate != k:
                    renamed.append((k, candidate))
                matched = True
                break
        if not matched:
            dropped.append(k)

    missing_keys, unexpected_keys = target_model.load_state_dict(filtered, strict=False)

    warnings_to_report = []
    if renamed:
        warnings_to_report.append(f'Stripped prefixes for {len(renamed)} keys (e.g., {renamed[0][0]} -> {renamed[0][1]}).')
    if dropped:
        warnings_to_report.append(f'Ignored {len(dropped)} mismatched keys: {dropped[:5]}{"..." if len(dropped) > 5 else ""}.')
    if missing_keys:
        warnings_to_report.append(f'Missing keys after flexible load: {missing_keys}.')
    if unexpected_keys:
        warnings_to_report.append(f'Unexpected keys after flexible load: {unexpected_keys}.')

    if warnings_to_report:
        context_notes = []
        if isinstance(model, parallel_wrappers):
            context_notes.append('Model is wrapped in DataParallel; keys were loaded on the underlying module.')
        context_notes.append('Missing parameters retain initialized values; EMA will track them from initialization.')
        message = ' '.join(warnings_to_report + context_notes)
        print(message)
        if log_path:
            log_path = os.fspath(log_path)
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(message + '\n')

    if strict and (missing_keys or unexpected_keys):
        print(f'Missing keys after flexible load: {missing_keys}')
        print(f'Unexpected keys after flexible load: {unexpected_keys}')
    return missing_keys, dropped, unexpected_keys


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
        )
        return RMSD

def get_align_rotran(coords,reference_coords):
    # center on centroid
    av1 = coords.mean(0,keepdims=True)
    av2 = reference_coords.mean(0,keepdims=True)
    coords = coords - av1
    reference_coords = reference_coords - av2
    # correlation matrix
    a = dot(transpose(coords), reference_coords)
    u, d, vt = svd(a)
    rot = transpose(dot(transpose(vt), transpose(u)))
    # check if we have found a reflection
    if det(rot) < 0:
        vt[2] = -vt[2]
        rot = transpose(dot(transpose(vt), transpose(u)))
    tran = av2 - dot(av1, rot)
    return tran, rot

class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]
