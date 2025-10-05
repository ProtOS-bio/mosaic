from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import gemmi
import jax, jax.numpy as jnp
from mosaic.common import TOKENS
from mosaic.proteinmpnn.utils import get_bb_coords_and_tmask
from protodev.utils.protein.metrics import calculate_rmsds


# BindCraft-like gates
BC_THRESH = dict(plddt=0.80, iptm=0.50, ipae=0.35, rmsd=5.0)


def _mean_plddt(plddt) -> float:
    p = np.asarray(plddt)
    if p.size == 0: return 0.0
    if p.max() > 1.5: p = p / 100.0  # 0..100 -> 0..1
    return float(np.mean(p))


def _ipae_on_interface(
    pae: np.ndarray,
    coords: Optional[np.ndarray],
    bmask: np.ndarray,
    tmask: np.ndarray,
    cutoff: float = 8.0,
) -> float:
    """Mean inter-chain PAE; if coords given, restrict to CA–CA ≤ cutoff Å. Normalize by ~31 Å."""
    pae = np.asarray(pae)
    inter = np.ix_(bmask, tmask)
    pae_bt = pae[inter]
    if pae_bt.size == 0:
        return 1.0
    if coords is not None and np.size(coords) > 0:
        CA = np.asarray(coords)[:, 1, :]  # [L,3], use CA (N,CA,C,O -> index 1)
        d2 = np.sum((CA[bmask][:, None, :] - CA[tmask][None, :, :])**2, axis=-1)
        near = d2 <= cutoff**2
        if near.any():
            pae_bt = pae_bt[near]
    return float(np.mean(pae_bt) / 31.0)


def _passes(metrics: Dict, thr: Dict = BC_THRESH) -> bool:
    return (
        metrics["plddt"] >= thr["plddt"]
        and metrics["iptm"] >= thr["iptm"]
        and metrics["ipae"] <= thr["ipae"]
        and metrics["rmsd"] <= thr["rmsd"]
    )


@dataclass
class AF2ScreenHit:
    seq: str
    plddt: float
    iptm: float
    ipae: float
    rmsd: float
    st_model: Optional[gemmi.Model] = None
    extras: Optional[Dict] = None   # e.g., per-res pLDDT


def af2_screen_mpnn_seqs(
    *,
    af2,                                  # your initialized AF2 object (has .predict)
    features,                             # precomputed features PyTree (reused)
    binder_seqs: List[str],
    trajectory_model: gemmi.Model,
    model_indices: Sequence[int] = (0,),  # e.g., (0,1,2,3,4) for ensemble
    recycling_steps: int = 1,
    rng_seed: int = 0,
    return_rejects: bool = False,
):
    """
    For each binder seq:
      - run af2.predict for each model_idx (one-by-one)
      - average metrics; apply BindCraft AF2 gates
    """
    key0 = jax.random.PRNGKey(rng_seed)
    passed: List[AF2ScreenHit] = []
    rejects: List[Dict] = []

    trajectory_bb, rmsd_tmask = get_bb_coords_and_tmask(trajectory_model)

    for i, seq in enumerate(binder_seqs):
        plddt_list, iptm_list, ipae_list, rmsd_list = [], [], [], []
        st_keep = None
        plddt_per_res_keep = None

        for j, midx in enumerate(model_indices):
            key = jax.random.fold_in(key0, (i << 8) + j)
            pred = af2.predict(
                PSSM=jax.nn.one_hot([TOKENS.index(c) for c in seq], 20),
                features=features,
                writer=None,
                recycling_steps=recycling_steps,
                sampling_steps=None,
                model_idx=int(midx),
                key=key,
            )

            # Pull fields (adjust names if your StructurePrediction differs)
            plddt_arr = getattr(pred, "plddt", getattr(pred, "plddt_confidence", None))
            if plddt_arr is None:
                raise RuntimeError("AF2 prediction missing pLDDT.")
            iptm = float(getattr(pred, "iptm", getattr(pred, "iptm_score", 0.0)))
            pae  = getattr(pred, "pae", getattr(pred, "predicted_aligned_error", None))
            if pae is None:
                raise RuntimeError("AF2 prediction missing PAE.")
            chain_idx = getattr(pred, "chain_index", getattr(pred, "atom_chain_index", np.array([])))
            coords    = getattr(pred, "backbone_coordinates", getattr(pred, "coords", None))

            L = pae.shape[0]
            binder_len = len(seq)
            bmask = np.zeros(L, dtype=bool); bmask[:binder_len] = True
            tmask = ~bmask
            ipae = _ipae_on_interface(np.asarray(pae), coords, bmask, tmask)

            plddt_list.append(_mean_plddt(plddt_arr))
            iptm_list.append(iptm); ipae_list.append(ipae)

            if j == 0:
                st_keep = pred.st[0]
                p = np.asarray(plddt_arr)
                plddt_per_res_keep = p/100.0 if p.max() > 1.5 else p
            design_bb, _ = get_bb_coords_and_tmask(pred.st[0], binder_chain='B', target_chain='C')
            rmsd_list.append(calculate_rmsds(trajectory_bb, design_bb, tmask=rmsd_tmask, atom_idx=0)['target_aligned_rmsd'])


        metrics = dict(
            plddt=float(np.mean(plddt_list)),
            iptm=float(np.mean(iptm_list)),
            ipae=float(np.mean(ipae_list)),
            rmsd=float(np.mean(rmsd_list)),
        )

        if _passes(metrics):
            passed.append(
                AF2ScreenHit(
                    seq=seq,
                    plddt=metrics["plddt"],
                    iptm=metrics["iptm"],
                    ipae=metrics["ipae"],
                    rmsd=metrics["rmsd"],
                    st_model=st_keep,
                    extras=dict(plddt_per_res=plddt_per_res_keep) if plddt_per_res_keep is not None else None,
                )
            )
        elif return_rejects:
            rejects.append(dict(seq=seq, **metrics))

    return (passed, rejects) if return_rejects else (passed, [])