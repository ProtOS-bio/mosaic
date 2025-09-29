#%%
%load_ext autoreload
%autoreload 2
#%%
import jax
import marimo as mo
import os
import numpy as np
import matplotlib.pyplot as plt

#%%
from mosaic.models.boltz2 import Boltz2
from mosaic.models.af2 import AlphaFold2
from mosaic.proteinmpnn.mpnn import ProteinMPNN

#%%
from mosaic.structure_prediction import TargetChain
from mosaic.losses.protein_mpnn import FixedStructureInverseFoldingLL, InverseFoldingSequenceRecovery
import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.models.af2 import AlphaFoldLoss
from mosaic.optimizers import simplex_APGM
from mosaic.notebook_utils import pdb_viewer

# %%
model_af = AlphaFold2('/content/mosaic')
model_boltz2 = Boltz2()
mpnn = ProteinMPNN.from_pretrained()

# %%
# PDL1
target_sequence = "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALEHHHHHH"
pdl1_msa = '/tmp/c071654bde2fa734272a562e46732c2f966d07d7be66e0f7c83018ffc9e5cab3.a3m'
# %%
template_features, template_writer = model_boltz2.target_only_features(chains=[TargetChain(sequence=target_sequence, use_msa=True, msa_a3m_path=pdl1_msa)])

# %%
template_st = model_boltz2.predict(
    PSSM=jax.nn.one_hot([TOKENS.index(c) for c in target_sequence], 20),
    features=template_features,
    writer=template_writer,
    key=jax.random.PRNGKey(0),
)
pdb_viewer(template_st.st)

# %%
binder_length = 85
af_binder_features, _ = model_af.binder_features(binder_length=binder_length, chains=[TargetChain(target_sequence, use_msa=False, template_chain=template_st.st[0][0])])

# %%
structure_loss = (
    # Contacts (binder compactness & interface formation)
    1.0 * sp.WithinBinderContact() +
    1.0 * sp.BinderTargetContact() + # mask to hotspots when provided

    # PAE terms (confidence / alignment error)
    0.4  * sp.WithinBinderPAE() +             # PAE within binder (intra)
    0.05 * sp.TargetBinderPAE() +             # PAE target→binder (inter, dir 1)
    0.05 * sp.BinderTargetPAE() +             # PAE binder→target (inter, dir 2)

    # ipTM / pTM energy (interface-specific confidence/energy)
    0.025 * sp.IPTMLoss() +                   # interface pTM
    0.025 * sp.pTMEnergy() +                  # “energy-style” ipTM (see note below)

    # pLDDT on the designed chain (keep the binder itself confident)
    0.10 * sp.PLDDTLoss()

    # Shape & secondary structure priors (BindCraft defaults)
    + 0.30  * sp.DistogramRadiusOfGyration()  
    + (-0.30) * sp.HelixLoss()                         # negative weight biases away from helices
    # + 0.10 * sp.TerminiDistance(target_distance=8.0) # enable if you need N/C proximity (optional)
)
mpnn_loss  = 5.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.01))

loss = model_af.build_loss(
    loss=structure_loss + mpnn_loss,
    features=af_binder_features,
    recycling_steps=3
)
# %%
_, PSSM, trajectory = simplex_APGM(
    loss_function=loss,
    n_steps=150,
    x=jax.nn.softmax(
        0.5*jax.random.gumbel(
            key=jax.random.key(np.random.randint(100000)),
            shape=(binder_length, 20),
        )
    ),
    trajectory_fn=lambda aux, x: {**aux, "PSSM": x},
    stepsize=0.1,
    momentum=0.0,
)

# %%
# show animation of PSSM evolution over optimization steps
# trajectory[i]['PSSM'] gives the PSSM at step i
def visualize_trajectory(trajectory):
    import matplotlib.animation as animation
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(trajectory[0]['PSSM'].T, aspect='auto', cmap='viridis', clim=(0, 0.5))
    fig.colorbar(cax, label='Probability')
    ax.set_yticks(ticks=np.arange(20), labels=[TOKENS[i] for i in range(20)])
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Amino Acid')
    ax.set_title('Amino Acid Probability Distribution Across Binder Sequence')
    def update(frame):
        ax.clear()
        cax = ax.imshow(trajectory[frame]['PSSM'].T, aspect='auto', cmap='viridis', clim=(0, 0.5))
        ax.set_yticks(ticks=np.arange(20), labels=[TOKENS[i] for i in range(20)])
        ax.set_xlabel('Residue Position')
        ax.set_ylabel('Amino Acid')
        ax.set_title(f'Step {frame+1}/{len(trajectory)}; loss: {trajectory[frame]["loss"]:.2f}; plddt: {trajectory[frame]['']['af2'][7]['plddt']:.1f}')
        return cax,
    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=200)
    plt.close()
    from IPython.display import HTML
    return HTML(ani.to_jshtml())

# %%
# _, PSSM_sharper, trajectory_sharper = simplex_APGM(
#     loss_function=loss,
#     n_steps=50,
#     x=PSSM,
#     trajectory_fn=lambda aux, x: {**aux, "PSSM": x},
#     stepsize = 0.5,
#     scale = 1.5,
#     momentum=0.0
# )
# %%
visualize_trajectory(trajectory)
# %%
features, structure_writer = model_boltz2.binder_features(binder_length=binder_length, chains = [TargetChain(target_sequence, use_msa=True, msa_a3m_path=pdl1_msa)])

predicted_st = model_boltz2.predict(
    PSSM=PSSM,
    features=features,
    writer=structure_writer,
    key=jax.random.PRNGKey(0),
)

# %%
pdb_viewer(predicted_st.st)
# %%


# %%
import math, gemmi, numpy as np, jax, jax.numpy as jnp
from typing import List
from mosaic.proteinmpnn.mpnn import ProteinMPNN

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
PAD_IDX = 20  # 20 AAs + PAD -> 21 channels

# --- helpers from your current code (unbatched packer + order) ----------------
def gemmi_to_mpnn_inputs(st: gemmi.Structure, ca_only: bool=False, chain_offset:int=1000):
    names = ["CA"] if ca_only else ["N","CA","C","O"]
    coords, mask, residx, chain_enc, aa_idx = [], [], [], [], []
    for ci, ch in enumerate(st[0]):
        r_i = 0
        for r in ch.get_polymer():
            ri = gemmi.find_tabulated_residue(r.name)
            if (not ri.found()) or (not ri.is_amino_acid()):
                continue
            atoms = {a.name: a for a in r}
            have_all = all(n in atoms for n in names)
            coords.append([atoms[n].pos.tolist() for n in names] if have_all else [[0.0,0.0,0.0]]*len(names))
            mask.append(1 if have_all else 0)
            residx.append(r_i + ci*chain_offset)
            chain_enc.append(ci)
            letter = (ri.fasta_code() or "X").upper()
            aa_idx.append({"A":0,"C":1,"D":2,"E":3,"F":4,"G":5,"H":6,"I":7,"K":8,
                           "L":9,"M":10,"N":11,"P":12,"Q":13,"R":14,"S":15,"T":16,
                           "V":17,"W":18,"Y":19}.get(letter, PAD_IDX))
            r_i += 1
    if not coords:
        raise ValueError("No amino-acid residues found.")
    X  = jnp.array(coords, dtype=jnp.float32)   # [L, A, 3]
    M  = jnp.array(mask, dtype=jnp.int32)       # [L]
    RI = jnp.array(residx, dtype=jnp.int32)     # [L]
    CE = jnp.array(chain_enc, dtype=jnp.int32)  # [L]
    S_nat = jnp.array(aa_idx, dtype=jnp.int32)  # [L]
    return X, M, RI, CE, S_nat

def _random_orders_batch(design_mask_1L: np.ndarray, B: int, seed: int):
    rng = np.random.RandomState(seed)
    L = design_mask_1L.size
    orders = np.full((B, L), 10_000, dtype=np.int32)
    idx = np.where(design_mask_1L)[0]
    for b in range(B):
        idx_b = idx.copy()
        rng.shuffle(idx_b)
        orders[b, idx_b] = np.arange(idx_b.size, dtype=np.int32)
    return jnp.array(orders)  # [B, L]

# --- batched autoregressive sampler ------------------------------------------
def mpnn_autoreg_from_structure_batched(
    *,
    pdb_path: str | None = None,
    st: gemmi.Structure | None = None,
    design_chain_indices: List[int] = [0],   # design first chain by default (binder-first)
    K: int = 64,
    batch_size: int = 16,
    temperature: float = 0.2,
    ca_only: bool = False,
    seed: int = 0,
):
    assert (pdb_path is not None) != (st is not None), "Provide exactly one of pdb_path or st."
    if pdb_path is not None:
        st = gemmi.read_structure(pdb_path)

    # Pack inputs (unbatched) and encode once
    X, M, RI, CE, S_nat = gemmi_to_mpnn_inputs(st, ca_only=ca_only)
    mpnn = ProteinMPNN.from_pretrained()
    key = jax.random.PRNGKey(seed)

    h_V, h_E, E_idx = mpnn.encode(
        X=X, mask=M, residue_idx=RI, chain_encoding_all=CE, key=key
    )

    L = X.shape[0]
    C = PAD_IDX + 1  # channels (21)
    ce_np   = np.array(CE)
    valid_np = (np.array(M) == 1)
    design_mask = np.isin(ce_np, np.array(design_chain_indices)).astype(bool) & valid_np
    n_steps = int(design_mask.sum())

    # One-hot S with PAD at design sites, native elsewhere
    S0 = jax.nn.one_hot(S_nat, num_classes=C).astype(jnp.float32)  # [L, 21]
    S0 = S0.at[design_mask].set(jax.nn.one_hot(jnp.array(PAD_IDX), C))  # PAD at design sites

    # Wrap decode to make it vmap-friendly (expects unbatched, returns [L,C])
    def decode_one(S_1L, order_1L):
        return mpnn.decode(S=S_1L, h_V=h_V, h_E=h_E, E_idx=E_idx, mask=M, decoding_order=order_1L)[0]
    batched_decode = jax.jit(jax.vmap(decode_one, in_axes=(0,0), out_axes=0))  # (B,[L,C])

    seqs = []
    n_batches = math.ceil(K / batch_size)
    for bi in range(n_batches):
        B = min(batch_size, K - bi*batch_size)
        # Expand S and orders for this batch
        S = jnp.broadcast_to(S0, (B, L, C)).copy()        # [B, L, C]
        orders = _random_orders_batch(design_mask, B, seed + 13 + bi)  # [B, L]

        for t in range(n_steps):
            # Decode all B sequences at once → logits [B,L,C]
            logits = batched_decode(S, orders)

            # Position to fill for each sequence at this step
            big = 10_000
            pos = jnp.argmin(jnp.where(orders == t, orders, big), axis=1)   # [B]

            # Grab per-seq AA logits (exclude PAD if present)
            aa_logits = logits[jnp.arange(B), pos, :20]    # [B, 20]
            keys = jax.random.split(jax.random.fold_in(key, (bi<<8) + t), B)
            aa = jax.vmap(lambda k, l: jax.random.categorical(k, l/temperature))(keys, aa_logits)  # [B]

            # Write tokens
            S = S.at[jnp.arange(B), pos, :].set(jax.nn.one_hot(aa, C))

        # Extract the first designed chain’s tokens → sequences
        design_chain = design_chain_indices[0]
        chain_pos = np.where((ce_np == design_chain) & valid_np)[0]
        tokens = jnp.argmax(S[:, chain_pos, :20], axis=-1).astype(jnp.int32)  # [B, len(chain_pos)]
        seqs.extend(["".join(AA[np.array(row)]) for row in np.array(tokens)])

    return seqs[:K]


# %%
seqs = mpnn_autoreg_from_structure_batched(st=predicted_st.st, K=16)

# %%
mpnn_struct = []
for i, seq in enumerate(seqs):
    mpnn_struct.append(
        model_boltz2.predict(
            PSSM=jax.nn.one_hot([TOKENS.index(c) for c in seq], 20),
            features=features,
            writer=structure_writer,
            key=jax.random.PRNGKey(i+1234),
        )
    )
# %%
from mosaic.notebook_utils import gemmi_structure_from_models
complexes = gemmi_structure_from_models("designs", [st.st[0] for st in mpnn_struct])
pdb_viewer(complexes)
# %%
# %%
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import jax, jax.numpy as jnp

# -----------------------
# BindCraft-like gates
# -----------------------
BC_THRESH = dict(plddt=0.80, iptm=0.50, ipae=0.35)

# -----------------------
# Utilities
# -----------------------

def _mean_plddt(plddt) -> float:
    p = np.asarray(plddt)
    if p.size == 0: return 0.0
    if p.max() > 1.5: p = p / 100.0  # 0..100 -> 0..1
    return float(np.mean(p))

def _split_masks(chain_index: np.ndarray, binder_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """binder mask, target mask. Fallback: binder comes first."""
    chain_index = np.asarray(chain_index)
    if chain_index.size >= 2:
        ids = np.unique(chain_index)
        return (chain_index == ids[0]), (chain_index == ids[1])
    L = chain_index.shape[0] if chain_index.size else binder_len * 2
    bmask = np.zeros(L, dtype=bool); bmask[:binder_len] = True
    return bmask, ~bmask

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
    )

@dataclass
class AF2ScreenHit:
    seq: str
    plddt: float
    iptm: float
    ipae: float
    pdb_str: Optional[str] = None
    extras: Optional[Dict] = None   # e.g., per-res pLDDT

# -----------------------
# Core screen (no batching, binder-only PSSM)
# -----------------------
def af2_screen_mpnn_seqs_with_pssm_minimal(
    *,
    af2,                                  # your initialized AF2 object (has .predict)
    features,                             # precomputed features PyTree (reused)
    binder_seqs: List[str],
    model_indices: Sequence[int] = (0,),  # e.g., (0,1,2,3,4) for ensemble
    recycling_steps: int = 1,
    rng_seed: int = 0,
    save_pdb: bool = False,
    return_rejects: bool = False,
):
    """
    For each binder seq:
      - build binder-only one-hot PSSM [Lb,20]
      - run af2.predict for each model_idx (one-by-one)
      - average metrics; apply BindCraft AF2 gates
    """
    key0 = jax.random.PRNGKey(rng_seed)
    passed: List[AF2ScreenHit] = []
    rejects: List[Dict] = []

    for i, seq in enumerate(binder_seqs):
        plddt_list, iptm_list, ipae_list = [], [], []
        pdb_keep = None
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

            bmask, tmask = _split_masks(np.asarray(chain_idx), binder_len=len(seq))
            ipae = _ipae_on_interface(np.asarray(pae), coords, bmask, tmask)

            plddt_list.append(_mean_plddt(plddt_arr))
            iptm_list.append(iptm); ipae_list.append(ipae)

            if j == 0 and save_pdb:
                pdb_keep = getattr(pred, "pdb_str", getattr(pred, "pdb", None))
            if j == 0:
                p = np.asarray(plddt_arr)
                plddt_per_res_keep = p/100.0 if p.max() > 1.5 else p

        metrics = dict(
            plddt=float(np.mean(plddt_list)),
            iptm=float(np.mean(iptm_list)),
            ipae=float(np.mean(ipae_list)),
        )

        if _passes(metrics):
            passed.append(
                AF2ScreenHit(
                    seq=seq,
                    plddt=metrics["plddt"],
                    iptm=metrics["iptm"],
                    ipae=metrics["ipae"],
                    pdb_str=pdb_keep,
                    extras=dict(plddt_per_res=plddt_per_res_keep) if plddt_per_res_keep is not None else None,
                )
            )
        elif return_rejects:
            rejects.append(dict(seq=seq, **metrics))

    return (passed, rejects) if return_rejects else (passed, [])

# %%
# seqs = ['PLPEEERRRRAEREERMRRSIAQIEEAYAVLLSKTTDPEDQAWVEKTKERMIEEIRQHWEA']
# af_binder_features, _ = model_af.binder_features(binder_length=len(seqs[0]), chains=[TargetChain(target_sequence, use_msa=False, template_chain=template_st.st[0][0])])

passed, rejected = af2_screen_mpnn_seqs_with_pssm_minimal(
    af2=model_af,
    features=af_binder_features,
    binder_seqs=seqs,
    model_indices=(0,),
    recycling_steps=3,
    rng_seed=42,
    save_pdb=False,
    return_rejects=True,
)
# %%
passed
# %%
