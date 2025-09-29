import math, gemmi, numpy as np, jax, jax.numpy as jnp
from typing import List
from mosaic.proteinmpnn.mpnn import ProteinMPNN


AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
PAD_IDX = 20  # 20 AAs + PAD -> 21 channels


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