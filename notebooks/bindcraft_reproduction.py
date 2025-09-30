#%%
%load_ext autoreload
%autoreload 2
#%%
import jax
import numpy as np
#%%
from mosaic.models.boltz2 import Boltz2
from mosaic.models.af2 import AlphaFold2
from mosaic.proteinmpnn.mpnn import ProteinMPNN

#%%
from mosaic.optimizers import simplex_APGM
from mosaic.common import TOKENS
from mosaic.structure_prediction import TargetChain
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
import mosaic.losses.structure_prediction as sp
from mosaic.proteinmpnn.utils import mpnn_gen_sequence, get_binder_seqs
from mosaic.alphafold.utils import af2_screen_mpnn_seqs
from mosaic.notebook_utils import pdb_viewer, visualize_trajectory, gemmi_structure_from_models

# %%
model_af = AlphaFold2()
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
binder_length = 60
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
visualize_trajectory(trajectory)
# %%
features, structure_writer = model_boltz2.binder_features(binder_length=binder_length, chains = [TargetChain(target_sequence, use_msa=True, msa_a3m_path=pdl1_msa)])

predicted_st = model_boltz2.predict(
    PSSM=PSSM,
    features=features,
    writer=structure_writer,
    key=jax.random.PRNGKey(0),
)
pdb_viewer(predicted_st.st)
# %%
full_seqs = mpnn_gen_sequence(predicted_st.st, num_seqs=16)
mpnn_seqs, mpnn_scores = get_binder_seqs(full_seqs, binder_length)
# %%
mpnn_struct = []
for i, seq in enumerate(mpnn_seqs):
    mpnn_struct.append(
        model_boltz2.predict(
            PSSM=jax.nn.one_hot([TOKENS.index(c) for c in seq], 20),
            features=features,
            writer=structure_writer,
            key=jax.random.PRNGKey(i+1234),
        )
    )
complexes = gemmi_structure_from_models("designs", [st.st[0] for st in mpnn_struct])
pdb_viewer(complexes)
# %%
passed, rejected = af2_screen_mpnn_seqs(
    af2=model_af,
    features=af_binder_features,
    binder_seqs=mpnn_seqs,
    trajectory_model=predicted_st.st[0],
    model_indices=(0,),
    recycling_steps=3,
    rng_seed=42,
    save_pdb=False,
    return_rejects=True,
)
# %%
# %%

