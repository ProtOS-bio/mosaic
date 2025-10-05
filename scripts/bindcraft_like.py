import os, fire
from typing import Dict
import yaml
import pickle
from dataclasses import dataclass, field, asdict, is_dataclass, fields
import jax
import numpy as np
import pandas as pd
from mosaic.models.boltz2 import Boltz2
from mosaic.models.af2 import AlphaFold2
from mosaic.proteinmpnn.mpnn import ProteinMPNN
from mosaic.optimizers import simplex_APGM
from mosaic.common import TOKENS
from mosaic.structure_prediction import TargetChain
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
import mosaic.losses.structure_prediction as sp
from mosaic.proteinmpnn.utils import mpnn_gen_sequence, get_binder_seqs
from mosaic.alphafold.utils import af2_screen_mpnn_seqs
from mosaic.notebook_utils import gemmi_structure_from_models
from protodev.structure_prediction.common import get_msa_path_for_sequence


def from_dict(cls, data):
    """Recursively build nested dataclasses from dicts."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass type")
    kwargs = {}
    for f in fields(cls):
        val = data.get(f.name)
        if val is None:
            continue
        if is_dataclass(f.type):
            val = from_dict(f.type, val)
        kwargs[f.name] = val
    return cls(**kwargs)


def to_dict(obj):
    """Recursively turn dataclasses into plain dicts (tuples → lists)."""
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = to_dict(val)
        return result
    elif isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    else:
        return obj


def to_yaml(obj, path=None):
    """Dump dataclass (possibly nested) to YAML string or file."""
    d = to_dict(obj)
    yaml_str = yaml.safe_dump(d, sort_keys=False)
    if path:
        with open(path, "w") as f:
            f.write(yaml_str)
    return yaml_str


@dataclass(frozen=True)
class StructurePredictionLossConfig:
    within_binder_contact: float = 1.0
    binder_target_contact: float = 1.0
    within_binder_pae: float = 0.4
    target_binder_pae: float = 0.05
    binder_target_pae: float = 0.05
    iptm: float = 0.025
    ptm: float = 0.025
    plddt: float = 0.1
    rardius_of_gyration: float = 0.3
    helix: float = -0.3


@dataclass(frozen=True)
class AF2ScreeningConfig:
    num_mpnn_seqs: int = 16
    model_indices: tuple[int] = (0,)
    recycles: int = 3
    plddt: float = 0.8
    iptm: float = 0.50
    ipae: float = 0.35
    rmsd: float = 5.0


@dataclass(frozen=True)
class BindCraftLikeConfig:
    target_sequence: str
    output_dir: str
    binder_length_range: tuple[int] = (60, 100)
    structure_loss: StructurePredictionLossConfig = field(default_factory=StructurePredictionLossConfig)
    screening: AF2ScreeningConfig = field(default_factory=AF2ScreeningConfig)
    random_seed: int = 0
    num_trajectories: int = 10
    num_design_steps: int = 150
    design_stepsize: float = 0.1
    design_momentum: float = 0.0
    mpnn_temp: float = 0.01
    mpnn_weight: float = 5.0


def main(config: BindCraftLikeConfig | str | Dict):
    if isinstance(config, str):
        with open(config) as f:
            config = from_dict(BindCraftLikeConfig, yaml.safe_load(f))
    elif isinstance(config, Dict):
        config = from_dict(BindCraftLikeConfig, config)

    os.makedirs(config.output_dir, exist_ok=True)
    to_yaml(config, os.path.join(config.output_dir, "config.yaml"))

    succesful_trajectories = 0
    trajectory_index = 0
    
    model_af = AlphaFold2()
    model_boltz2 = Boltz2()
    mpnn = ProteinMPNN.from_pretrained()

    target_msa = get_msa_path_for_sequence(config.target_sequence)
    template_features, template_writer = model_boltz2.target_only_features(chains=[TargetChain(sequence=config.target_sequence, 
                                                                                               use_msa=True, 
                                                                                               msa_a3m_path=target_msa)])

    template_st = model_boltz2.predict(
        PSSM=jax.nn.one_hot([TOKENS.index(c) for c in config.target_sequence], 20),
        features=template_features,
        writer=template_writer,
        key=jax.random.PRNGKey(config.random_seed),
    )

    while succesful_trajectories < config.num_trajectories:
        try:
            run_single_trajectory(config, model_af, model_boltz2, mpnn, template_st, trajectory_index=trajectory_index, target_msa=target_msa)
            succesful_trajectories += 1
        except Exception as e:
            print(f"Trajectory failed with error: {e}. Retrying...")
        trajectory_index += 1
    

def run_single_trajectory(config: BindCraftLikeConfig, model_af: AlphaFold2, model_boltz2: Boltz2, mpnn: ProteinMPNN, template_st, trajectory_index: int, target_msa: str):
    binder_length = np.random.randint(config.binder_length_range[0], config.binder_length_range[1]+1)
    af_binder_features, _ = model_af.binder_features(binder_length=binder_length, chains=[TargetChain(config.target_sequence, use_msa=False, template_chain=template_st.st[0][0])])

    structure_loss = (
        config.structure_loss.within_binder_contact * sp.WithinBinderContact() +
        config.structure_loss.binder_target_contact * sp.BinderTargetContact() +
        config.structure_loss.within_binder_pae * sp.WithinBinderPAE() +
        config.structure_loss.target_binder_pae * sp.TargetBinderPAE() +
        config.structure_loss.binder_target_pae * sp.BinderTargetPAE() +
        config.structure_loss.iptm * sp.IPTMLoss() +
        config.structure_loss.ptm * sp.pTMEnergy() +
        config.structure_loss.plddt * sp.PLDDTLoss() +
        config.structure_loss.rardius_of_gyration * sp.DistogramRadiusOfGyration() +
        config.structure_loss.helix * sp.HelixLoss()
    )
    mpnn_loss  = config.mpnn_weight * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(config.mpnn_temp))

    loss = model_af.build_loss(
        loss=structure_loss + mpnn_loss,
        features=af_binder_features,
        recycling_steps=3
    )

    _, PSSM, trajectory = simplex_APGM(
        loss_function=loss,
        n_steps=config.num_design_steps,
        x=jax.nn.softmax(
            0.5*jax.random.gumbel(
                key=jax.random.key(config.random_seed + trajectory_index),
                shape=(binder_length, 20),
            )
        ),
        trajectory_fn=lambda aux, x: {**aux, "PSSM": x},
        stepsize=config.design_stepsize,
        momentum=config.design_momentum,
    )

    traj_dir = os.path.join(config.output_dir, f"trajectory_{trajectory_index}")
    os.makedirs(traj_dir, exist_ok=True)

    with open(os.path.join(traj_dir, "trajectory.pkl"), "wb") as f:
        pickle.dump(trajectory, f)

    features, structure_writer = model_boltz2.binder_features(binder_length=binder_length, chains = [TargetChain(config.target_sequence, use_msa=True, 
                                                                                                                 msa_a3m_path=target_msa)])

    predicted_st = model_boltz2.predict(
        PSSM=PSSM,
        features=features,
        writer=structure_writer,
        key=jax.random.PRNGKey(config.random_seed),
    )

    full_seqs = mpnn_gen_sequence(predicted_st.st, num_seqs=config.screening.num_mpnn_seqs)
    mpnn_seqs, _ = get_binder_seqs(full_seqs, binder_length)

    # PSSM argmax sequence
    amax_seq = "".join([TOKENS[int(jax.numpy.argmax(p))] for p in PSSM])
    mpnn_seqs += [amax_seq]

    passed, rejected = af2_screen_mpnn_seqs(
        af2=model_af,
        features=af_binder_features,
        binder_seqs=mpnn_seqs,
        trajectory_model=predicted_st.st[0],
        model_indices=config.screening.model_indices,
        recycling_steps=config.screening.recycles,
        plddt_thresh=config.screening.plddt,
        iptm_thresh=config.screening.iptm,
        ipae_thresh=config.screening.ipae,
        rmsd_thresh=config.screening.rmsd,
        rng_seed=config.random_seed,
        return_rejects=True,
    )

    passed_df = pd.DataFrame([{'seq': p.seq, 'plddt': p.plddt, 'ipae': p.ipae, 'iptm': p.iptm, 'rmsd': p.rmsd} for p in passed])
    passed_df['passed'] = True
    rejected_df = pd.DataFrame(rejected)
    rejected_df['passed'] = False
    all_results_df = pd.concat([passed_df, rejected_df], ignore_index=True)
    all_results_df['name'] = [f"trajectory_{trajectory_index}_mpnn_{i}" for i in range(len(all_results_df))]
    all_results_df[all_results_df['seq'] == amax_seq]['name'] = f"trajectory_{trajectory_index}_PSSM_argmax"
    all_results_df.to_csv(os.path.join(traj_dir, "screening_results.csv"), index=False)

    if len(passed) > 0:
        print(f"Trajectory {trajectory_index} succeeded with {len(passed)} designs passing AF2 screening.")
        
    else:
        raise RuntimeError("No designs passed AF2 screening.")

if __name__ == "__main__":
        fire.Fire(main)
        exit(0)