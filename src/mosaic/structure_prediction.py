#####################
#
#   Uniform interface for structure prediction models: generating features, building losses, and running structure prediction.
#
#


# TODO: remove lots of redundant code.
# It would be nice if all models `predict` method went through `model_output` ( right now this is only the case for Protenix and AF2).

import gemmi
from dataclasses import dataclass
from abc import ABC, abstractmethod
from jaxtyping import Array, Float, PyTree

from mosaic.losses.structure_prediction import AbstractStructureOutput
from mosaic.common import LossTerm, LinearCombination

class PolymerType:
    PROTEIN = "PROTEIN"
    RNA = "RNA"
    DNA = "DNA"

@dataclass(frozen=True, eq=True, slots=True)
class TargetChain:
    sequence: str
    polymer_type: str = PolymerType.PROTEIN
    use_msa: bool = True
    template_chain: gemmi.Chain | None = None


class StructureWriter(ABC):
    pass

@dataclass(frozen=True, eq=True, slots=True)
class StructurePrediction:
    st: gemmi.Structure
    plddt: Float[Array, "N"]
    pae: Float[Array, "N N"]
    iptm: float


class StructurePredictionModel(ABC):
    @abstractmethod
    def target_only_features(self, chains: list[TargetChain]) -> tuple[PyTree, StructureWriter]:
        """
        Generate model features and postprocessor for the target chains only.

        Args:
            chains: List of TargetChain objects representing the target chains.
            
        Returns:
            tuple of (PyTree, StructureWriter) containing the generated features and an object for turning a prediction into a gemmi.Structure.

        """
        pass

    @abstractmethod
    def binder_features(self, binder_length: int, chains: list[TargetChain]) -> tuple[PyTree, StructureWriter]:
        """
        Generate model features and postprocessor for a binder of given length and the target chains.

        Args:
            binder_length: Length of the binder chain.
            chains: List of TargetChain objects representing the target chains.

        Returns:
            tuple of (PyTree, StructureWriter) containing the generated features and an object for turning a prediction into a gemmi.Structure.

        """
        pass


    @abstractmethod
    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        writer: StructureWriter,
        recycling_steps: int = 1,
        sampling_steps: int | None = None,
        key,
    ) -> StructurePrediction:
        pass
       
    @abstractmethod
    def model_output(self, *, PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        recycling_steps: int = 1,
        sampling_steps: int | None = None,
        key) -> AbstractStructureOutput:
        pass


    @abstractmethod
    def build_loss(self, *, loss: LossTerm | LinearCombination, features: PyTree,  recycling_steps: int = 1, sampling_steps: int | None = None,) -> LossTerm:
        pass


