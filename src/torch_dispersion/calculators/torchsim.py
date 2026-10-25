"""TorchSim model for DFT-D4: ``D4Model(settings)``, a batched ``ModelInterface``.

Importing this module imports ``torch_sim`` (the ``torchsim`` extra); the package's top-level
``__init__`` does not import it, so the core install does not need torch-sim.

    from torch_sim.models.interface import SumModel
    from torch_dispersion.calculators.torchsim import D4Model

    model = SumModel(mlip, D4Model(D4Settings(s8=..., a1=..., a2=..., s9=0.0)))

The batch is evaluated by ``torch_dispersion.d4.d4_dispersion_batched``: one batched
neighbour list, then per system the graph stage, EEQ solve, tile kernels and chain closure on
that system's own C6 block. Energies, forces and stresses come from the analytic kernels plus
one internal autograd pass, so ``positions.requires_grad`` is not needed and all outputs are
detached.
"""

import torch
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState

from ..d4.batched import d4_dispersion_batched
from ..settings import D4Settings


class D4Model(ModelInterface):
    """DFT-D4 (dftd4 4.3.0) as a TorchSim model.

    Outputs, for a batch of S systems and N atoms (``dtype``, float64 by default):

    - ``energy`` (S,) eV;
    - ``forces`` (N, 3) eV/Å, if ``compute_forces``;
    - ``stress`` (S, 3, 3) eV/Å^3 = sym(dE/dε) / V (ASE and TorchSim sign convention), if
      ``compute_stress`` and the batch is periodic (molecules have no stress);
    - ``charges`` (N,) EEQ partial charges (e), if ``compute_charges``.

    The total charge of each system is the state's ``total_charge`` (or ``charge``) per-system
    extra if present, else 0. The batch's ``pbc`` must be all True or all False: TorchSim
    states carry one ``pbc`` for the whole batch, and D4 supports only molecules and 3D cells.

    ``batch_neighbor_list=False`` builds each system's neighbour list just before it is
    evaluated instead of all in one batched call: the peak memory of a per-system loop, at the
    cost of one list launch per system (see ``d4_dispersion_batched``).
    """

    def __init__(
        self,
        settings: D4Settings,
        *,
        compute_forces: bool = True,
        compute_stress: bool = True,
        compute_charges: bool = False,
        batch_neighbor_list: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.compute_charges = compute_charges
        self.batch_neighbor_list = batch_neighbor_list
        self._device = settings.torch_device
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        # the 60 bohr two-body list (and the 40 bohr ATM list if s9 != 0) dominates memory
        self._memory_scales_with = "n_atoms_x_density"

    @ModelInterface.compute_forces.setter
    def compute_forces(self, value: bool) -> None:
        self._compute_forces = value

    @ModelInterface.compute_stress.setter
    def compute_stress(self, value: bool) -> None:
        self._compute_stress = value

    def forward(self, state: SimState, **_kwargs) -> dict[str, torch.Tensor]:
        system_idx = state.system_idx
        if system_idx.numel() > 1 and bool((system_idx[1:] < system_idx[:-1]).any()):
            raise ValueError("D4Model needs the atoms ordered by system")
        n_systems = state.cell.shape[0]
        counts = torch.bincount(system_idx, minlength=n_systems)
        charge = getattr(state, "total_charge", None)
        if charge is None:
            charge = getattr(state, "charge", None)
        result = d4_dispersion_batched(
            state.atomic_numbers,
            state.positions,
            state.row_vector_cell,
            state.pbc,
            counts,
            self.settings,
            charges=charge,
            compute_forces=self._compute_forces,
            compute_stress=self._compute_stress,
            batch_neighbor_list=self.batch_neighbor_list,
        )
        out = {"energy": result.energy}
        if result.forces is not None:
            out["forces"] = result.forces
        if result.stress is not None:
            out["stress"] = result.stress
        if self.compute_charges:
            out["charges"] = result.charges
        return {k: v.detach().to(device=self._device, dtype=self._dtype) for k, v in out.items()}


__all__ = ["D4Model"]
