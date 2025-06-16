from ase import Atoms

from ..base import EngineBase
from ..data import MTPData

from .conversion import BasisConverter, moments_count_to_level_map
from .jax_jax_deep2 import calc_energy_forces_stress as jax_calc
from .moment_jax import MomentBasis

import jax


class JaxMTPEngine(EngineBase):
    """MTP Engine in 'full tensor' version based on jax."""

    def __init__(self, *args, **kwargs):
        """Intialize the engine."""
        self.moment_basis = None
        self.basis_converter = None
        super().__init__(*args, **kwargs)

        #with jax.profiler.trace("/path/to/log"):
        #    forces, stress = jax_calc(...)

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        super().update(mtp_data)
        if self.mtp_data.alpha_moments_count is not None:
            level = moments_count_to_level_map[mtp_data.alpha_moments_count]
            if self.moment_basis is None:
                self.moment_basis = MomentBasis(level)
                self.moment_basis.init_moment_mappings()
                self.basis_converter = BasisConverter(self.moment_basis)
            elif self.moment_basis.max_level != level:
                raise RuntimeError(
                    "Changing moments/level is not allowed. "
                    "Use a new instance instead."
                )
            self.basis_converter.remap_mlip_moment_coeffs(self.mtp_data)
            
            
    def calculate_(self, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params):
        #self.update_neighbor_list(atoms)
        mtp_data = self.mtp_data
        
        # set params for gradient computation
        mtp_data.species_coeffs = params['species']
        mtp_data.radial_coeffs = params['radial']
        self.basis_converter.remapped_coeffs = params['basis']

        static_parameters = jax.device_put({self.moment_basis.basic_moments,self.moment_basis.pair_contractions,self.moment_basis.scalar_contractions,})

        energies, forces, stress = jax_calc(
            self,
            itypes,
            all_js,
            all_rijs,
            all_jtypes,
            cell_rank,
            volume,
            mtp_data.species,
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            mtp_data.species_coeffs,
            self.basis_converter.remapped_coeffs,
            mtp_data.radial_coeffs,
            static_parameters
        )
        results = {}
        results["energies"] = energies
        results["energy"] = results["energies"].sum()
        results["forces"] = forces
        results["stress"] = stress
        return results
    


    def calculate(self, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params):
        #self.update_neighbor_list(atoms)
        mtp_data = self.mtp_data
        
        # set params for gradient computation
        mtp_data.species_coeffs = params['species']
        mtp_data.radial_coeffs = params['radial']
        self.basis_converter.remapped_coeffs = params['basis']

        energies, forces, stress = jax_calc(
            self,
            itypes,
            all_js,
            all_rijs,
            all_jtypes,
            cell_rank,
            volume,
            mtp_data.species,
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            mtp_data.species_coeffs,
            self.basis_converter.remapped_coeffs,
            mtp_data.radial_coeffs,
            # Static parameters:
            self.moment_basis.basic_moments,
            self.moment_basis.pair_contractions,
            self.moment_basis.scalar_contractions,
        )
        results = {}
        results["energies"] = energies
        results["energy"] = results["energies"].sum()
        results["forces"] = forces
        results["stress"] = stress
        return results
    
