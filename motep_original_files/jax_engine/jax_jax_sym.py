import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from ase import Atoms

from jax import lax

# Import our pickle-based tensor operations
from .improved_symmetric import (
    pack_symmetric_single, 
    unpack_symmetric_single,
    custom_symmetric_tensordot,
    D_FIXED
)

jax.config.update("jax_enable_x64", False)

def get_types(atoms: Atoms, species: list[int] = None) -> np.ndarray:
    if species is None:
        return np.array(atoms.numbers, dtype=int)
    else:
        return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=int)

@partial(jax.jit, static_argnums=(9, 10, 11, 12))
def calc_energy_forces_stress(
    engine,
    itypes,
    all_js,
    all_rijs,
    all_jtypes,
    cell_rank,
    volume,
    species,
    scaling,
    min_dist,
    max_dist,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    local_energies, local_gradient = _jax_calc_local_energy_and_derivs(
        all_rijs,
        itypes,
        all_jtypes,
        species_coeffs,
        moment_coeffs,
        radial_coeffs,
        scaling,
        min_dist,
        max_dist,
        # Static parameters:
        radial_coeffs.shape[3],
        basic_moments,
        pair_contractions,
        scalar_contractions,
    )

    forces = jnp.array(local_gradient.sum(axis=1))
    forces = jnp.subtract.at(forces, all_js, local_gradient, inplace=False)

    stress = jnp.array((all_rijs.transpose((0, 2, 1)) @ local_gradient).sum(axis=0))
    
    def compute_stress_true(stress, volume):
        stress_sym = (stress + stress.T) * 0.5  # symmetrize
        stress_sym = stress_sym / volume
        indices = jnp.array([0, 4, 8, 5, 2, 1])
        return stress_sym.reshape(-1)[indices]

    def compute_stress_false(_):
        return jnp.full(6, np.nan)
    
    stress = lax.cond(jnp.equal(cell_rank, 3),
                lambda _: compute_stress_true(stress, volume),
                lambda _: compute_stress_false(stress),
                operand=None)
                
    return local_energies, forces, stress

@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 10, out_axes=0)
def _jax_calc_local_energy_and_derivs(
    r_ijs,
    itype,
    jtypes,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    scaling,
    min_dist,
    max_dist,
    # Static parameters:
    rb_size,
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    energy = _jax_calc_local_energy(
        r_ijs,
        itype,
        jtypes,
        species_coeffs,
        moment_coeffs,
        radial_coeffs,
        scaling,
        min_dist,
        max_dist,
        # Static parameters:
        rb_size,
        basic_moments,
        pair_contractions,
        scalar_contractions,
    )
    derivs = jax.jacobian(_jax_calc_local_energy)(
        r_ijs,
        itype,
        jtypes,
        species_coeffs,
        moment_coeffs,
        radial_coeffs,
        scaling,
        min_dist,
        max_dist,
        # Static parameters:
        rb_size,
        basic_moments,
        pair_contractions,
        scalar_contractions,
    )
    return energy, derivs

@partial(jax.jit, static_argnums=(9, 10, 11, 12))
def _jax_calc_local_energy(
    r_ijs,
    itype,
    jtypes,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    scaling,
    min_dist,
    max_dist,
    # Static parameters:
    rb_size,
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    r_abs = jnp.linalg.norm(r_ijs, axis=1)
    smoothing = jnp.where(r_abs < max_dist, (max_dist - r_abs) ** 2, 0)
    radial_basis = _jax_chebyshev_basis(r_abs, rb_size, min_dist, max_dist)
    
    rb_values = (
        scaling
        * smoothing
        * jnp.einsum("jmn, jn -> mj", radial_coeffs[itype, jtypes], radial_basis)
    )
    basis = _jax_calc_basis(
        r_ijs, r_abs, rb_values, basic_moments, pair_contractions, scalar_contractions
    )
    energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)
    return energy

def _jax_calc_basis(
    r_ijs,
    r_abs,
    rb_values,
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    calculated_moments, nus = _jax_calc_moments(r_ijs, r_abs, rb_values, basic_moments)

    for contraction in pair_contractions:
        m1 = calculated_moments[contraction[0]]
        nu1 = nus[contraction[0]]

        m2 = calculated_moments[contraction[1]]
        nu2 = nus[contraction[1]]

        # Only unpack if needed (nu >= 2)
        if nu1 >= 2:
            m1 = unpack_symmetric_single(m1, nu1, D_FIXED)
        
        if nu2 >= 2:
            m2 = unpack_symmetric_single(m2, nu2, D_FIXED)

        # Use the updated contraction function which handles nu values
        result, result_nu = _jax_contract_over_axes(
            m1, m2, nu1, nu2, contraction[3]
        )
        
        calculated_moments[contraction] = result
        nus[contraction] = result_nu

    basis = []
    for contraction in scalar_contractions:
        b = calculated_moments[contraction]
        basis.append(b)
    
    return jnp.array(basis)

@partial(jax.vmap, in_axes=[0, None, None, None], out_axes=0)
def _jax_chebyshev_basis(r, number_of_terms, min_dist, max_dist):
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    rb = [1, r_scaled]
    for i in range(2, number_of_terms):
        rb.append(2 * r_scaled * rb[i - 1] - rb[i - 2])
    return jnp.array(rb)

def _jax_calc_moments(r_ijs, r_abs, rb_values, moments):
    calculated_moments = {}
    nus = {}
    r_ijs_unit = (r_ijs.T / r_abs).T

    for moment in moments:
        mu = moment[0]
        nu = moment[1]
        m = _jax_make_tensor(r_ijs_unit, nu)
        
        m = (m.T * rb_values[mu]).sum(axis=-1)
        
        # Only pack if nu >= 2
        if nu >= 2:
            m = pack_symmetric_single(m, nu, D_FIXED)        
        
        calculated_moments[moment] = m
        nus[moment] = nu
    
    return calculated_moments, nus

@partial(jax.vmap, in_axes=[0, None], out_axes=0)
def _jax_make_tensor(r, nu):
    m = 1
    for _ in range(nu):
        m = jnp.tensordot(r, m, axes=0)
    
    return m

def _jax_contract_over_axes(m1, m2, nu1, nu2, axes):
    """Updated contraction function that uses nu values explicitly."""
    # Use our custom tensordot that handles symmetric tensors
    result, result_nu = custom_symmetric_tensordot(m1, m2, nu1, nu2, axes)
    return result, result_nu