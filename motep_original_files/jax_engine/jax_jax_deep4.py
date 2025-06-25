import os
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=4'

from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np

from jax import lax, checkpoint, dtypes
import numpy.typing as npt
from ase import Atoms
import math
from typing import Dict, List, Tuple, Any
import dataclasses

import string
import operator
from collections import defaultdict


# have two different versions in here
# normal one
# vectorized one
# also have mixed precision activated
# makes it slower, but should be more memory efficient
# still need to decide which approach (normal or vect) is actually better
# do a training timing run for this


def get_types(atoms: Atoms, species: list[int] = None) -> npt.NDArray[np.int64]:
    if species is None:
        return np.array(atoms.numbers, dtype=int)
    else:
        return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=int)


@partial(jax.jit, static_argnames=('execution_order', 'scalar_contractions'))
def calc_energy_forces_stress(
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
    execution_order,
    scalar_contractions
):
    
    local_energies, forces = _jax_calc_local_energy_and_derivs(
        all_rijs,
        itypes,
        all_jtypes,
        species_coeffs,
        moment_coeffs,
        radial_coeffs,
        scaling,
        min_dist,
        max_dist,
        itypes.shape,
        len(itypes),
        # Static parameters:
        radial_coeffs.shape[3],
        execution_order,
        scalar_contractions
    )
    
        
    stress = jnp.array((all_rijs.transpose((0, 2, 1)) @ forces).sum(axis=0))
    
    def compute_stress_true(stress, volume):
        stress_sym = (stress + stress.T) * 0.5 / volume
        indices = jnp.array([0, 4, 8, 5, 2, 1])
        return stress_sym.reshape(-1)[indices]


    def compute_stress_false(_):
        return jnp.full(6, jnp.nan)
    
    stress_voigt = lax.cond(
        jnp.equal(cell_rank, 3),
        lambda _: compute_stress_true(stress, volume),
        lambda _: compute_stress_false(stress),
        operand=None
    )

    forces = jnp.sum(forces, axis=-2)
    
    return local_energies, forces, stress_voigt



@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 11, out_axes=0)
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
    itypes_shape,
    itypes_len,
    # Static parameters:
    rb_size,
    execution_order,
    scalar_contractions,
):
        
    def energy_fn(r_ijs):
        return _jax_calc_local_energy(
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
            execution_order,
            scalar_contractions,
        ).sum()  

    total_energy, forces = jax.value_and_grad(energy_fn)(r_ijs)
     
    local_energies = jnp.full(itypes_shape, total_energy / itypes_len)
    
    return local_energies, forces


@partial(jax.jit, static_argnums=(9, 10, 11))
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
    execution_order,
    scalar_contractions
):
    r_abs = jnp.linalg.norm(r_ijs, axis=1)

    radial_basis = _jax_chebyshev_basis(r_abs, rb_size, min_dist, max_dist) 
    smoothing = jnp.where(r_abs < max_dist, (max_dist - r_abs) ** 2, 0)      
    scaled_smoothing = scaling * smoothing                                   

    coeffs = radial_coeffs[itype, jtypes] 
    rb_values = jnp.einsum(
        'j, jmn, jn -> mj',
        scaled_smoothing,
        coeffs,
        radial_basis
    )

    basis = _jax_calc_basis_symmetric_fused(
        r_ijs, r_abs, rb_values, execution_order, scalar_contractions
    )
    energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)
    return energy



def _jax_chebyshev_basis(r, n_terms, min_dist, max_dist):
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    T = jnp.zeros((n_terms, r.shape[0]))
    T = T.at[0].set(1.0)
    if n_terms > 1:
        T = T.at[1].set(r_scaled)
    for i in range(2, n_terms):
        T = T.at[i].set(2 * r_scaled * T[i-1] - T[i-2])
    return T.T
    
    
def _safe_tensor_sum(r, rb_values_mu, nu):
    if nu == 0:
        return jnp.sum(rb_values_mu)
    elif nu == 1:
        return jnp.dot(rb_values_mu, r)  
    elif nu == 2:
        return jnp.einsum('i,ij,ik->jk', rb_values_mu, r, r)  
    elif nu == 3:
        return jnp.einsum('i,ij,ik,il->jkl', rb_values_mu, r, r, r)  
    else:
        operands = [rb_values_mu] + [r] * nu
        letters = string.ascii_lowercase[:nu]
        input_subs = ['i'] + [f'i{l}' for l in letters]
        return jnp.einsum(
            f'{",".join(input_subs)}->{"".join(letters)}', 
            *operands
        )

def _jax_calc_basis_symmetric_fused(
    r_ijs,
    r_abs,
    rb_values,
    # Static parameters:
    execution_order,
    scalar_contractions
):
    r = (r_ijs.T / r_abs).T

    results = {}
    
    for op_type, key in execution_order:
        str_key = str(key)
        
        if op_type == 'basic':
            mu, nu, _ = key
            results[str_key] = _safe_tensor_sum(r, rb_values[mu], nu)
            
        elif op_type == 'contract':
            key_left, key_right, _, (axes_left, axes_right) = key
            
            left_val = results[str(key_left)]
            right_val = results[str(key_right)]
            
            results[str_key] = _jax_contract_over_axes(left_val, right_val, (axes_left, axes_right))
    
    basis_vals = [results[str(k)] for k in scalar_contractions]
    return jnp.stack(basis_vals)

