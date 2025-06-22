import os
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np

from jax import lax
import numpy.typing as npt
from ase import Atoms
import math
from typing import Dict, List, Tuple, Any
import dataclasses

import string
import operator

jax.config.update("jax_enable_x64", False)

def get_types(atoms: Atoms, species: list[int] = None) -> npt.NDArray[np.int64]:
    if species is None:
        return np.array(atoms.numbers, dtype=int)
    else:
        return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=int)


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
    basic_moments,
    pair_contractions,
    scalar_contractions,
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
        basic_moments,
        pair_contractions,
        scalar_contractions,
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

@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 12, out_axes=0)
@partial(jax.jit, static_argnums=(9,10,11,12,13,14))
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
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    
        
    execution_order, _ = _flatten_computation_graph(
        basic_moments, pair_contractions, scalar_contractions
    )
    
    execution_order = tuple(execution_order)
    
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
    #smoothing = jnp.where(r_abs < max_dist, (max_dist - r_abs) ** 2, 0)
    #radial_basis = _jax_chebyshev_basis(r_abs, rb_size, min_dist, max_dist) 
    # j, rb_size
    #rb_values = (
    #    scaling
    #    * smoothing
    #    * jnp.einsum("jmn, jn -> mj", radial_coeffs[itype, jtypes], radial_basis)
    #)
    
    radial_basis = _jax_chebyshev_basis(r_abs, rb_size, min_dist, max_dist)
    smoothing = jnp.where(r_abs < max_dist, (max_dist - r_abs) ** 2, 0)
    scaled_smoothing = scaling * smoothing
    
    coeffs = radial_coeffs[itype, jtypes]  
    contracted = jnp.sum(coeffs * radial_basis[:, None, :], axis=-1) 
    rb_values = scaled_smoothing[:, None] * contracted  
    rb_values = rb_values.T  

    
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
    
    
def _compute_tensor_chunk(rb, r, nu):
    """Compute tensor sum for a chunk of neighbors."""
    if nu == 0:
        return jnp.sum(rb)
    elif nu == 1:
        return jnp.sum(rb[:, None] * r, axis=0)
    elif nu == 2:
        return jnp.einsum('i,ij,ik->jk', rb, r, r)
    elif nu == 3:
        return jnp.einsum('i,ij,ik,il->jkl', rb, r, r, r)
    else:
        operands = [rb] + [r] * nu
        letters = string.ascii_lowercase[:nu]
        input_subs = ['i'] + [f'i{l}' for l in letters]
        output_str = ''.join(letters)
        input_str = ','.join(input_subs)
        return jnp.einsum(f'{input_str}->{output_str}', *operands)

def _safe_tensor_sum_(r, rb_values_mu, nu):
    CHUNK_SIZE = 1000
    n = r.shape[0]
    if nu <= 3 or n <= CHUNK_SIZE:
        return _compute_tensor_chunk(rb_values_mu, r, nu)
    
   
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    init_val = jnp.zeros((3,) * nu) if nu > 0 else 0.0

    def body(i, carry):
        start = i * CHUNK_SIZE
        chunk_r = r[start:start + CHUNK_SIZE]
        chunk_rb = rb_values_mu[start:start + CHUNK_SIZE]
        chunk_val = _compute_tensor_chunk(chunk_rb, chunk_r, nu)
        return carry + chunk_val

    return lax.fori_loop(0, n_chunks, body, init_val)
    
    
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

def _jax_contract_over_axes(m1, m2, axes):
    return jnp.tensordot(m1, m2, axes=axes)

def _flatten_computation_graph(basic_moments, pair_contractions, scalar_contractions):
    execution_order = []
    dependencies = {}
    
    for moment_key in basic_moments:
        execution_order.append(('basic', moment_key))
        dependencies[moment_key] = []
    
    remaining_contractions = list(pair_contractions)
    while remaining_contractions:
        for i, contraction_key in enumerate(remaining_contractions):
            key_left, key_right, _, axes = contraction_key
            if key_left in dependencies and key_right in dependencies:
                execution_order.append(('contract', contraction_key))
                dependencies[contraction_key] = [key_left, key_right]
                remaining_contractions.pop(i)
                break
        else:
            raise ValueError("Circular dependency in contraction graph")
    
    return execution_order, dependencies


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
        if op_type == 'basic':
            mu, nu, _ = key
            
            results[key] = _safe_tensor_sum(r, rb_values[mu], nu)
            
        elif op_type == 'contract':
            key_left, key_right, _, (axes_left, axes_right) = key
            left_val = results[key_left]
            right_val = results[key_right]
            results[key] = _jax_contract_over_axes(left_val, right_val, (axes_left, axes_right))
    
    basis_vals = [results[k] for k in scalar_contractions]
    return jnp.stack(basis_vals)

