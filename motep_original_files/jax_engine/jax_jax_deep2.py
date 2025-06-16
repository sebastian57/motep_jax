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

jax.config.update("jax_enable_x64", False)

def get_types(atoms: Atoms, species: list[int] = None) -> npt.NDArray[np.int64]:
    if species is None:
        return np.array(atoms.numbers, dtype=int)
    else:
        return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=int)


# @partial(jax.jit, static_argnums=(9, 10, 11, 12))
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
    
    return local_energies, forces, stress_voigt

@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 12, out_axes=0)
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
            basic_moments,
            pair_contractions,
            scalar_contractions,
        ).sum()  

    total_energy, forces = jax.value_and_grad(energy_fn)(r_ijs)

    local_energies = jnp.full(itypes_shape, total_energy / itypes_len)
    
    return local_energies, forces

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
    # j, rb_size
    rb_values = (
        scaling
        * smoothing
        * jnp.einsum("jmn, jn -> mj", radial_coeffs[itype, jtypes], radial_basis)
    )
    basis = _jax_calc_basis_symmetric_fused(
        r_ijs, r_abs, rb_values, basic_moments, pair_contractions, scalar_contractions
    )
    energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)
    return energy

def _jax_chebyshev_basis(r, n_terms, min_dist, max_dist):
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    # Precompute T0 and T1 for all r
    T = jnp.zeros((n_terms, r.shape[0]))
    T = T.at[0].set(1.0)
    if n_terms > 1:
        T = T.at[1].set(r_scaled)
    # Vectorized recurrence for n>=2
    for i in range(2, n_terms):
        T = T.at[i].set(2 * r_scaled * T[i-1] - T[i-2])
    return T.T


@dataclasses.dataclass
class ContractionNode:
    key: Any
    kind: str
    left: 'ContractionNode' = None
    right: 'ContractionNode' = None
    axes: Tuple = None
    result: Any = None

def _generate_symmetric_indices(nu):
    if nu == 0:
        return [((), 1)]
    
    indices = []
    multiplicities = []
    
    for i in range(nu + 1):
        for j in range(nu + 1 - i):
            k = nu - i - j
            if k >= 0:
                from math import factorial
                mult = factorial(nu) // (factorial(i) * factorial(j) * factorial(k))
                indices.append((i, j, k))
                multiplicities.append(mult)
    
    return list(zip(indices, multiplicities))

def _compute_symmetric_weighted_sum(r, rb_values_mu, nu):
    if nu == 0:
        return rb_values_mu.sum()
    
    n_neighbors = r.shape[0]
    
    sym_indices_mults = _generate_symmetric_indices(nu)
    
    result_shape = (3,) * nu
    result = jnp.zeros(result_shape)
    
    for (i, j, k), multiplicity in sym_indices_mults:
        if i + j + k != nu:
            continue
        contribution = (r[:, 0]**i * r[:, 1]**j * r[:, 2]**k * rb_values_mu).sum()
        
    return _safe_tensor_sum(r, rb_values_mu, nu)

def _safe_tensor_sum(r, rb_values_mu, nu):
    if nu == 0:
        return rb_values_mu.sum()
    
    einsum_str = {
        1: "ni,n->i",
        2: "ni,nj,n->ij",
        3: "ni,nj,nk,n->ijk"
    }.get(nu, None)
    
    if einsum_str:
        return jnp.einsum(einsum_str, r, r, rb_values_mu, 
                          optimize='optimal', 
                          preferred_element_type=jnp.float32)
    
    return jnp.tensordot(
        reduce(jnp.multiply, [r] * nu),
        rb_values_mu,
        axes=([0], [0])
    )

def _vector_outer_product(vec, nu):
    if nu == 0:
        return jnp.array(1.0)
    
    result = vec
    
    for _ in range(1, nu):
        result = jnp.expand_dims(result, axis=0) * jnp.expand_dims(vec, axis=tuple(range(1, result.ndim + 1)))
    
    return result

def _outer_product_recursive(vec, nu):
    if nu == 0:
        return jnp.array(1.0)
    elif nu == 1:
        return vec
    else:
        m = vec
        for _ in range(nu - 1):
            m = jnp.tensordot(vec, m, axes=0)
        return m

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
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    r = (r_ijs.T / r_abs).T
    
    execution_order, dependencies = _flatten_computation_graph(
        basic_moments, pair_contractions, scalar_contractions
    )
    
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

