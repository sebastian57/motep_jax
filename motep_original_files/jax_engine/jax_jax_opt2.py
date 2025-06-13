from functools import partial

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

    # changed from np. to jnp.
    forces = jnp.array(local_gradient.sum(axis=1))
    forces = jnp.subtract.at(forces, all_js, local_gradient, inplace=False)
    #jnp.subtract.at(forces, all_js, local_gradient)

    # fix this later. Must be possible with just the information itself
    stress = jnp.array((all_rijs.transpose((0, 2, 1)) @ local_gradient).sum(axis=0))
    
    
    # does not work because of if condition
    ###############################################
    # ijk @ ikj -> ikk -> kk
    #if cell_rank == 3:
    #    stress = (stress + stress.T) * 0.5  # symmetrize
    #    stress /= volume
    #    # new
    #    indices = jnp.array([0, 4, 8, 5, 2, 1])
    #    stress = stress.reshape(-1)[indices]
    #    #stress = stress.flat[[0, 4, 8, 5, 2, 1]] # was stress.flat[[...]]
    #else:
    #    stress = jnp.full(6, np.nan)
    ##############################################  
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
    ##############################################  
    return local_energies, forces, stress


#@partial(jax.jit, static_argnums=(9, 10, 11, 12))
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
    radial_basis = _jax_chebyshev_basis(r_abs, rb_size, min_dist, max_dist) # why do I get an error here?
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


#@partial(jax.jit, static_argnums=[1, 2, 3])
@partial(jax.vmap, in_axes=[0, None, None, None], out_axes=0)
def _jax_chebyshev_basis(r, number_of_terms, min_dist, max_dist):
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    rb = [1, r_scaled]
    for i in range(2, number_of_terms):
        rb.append(2 * r_scaled * rb[i - 1] - rb[i - 2])
    #print(rb)
    return jnp.array(rb)

@dataclasses.dataclass
class ContractionNode:
    key: Any
    kind: str
    left: 'ContractionNode' = None
    right: 'ContractionNode' = None
    axes: Tuple = None
    result: Any = None

def _generate_symmetric_indices(nu):
    """Generate unique symmetric tensor indices and their multiplicities."""
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
    
    # For each unique symmetric pattern
    for (i, j, k), multiplicity in sym_indices_mults:
        if i + j + k != nu:
            continue
            
        # Compute the contribution from this symmetric pattern
        # This is equivalent to r[:, 0]^i * r[:, 1]^j * r[:, 2]^k weighted by rb_values
        contribution = (r[:, 0]**i * r[:, 1]**j * r[:, 2]**k * rb_values_mu).sum()
        
        # Fill all symmetric positions with this contribution
        # For efficiency, we'll use a different approach that directly computes the tensor
        
    # Alternative: Direct computation without explicit enumeration
    # This is more efficient for JAX compilation
    return _fused_tensor_sum(r, rb_values_mu, nu)

def _fused_tensor_sum(r, rb_values_mu, nu):
    """
    Fused computation that directly computes sum(r_outer_product^nu * rb_values).
    This avoids creating the full tensor and leverages JAX's optimization.
    """
    if nu == 0:
        return rb_values_mu.sum()
    elif nu == 1:
        return jnp.einsum('ni,n->i', r, rb_values_mu)
    elif nu == 2:
        return jnp.einsum('ni,nj,n->ij', r, r, rb_values_mu)
    elif nu == 3:
        return jnp.einsum('ni,nj,nk,n->ijk', r, r, r, rb_values_mu)
    elif nu == 4:
        return jnp.einsum('ni,nj,nk,nl,n->ijkl', r, r, r, r, rb_values_mu)
    elif nu == 5:
        return jnp.einsum('ni,nj,nk,nl,nm,n->ijklm', r, r, r, r, r, rb_values_mu)
    elif nu == 6:
        return jnp.einsum('ni,nj,nk,nl,nm,no,n->ijklmo', r, r, r, r, r, r, rb_values_mu)
    else:
        return _general_tensor_sum(r, rb_values_mu, nu)

def _general_tensor_sum(r, rb_values_mu, nu):
    """General fallback for arbitrary tensor ranks."""
    result = rb_values_mu[0] * _outer_product_recursive(r[0], nu)
    
    for i in range(1, r.shape[0]):
        result += rb_values_mu[i] * _outer_product_recursive(r[i], nu)
    
    return result

def _outer_product_recursive(vec, nu):
    """Compute outer product of vector with itself nu times."""
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
    """Tensor contraction optimized for symmetric tensors."""
    return jnp.tensordot(m1, m2, axes=axes)

def _flatten_computation_graph(basic_moments, pair_contractions, scalar_contractions):
    """Pre-compute execution order for optimal graph traversal."""
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

# Main optimized basis computation
def _jax_calc_basis_symmetric_fused(
    r_ijs,
    r_abs,
    rb_values,
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions,
):

    # Normalize r vectors (same as before)
    r = (r_ijs.T / r_abs).T
    
    # Get execution order
    execution_order, dependencies = _flatten_computation_graph(
        basic_moments, pair_contractions, scalar_contractions
    )
    
    # Results dictionary for intermediate values
    results = {}
    
    # Execute computation graph
    for op_type, key in execution_order:
        if op_type == 'basic':
            mu, nu, _ = key
            
            # FUSED OPERATION: This replaces the separate tensor creation, 
            # transpose, multiplication, and summation with a single fused operation
            results[key] = _fused_tensor_sum(r, rb_values[mu], nu)
            
        elif op_type == 'contract':
            key_left, key_right, _, (axes_left, axes_right) = key
            left_val = results[key_left]
            right_val = results[key_right]
            results[key] = _jax_contract_over_axes(left_val, right_val, (axes_left, axes_right))
    
    # Collect final results
    basis_vals = [results[k] for k in scalar_contractions]
    return jnp.stack(basis_vals)