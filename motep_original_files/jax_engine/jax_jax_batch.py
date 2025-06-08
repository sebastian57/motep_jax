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


import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

def calc_energy_forces_stress_batch_optimized(
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
    """
    Batch-optimized version that processes all atoms simultaneously
    instead of using vmap over individual atoms.
    """
    
    local_energies, local_gradient = _jax_calc_local_energy_and_derivs_batch(
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

    # Forces computation (unchanged)
    forces = jnp.array(local_gradient.sum(axis=1))
    forces = jnp.subtract.at(forces, all_js, local_gradient, inplace=False)

    # Stress computation (unchanged)
    stress = jnp.array((all_rijs.transpose((0, 2, 1)) @ local_gradient).sum(axis=0))
    
    def compute_stress_true(stress, volume):
        stress_sym = (stress + stress.T) * 0.5
        stress_sym = stress_sym / volume
        indices = jnp.array([0, 4, 8, 5, 2, 1])
        return stress_sym.reshape(-1)[indices]

    def compute_stress_false(_):
        return jnp.full(6, jnp.nan)
    
    stress = lax.cond(jnp.equal(cell_rank, 3),
                lambda _: compute_stress_true(stress, volume),
                lambda _: compute_stress_false(stress),
                operand=None)
    
    return local_energies, forces, stress


# Use vmap instead of full batch to avoid dynamic indexing issues
@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 10, out_axes=0)
def _jax_calc_local_energy_and_derivs_batch(
    r_ijs,            # (n_neighbors, 3) - per atom
    itype,            # scalar - per atom  
    jtypes,           # (n_neighbors,) - per atom
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
    """
    Optimized version that still uses vmap but with improved tensor operations.
    This avoids the dynamic indexing issue while still providing optimization.
    """
    
    # Energy computation (reuse existing optimized version)
    energy = _jax_calc_local_energy_optimized(
        r_ijs, itype, jtypes, species_coeffs, moment_coeffs,
        radial_coeffs, scaling, min_dist, max_dist, rb_size,
        basic_moments, pair_contractions, scalar_contractions
    )
    
    # Gradient computation
    def energy_fn(r_ijs):
        return _jax_calc_local_energy_optimized(
            r_ijs, itype, jtypes, species_coeffs, moment_coeffs,
            radial_coeffs, scaling, min_dist, max_dist, rb_size,
            basic_moments, pair_contractions, scalar_contractions
        )
    
    gradients = jax.jacobian(energy_fn)(r_ijs)
    
    return energy, gradients


@partial(jax.jit, static_argnums=(9, 10, 11, 12))
def _jax_calc_local_energy_optimized(
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
    """
    Optimized local energy computation with improved tensor operations.
    Uses the symmetric fused tensor sum for better performance.
    """
    r_abs = jnp.linalg.norm(r_ijs, axis=1)
    smoothing = jnp.where(r_abs < max_dist, (max_dist - r_abs) ** 2, 0)
    radial_basis = _jax_chebyshev_basis_vectorized(r_abs, rb_size, min_dist, max_dist)
    
    # This is the fixed version that avoids dynamic indexing
    rb_values = (
        scaling * smoothing[:, None] * 
        jnp.einsum("jmn, jn -> mj", radial_coeffs[itype, jtypes], radial_basis.T)
    )
    
    basis = _jax_calc_basis_symmetric_fused(
        r_ijs, r_abs, rb_values, basic_moments, pair_contractions, scalar_contractions
    )
    
    energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)
    return energy


# Alternative: True batch optimization with static indexing
@partial(jax.jit, static_argnums=(1, 9, 10, 11, 12))
def _jax_calc_local_energy_and_derivs_true_batch(
    all_data,         # Combined data structure to avoid dynamic indexing
    n_species,        # Static: number of species types
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
    """
    True batch optimization using static species indexing.
    Requires preprocessing to avoid dynamic indexing.
    """
    
    r_ijs_batch = all_data['r_ijs']      # (n_atoms, n_neighbors, 3)
    itypes = all_data['itypes']          # (n_atoms,)
    jtypes_batch = all_data['jtypes']    # (n_atoms, n_neighbors)
    
    n_atoms, n_neighbors = r_ijs_batch.shape[:2]
    
    # Process each species type separately to avoid dynamic indexing
    energies = jnp.zeros(n_atoms)
    gradients = jnp.zeros_like(r_ijs_batch)
    
    for species_i in range(n_species):
        # Mask for atoms of this species type
        species_mask = (itypes == species_i)
        
        if not jnp.any(species_mask):
            continue
            
        # Extract data for this species
        r_ijs_species = r_ijs_batch[species_mask]
        jtypes_species = jtypes_batch[species_mask]
        
        # Process this species batch
        species_energies, species_gradients = _process_species_batch(
            r_ijs_species, species_i, jtypes_species,
            species_coeffs, moment_coeffs, radial_coeffs,
            scaling, min_dist, max_dist, rb_size,
            basic_moments, pair_contractions, scalar_contractions
        )
        
        # Update results
        energies = energies.at[species_mask].set(species_energies)
        gradients = gradients.at[species_mask].set(species_gradients)
    
    return energies, gradients


def _process_species_batch(
    r_ijs_batch, species_type, jtypes_batch,
    species_coeffs, moment_coeffs, radial_coeffs,
    scaling, min_dist, max_dist, rb_size,
    basic_moments, pair_contractions, scalar_contractions
):
    """
    Process a batch of atoms of the same species type.
    This avoids dynamic indexing since species_type is known.
    """
    
    n_atoms_species, n_neighbors = r_ijs_batch.shape[:2]
    
    # Vectorized distance computation
    r_abs_batch = jnp.linalg.norm(r_ijs_batch, axis=2)
    smoothing_batch = jnp.where(r_abs_batch < max_dist, (max_dist - r_abs_batch) ** 2, 0)
    
    # Vectorized radial basis
    radial_basis_batch = _jax_chebyshev_basis_vectorized(r_abs_batch, rb_size, min_dist, max_dist)
    
    # Compute rb_values for each atom in the species batch
    rb_values_list = []
    for atom_idx in range(n_atoms_species):
        jtypes_atom = jtypes_batch[atom_idx]
        radial_basis_atom = radial_basis_batch[:, atom_idx, :]  # (rb_size, n_neighbors)
        smoothing_atom = smoothing_batch[atom_idx]  # (n_neighbors,)
        
        rb_values_atom = (
            scaling * smoothing_atom[None, :] * 
            jnp.einsum("mn, n -> m", radial_coeffs[species_type, jtypes_atom], radial_basis_atom.T)
        )
        rb_values_list.append(rb_values_atom)
    
    rb_values_batch = jnp.stack(rb_values_list)  # (n_atoms_species, rb_size, n_neighbors)
    
    # Batch basis computation
    basis_batch = _jax_calc_basis_species_batch(
        r_ijs_batch, r_abs_batch, rb_values_batch,
        basic_moments, pair_contractions, scalar_contractions
    )
    
    # Energy computation
    energies = species_coeffs[species_type] + jnp.einsum('ij,j->i', basis_batch, moment_coeffs)
    
    # Gradient computation
    def energy_fn(r_ijs_batch):
        return _compute_species_batch_energy(
            r_ijs_batch, species_type, jtypes_batch,
            species_coeffs, moment_coeffs, radial_coeffs,
            scaling, min_dist, max_dist, rb_size,
            basic_moments, pair_contractions, scalar_contractions
        )
    
    gradients = jax.jacobian(energy_fn)(r_ijs_batch)
    
    return energies, gradients


def _jax_chebyshev_basis_vectorized(r, number_of_terms, min_dist, max_dist):
    """
    Vectorized Chebyshev basis computation without vmap.
    Works directly on arrays of any shape.
    """
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    rb = [jnp.ones_like(r_scaled), r_scaled]
    for i in range(2, number_of_terms):
        rb.append(2 * r_scaled * rb[i - 1] - rb[i - 2])
    return jnp.stack(rb)  # Shape: (number_of_terms, *r.shape)


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


@partial(jax.jit, static_argnums=(3, 4, 5))
def _jax_calc_basis_batch_optimized(
    r_ijs_batch,        # (n_atoms, n_neighbors, 3)
    r_abs_batch,        # (n_atoms, n_neighbors)
    rb_values_batch,    # (n_radial, n_atoms, n_neighbors)
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    """
    Key optimization: Process all atoms' basis computation simultaneously.
    This eliminates the vmap overhead and enables better vectorization.
    """
    
    # Normalize position vectors: (n_atoms, n_neighbors, 3)
    r_normalized = r_ijs_batch / r_abs_batch[:, :, None]
    
    # Get execution order (computed once, cached by JIT)
    execution_order, dependencies = _flatten_computation_graph(
        basic_moments, pair_contractions, scalar_contractions
    )
    
    # Process all atoms simultaneously
    n_atoms = r_ijs_batch.shape[0]
    results = {}
    
    for op_type, key in execution_order:
        if op_type == 'basic':
            mu, nu, _ = key
            
            # This is the core optimization: batch over all atoms
            # Instead of vmap, we process the entire batch at once
            if nu == 0:
                results[key] = rb_values_batch[mu].sum(axis=1)  # (n_atoms,)
            else:
                # Batch tensor computation for all atoms
                results[key] = _batch_fused_tensor_sum(
                    r_normalized, rb_values_batch[mu], nu
                )
                
        elif op_type == 'contract':
            key_left, key_right, _, (axes_left, axes_right) = key
            left_val = results[key_left]
            right_val = results[key_right]
            
            # Batch contraction over all atoms
            results[key] = jnp.einsum(
                _get_contraction_einsum_string(left_val.shape, right_val.shape, 
                                             axes_left, axes_right),
                left_val, right_val
            )
    
    # Stack final results: (n_atoms, n_scalar_contractions)
    basis_vals = jnp.stack([results[k] for k in scalar_contractions], axis=1)
    return basis_vals


def _batch_general_tensor_sum(r_batch, rb_values_batch, nu):
    """
    General batch tensor computation for arbitrary nu.
    Less efficient than the specialized einsum versions above.
    """
    n_atoms, n_neighbors = r_batch.shape[:2]
    result_shape = (n_atoms,) + (3,) * nu
    result = jnp.zeros(result_shape)
    
    for atom_idx in range(n_atoms):
        for neighbor_idx in range(n_neighbors):
            contribution = rb_values_batch[atom_idx, neighbor_idx] * \
                          _outer_product_recursive(r_batch[atom_idx, neighbor_idx], nu)
            result = result.at[atom_idx].add(contribution)
    
    return result


def _batch_fused_tensor_sum(r, rb_values_mu, nu):
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
        return jnp.einsum('ni,nj,nk,nl,nm,no,n->ijklmn', r, r, r, r, r, r, rb_values_mu)
    elif nu == 7:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,n->ijklmno', r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 8:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,n->ijklmnop', r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 9:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,n->ijklmnopq', r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 10:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,n->ijklmnopqr', r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 11:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,nt,n->ijklmnopqrs', r, r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 12:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,nt,nu,n->ijklmnopqrst', r, r, r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 13:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,nt,nu,nv,n->ijklmnopqrstu', r, r, r, r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 14:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,nt,nu,nv,nw,n->ijklmnopqrstuv', r, r, r, r, r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 15:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,nt,nu,nv,nw,nx,n->ijklmnopqrstuvw', r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 16:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,nt,nu,nv,nw,nx,ny,n->ijklmnopqrstuvwx', r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 17:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,nt,nu,nv,nw,nx,ny,nz,n->ijklmnopqrstuvwxy', r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    elif nu == 18:
        return jnp.einsum('ni,nj,nk,nl,nm,no,np,nq,nr,ns,nt,nu,nv,nw,nx,ny,nz,na,n->ijklmnopqrstuvwxya', r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, r, rb_values_mu)
    else:
        return _batch_general_tensor_sum(r, rb_values_mu, nu)

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

def _batch_general_tensor_sum(r_batch, rb_values_batch, nu):
    n_atoms, n_neighbors = r_batch.shape[:2]
    result_shape = (n_atoms,) + (3,) * nu
    result = jnp.zeros(result_shape)
    
    for atom_idx in range(n_atoms):
        for neighbor_idx in range(n_neighbors):
            contribution = rb_values_batch[atom_idx, neighbor_idx] * \
                          _outer_product_recursive(r_batch[atom_idx, neighbor_idx], nu)
            result = result.at[atom_idx].add(contribution)
    return result


def _get_contraction_einsum_string(left_shape, right_shape, axes_left, axes_right):
    """
    Generate einsum string for batch contractions.
    Handles the batch dimension (first dimension) appropriately.
    """
    # Simplified version - you'd need to implement full einsum string generation
    # based on your specific contraction patterns
    # This is a placeholder that should be customized for your contraction types
    
    # For now, use tensordot which handles batched operations
    return None  # Fallback to tensordot




