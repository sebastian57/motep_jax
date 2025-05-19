from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax import lax
import numpy.typing as npt
from ase import Atoms
import math

from jax.experimental import sparse
from jax.experimental.sparse import BCOO, bcoo_dot_general, bcoo_transpose, coo_matvec

import pickle
import itertools

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

    #maps = load_maps(file_path='/home/sebastian/master_thesis/motep_jax_git/motep_jax/motep_original_files/jax_engine/tensor_maps.pkl')

    basis = _jax_calc_basis(
        r_ijs, r_abs, rb_values, basic_moments, pair_contractions, scalar_contractions
    )
    energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)
    return energy


@partial(jax.jit, static_argnums=(3, 4, 5))
def _jax_calc_basis(
    r_ijs,
    r_abs,
    rb_values,
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions
):
    calculated_moments, nus = _jax_calc_moments(r_ijs, r_abs, rb_values, basic_moments)

    for contraction in pair_contractions:

        m1 = calculated_moments[contraction[0]]
        nu1 = nus[contraction[0]]

        m2 = calculated_moments[contraction[1]]
        nu2 = nus[contraction[1]]

        if nu1 >= 2:
            m1 = unpack_symmetric_single(m1,nu1)

        if nu2 >= 2:
            m2 = unpack_symmetric_single(m2,nu2)

        calculated_moments[contraction], nus[contraction] = _jax_contract_over_axes(
            m1, m2, contraction[3]
        )

    basis = []
    for contraction in scalar_contractions:
        b = calculated_moments[contraction]
        basis.append(b)
        #print(b)
    
    return jnp.array(basis)


#@partial(jax.jit, static_argnums=[1, 2, 3])
@partial(jax.vmap, in_axes=[0, None, None, None], out_axes=0)
def _jax_chebyshev_basis(r, number_of_terms, min_dist, max_dist):
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    rb = [1, r_scaled]
    for i in range(2, number_of_terms):
        rb.append(2 * r_scaled * rb[i - 1] - rb[i - 2])
    #print(rb)
    return jnp.array(rb)


#@partial(jax.jit, static_argnums=(3,))
def _jax_calc_moments(r_ijs, r_abs, rb_values, moments):
    
    calculated_moments = {}
    nus = {}
    r_ijs_unit = (r_ijs.T / r_abs).T

    for moment in moments:
        mu = moment[0]
        nu = moment[1]
        #jax.debug.print('nu: {}', nu)
        m = _jax_make_tensor(r_ijs_unit, nu)
        
        m = (m.T * rb_values[mu]).sum(axis=-1)

        if nu >= 2: 
            m = pack_symmetric_single(m, nu)      
          
        calculated_moments[moment] = m
        nus[moment] = nu
    
    return calculated_moments, nus



@partial(jax.vmap, in_axes=[0, None], out_axes=0)
def _jax_make_tensor(r, nu):
    m = 1
    for _ in range(nu):
        m = jnp.tensordot(r, m, axes=0)
    
    return m


def _jax_contract_over_axes(m1, m2, axes):

    calculated_contraction = jnp.tensordot(m1, m2, axes=axes)

    if isinstance(axes, int):
        nu = m1.ndim + m2.ndim - 2*axes - 2
    else:
        nu = m1.ndim + m2.ndim - 2*len(axes[0]) - 2

    if nu >= 2:
        calculated_contraction = pack_symmetric_single(calculated_contraction, nu) 

    return calculated_contraction, nu

def load_maps(file_path="tensor_maps.pkl"):
        with open(file_path, 'rb') as f:
            maps = pickle.load(f)
        return maps


@partial(jax.jit, static_argnums=(1))
def pack_symmetric_single(dense, nu):
   
    map = load_maps(file_path=f'/home/sebastian/master_thesis/motep_jax_git/motep_jax/motep_original_files/jax_engine/tensor_maps/{nu}_map.pkl')
    unique_indices = map[nu][1]
    
    idxs = tuple(jnp.array(cols) for cols in zip(*unique_indices))
    return dense[idxs]

@partial(jax.jit, static_argnums=(1))
def unpack_symmetric_single(compact, nu):

    map = load_maps(file_path=f'/home/sebastian/master_thesis/motep_jax_git/motep_jax/motep_original_files/jax_engine/tensor_maps/{nu}_map.pkl')

    idx_map = jnp.array(map[nu][0])
    return compact[idx_map]
