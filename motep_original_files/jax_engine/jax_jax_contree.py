from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax import lax
import numpy.typing as npt
from ase import Atoms
import math

from jax.experimental.sparse import BCOO, bcoo_dot_general

jax.config.update("jax_enable_x64", False)

class ContractionNode:
    def __init__(self, key, kind):
        self.key = key
        self.kind = kind       
        self.left = None        
        self.right = None       
        self.axes = None       
        self.result = None     

    def __repr__(self):
        return f"<Node {self.key} ({self.kind})>"

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
    basis = _jax_calc_basis(
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


@partial(jax.vmap, in_axes=[0, None], out_axes=0)
def _jax_make_tensor(r, nu):
    m = 1
    for _ in range(nu):
        m = jnp.tensordot(r, m, axes=0)
        
    return m

#@partial(jax.jit, static_argnums=(2,))
def _jax_contract_over_axes(m1, m2, axes):
    #jax.debug.print('axes: {}', axes)
    calculated_contraction = jnp.tensordot(m1, m2, axes=axes)
    return calculated_contraction


def _create_roots(basic_moments, pair_contractions, scalar_contractions):
    nodes = {}

    for moment_key in basic_moments:
        nodes[moment_key] = ContractionNode(key=moment_key, kind='basic')

    for contraction_key in pair_contractions:
        nodes[contraction_key] = ContractionNode(key=contraction_key, kind='contract')

    for contraction_key in pair_contractions:
        node = nodes[contraction_key]           
        key_left, key_right, _, (axes_left, axes_right) = contraction_key

        node.left = nodes[key_left]             
        node.right = nodes[key_right]           
        node.axes = (axes_left, axes_right)     

    root_keys = set()
    for S in scalar_contractions:
        root_keys.add(S)

    root_nodes = [nodes[k] for k in root_keys]

    return nodes, root_nodes

def _clear_all_results(nodes_dict):
    for node in nodes_dict.values():
        node.result = None

def _evaluate_node(node, r, rb_values):
    if node.result is not None:
        return node.result

    if node.kind == 'basic':
        mu, nu, _ = node.key

        full_tensor = _jax_make_tensor(r, nu)

        small_tensor = (full_tensor.T * rb_values[mu]).sum(axis=-1)

        node.result = small_tensor
        return small_tensor

    if node.kind == 'contract':
        left_val = _evaluate_node(node.left, r, rb_values)
        right_val = _evaluate_node(node.right, r, rb_values)

        axes_left, axes_right = node.axes

        out = _jax_contract_over_axes(left_val, right_val, (axes_left, axes_right))

        node.result = out
        return out

    raise ValueError(f"Unknown node.kind = {node.kind!r} for key = {node.key!r}")

def _compute_basis_for_atom(r, rb_values, nodes_dict, root_nodes):
   
    _clear_all_results(nodes_dict)

    basis_vals = []
    for root in root_nodes:
        scalar_val = _evaluate_node(root, r, rb_values)
        basis_vals.append(scalar_val)
    return jnp.stack(basis_vals) 

#@partial(jax.jit, static_argnums=(3, 4, 5))
def _jax_calc_basis(
    r_ijs,
    r_abs,
    rb_values,
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    r = (r_ijs.T / r_abs).T

    nodes, root_nodes = _create_roots(basic_moments, pair_contractions, scalar_contractions)

    basis = _compute_basis_for_atom(r, rb_values, nodes, root_nodes)
    
    return basis



