import jax
import jax.numpy as jnp
from jax._src.core import Primitive, ShapedArray
from jax._src.interpreters import ad, batching, mlir
from functools import partial, lru_cache
from itertools import combinations_with_replacement, product
from math import comb as math_comb
import numpy as np
import pickle
import os

D_FIXED = 3
###### new funcs ######

def load_maps(file_path="tensor_maps.pkl"):
    
    with open(file_path, 'rb') as f:
        maps = pickle.load(f)
    
    return maps

maps = load_maps(file_path='/home/sebastian/master_thesis/motep_jax_git/motep_jax/motep_original_files/jax_engine/tensor_maps.pkl')

@partial(jax.jit, static_argnums=(1,2,3))
def pack_symmetric_single(dense, nu, D=D_FIXED, maps=maps):
   
    #if nu == 0 or nu == 1:
    #    return dense
    
    unique_indices = maps[nu]['unique_indices']
    
    idxs = tuple(jnp.array(cols) for cols in zip(*unique_indices))
    return dense[idxs]

@partial(jax.jit, static_argnums=(1,2,3))
def unpack_symmetric_single(compact, nu, D=D_FIXED, maps=maps):
   
    #if nu == 0 or nu == 1:
    #    return compact
    
    idx_map = jnp.array(maps[nu]['full_to_compact'])
    #idx_map = maps[nu]['full_to_compact']
    return compact[idx_map]

#################

@lru_cache(maxsize=100)
def get_precomputed_full_to_compact_map(nu: int, D: int = D_FIXED) -> jax.Array:
    """Map each full-index tuple to a compact index for a symmetric nu-tensor."""
    if nu < 2:
        raise ValueError(f"Map only for nu >= 2, got nu={nu}")
    # enumerate canonical combinations
    canon_to_idx = {
        idx_tuple: i
        for i, idx_tuple in enumerate(combinations_with_replacement(range(D), nu))
    }
    idx_map = np.empty((D,) * nu, dtype=np.int32)
    for full_idx in product(range(D), repeat=nu):
        canon = tuple(sorted(full_idx))
        idx_map[full_idx] = canon_to_idx[canon]
    return jnp.array(idx_map)

def pack_symmetric_single_old(dense: jax.Array, nu: int, D: int = D_FIXED) -> jax.Array:
    """Extract the compact representation of a single symmetric nu-tensor."""
    if nu == 0:
        return dense
    if nu == 1:
        return dense
    tuples = list(combinations_with_replacement(range(D), nu))
    idxs = tuple(jnp.array(cols) for cols in zip(*tuples))

    return dense[idxs]

def unpack_symmetric_single_(compact: jax.Array, nu: int, D: int = D_FIXED) -> jax.Array:
    """Reconstruct the full dense nu-tensor from its compact storage."""
    if nu == 0:
        return compact
    if nu == 1:
        return compact
    
    with open('/home/sebastian/master_thesis/motep_jax_git/motep_jax/motep_original_files/jax_engine/tensor_maps.pkl', 'rb') as f:
        maps = pickle.load(f)

    idx_map = maps[nu]

    return compact[idx_map]


def unpack_symmetric_single_old(compact: jax.Array, nu: int, D: int = D_FIXED) -> jax.Array:
    """Reconstruct the full dense nu-tensor from its compact storage."""
    if nu == 0:
        return compact
    if nu == 1:
        return compact
    idx_map = get_precomputed_full_to_compact_map(nu, D)
    return compact[idx_map]

pack_symmetric   = jax.vmap(pack_symmetric_single,   in_axes=(0, None, None), out_axes=0)
unpack_symmetric = jax.vmap(unpack_symmetric_single, in_axes=(0, None, None), out_axes=0)

symmetric_tensordot_p = Primitive("symmetric_tensordot")
symmetric_tensordot_p.multiple_results = False

@symmetric_tensordot_p.def_abstract_eval
def _symmetric_tensordot_abstract_eval(m1_s, m2_s, *, nu1, nu2, D, axes):
    s1 = m1_s.shape if nu1 < 2 else (D,)*nu1
    s2 = m2_s.shape if nu2 < 2 else (D,)*nu2
    def out_shape(a, b, ax):
        if isinstance(ax, int):
            a_axes = tuple(range(len(a)-ax, len(a)))
            b_axes = tuple(range(ax))
        else:
            a_axes, b_axes = ax
        return tuple(a[i] for i in range(len(a)) if i not in a_axes) + \
               tuple(b[i] for i in range(len(b)) if i not in b_axes)
    os = out_shape(s1, s2, axes)
    dt = jnp.promote_types(m1_s.dtype, m2_s.dtype)
    wt = m1_s.weak_type or m2_s.weak_type
    return ShapedArray(os, dt, weak_type=wt)

def symmetric_tensordot_impl(m1_c, m2_c, *, D, axes):

    #m1_full = unpack_symmetric_single(m1_c, nu1, D)
    #m2_full = unpack_symmetric_single(m2_c, nu2, D)

    return jnp.tensordot(m1_c, m2_c, axes=axes)

symmetric_tensordot_p.def_impl(symmetric_tensordot_impl)

def _symmetric_tensordot_xla_lowering(ctx, m1_x, m2_x, *, nu1, nu2, D, axes):
    if nu1 >= 2:
        m1_x = mlir.lower_fun(
            partial(unpack_symmetric_single, nu=nu1, D=D), multiple_results=False
        )(ctx, m1_x)[0]
    if nu2 >= 2:
        m2_x = mlir.lower_fun(
            partial(unpack_symmetric_single, nu=nu2, D=D), multiple_results=False
        )(ctx, m2_x)[0]
    dims = ((),()), (axes[0], axes[1])
    out, = mlir.lower_fun(
        partial(jax.lax.dot_general, dimension_numbers=dims),
        multiple_results=False
    )(ctx, m1_x, m2_x)
    return [out]

mlir.register_lowering(symmetric_tensordot_p, _symmetric_tensordot_xla_lowering)

def _symmetric_tensordot_jvp(primals, tangents, *, nu1, nu2, D, axes):
    m1_c, m2_c = primals
    d1, d2   = tangents
    def f(m1, m2):
        return jnp.tensordot(unpack_symmetric_single(m1, nu1, D),
                             unpack_symmetric_single(m2, nu2, D),
                             axes=axes)
    p, t = jax.jvp(f, (m1_c, m2_c), (d1, d2))
    return p, t

ad.defjvp(symmetric_tensordot_p, _symmetric_tensordot_jvp)

def _symmetric_tensordot_vjp(cot, m1_c, m2_c, *, nu1, nu2, D, axes):
    def f(m1, m2):
        return jnp.tensordot(unpack_symmetric_single(m1, nu1, D),
                             unpack_symmetric_single(m2, nu2, D),
                             axes=axes)
    _, vjp_fun = jax.vjp(f, m1_c, m2_c)
    return vjp_fun(cot)

ad.primitive_transposes[symmetric_tensordot_p] = _symmetric_tensordot_vjp

batching.defvectorized(symmetric_tensordot_p)

def custom_symmetric_tensordot(m1, m2, axes):
    """
    m1_dense, m2_dense: shape (batch, D, D, ..., D) with nu+1 dims
    axes: a pair of tuples specifying which tensor dimensions to contract.
    """
    D_val = D_FIXED

    def call_prim(a, b):
        return symmetric_tensordot_p.bind(a, b, D=D_val, axes=axes)

    result =  call_prim(m1, m2) 

    print('result shape')
    print(result.shape)

    return result, result.ndim - 1





