import itertools
from itertools import permutations
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from itertools import combinations_with_replacement, permutations
import pickle
import time
from jax import lax
from jax import pmap


jax.config.update("jax_enable_x64", False)
jax.clear_caches()



@partial(jax.vmap, in_axes=[0, None], out_axes=0)
def _jax_make_tensor(r, nu):
    m = 1
    for _ in range(nu):
        m = jnp.tensordot(r, m, axes=0)

    return m

def get_data_for_indices(jax_images, index):
    '''
    Extracts pre-computed data for a specific configuration index from the JAX dataset.

    Retrieves the JAX arrays corresponding to a single atomic configuration
    (specified by `index`) from the larger `jax_images` dictionary, which
    contains the pre-processed data for the entire dataset.

    :param jax_images: Dictionary containing JAX arrays for the entire dataset
                       (e.g., 'itypes', 'all_js', 'all_rijs', 'E', 'F', 'sigma', etc.).
    :param index: The integer index of the desired configuration/image.
    :return: Tuple containing the data arrays for the specified configuration:
             (itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma).
    '''
    itypes = jax_images['itypes'][index]
    all_js = jax_images['all_js'][index]
    all_rijs = jax_images['all_rijs'][index]
    all_jtypes = jax_images['all_jtypes'][index]
    cell_rank = jax_images['cell_ranks'][index]
    volume = jax_images['volumes'][index]
    E = jax_images['E'][index]
    F = jax_images['F'][index]
    sigma = jax_images['sigma'][index]

    return itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma

def load_data_pickle(filename):
    '''
    Loads Python data from a pickle file.

    :param filename: The path to the pickle file to load.
    :return: The Python object loaded from the file.
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
@jax.jit
def extract_unique_optimized(sym_tensor):
    order = sym_tensor.ndim
    dim = sym_tensor.shape[0]
    unique_indices = jnp.array(list(itertools.combinations_with_replacement(range(dim), order)))
    unique_elements = sym_tensor[tuple(unique_indices.T)]
    return unique_elements, unique_indices
    

jax_images = load_data_pickle('training_data/jax_images_data_subset.pkl')
itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, 0)
r = all_rijs[0]
nu = 3
tensors = _jax_make_tensor(r,nu)
tensor = tensors[0]
shape = tensor.shape
unique_elements, unique_indices = extract_unique_optimized(tensor)
all_perms = jnp.stack([
    jnp.array(list(itertools.permutations(idx)))
    for idx in unique_indices  
]) 
    

@partial(jax.jit, static_argnums=(2,))
def reconstruct_with_static_perms(vals, perms, order, dim=3):
    zeros = jnp.zeros((dim,)*order)
    flat_idx = perms.reshape(-1, order)
    flat_vals = jnp.repeat(vals, perms.shape[1])
    return zeros.at[tuple(flat_idx.T)].set(flat_vals)



scatter_one = lambda p, v: jnp.zeros(shape).at[tuple(p.T)].set(v)
batched_scatter = jax.vmap(scatter_one, in_axes=(0, 0))
@jax.jit
def rec_vmap(vals, idxs):
    per_tensor = batched_scatter(idxs, vals) 
    return per_tensor.sum(0)


start_time = time.time()
reconstructed_with_perms = reconstruct_with_static_perms(unique_elements, all_perms, nu)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
print(reconstructed_with_perms.shape)
print(reconstructed_with_perms)

start_time = time.time()
reconstructed_with_vmap = rec_vmap(unique_elements, unique_indices)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
print(reconstructed_with_vmap.shape)
print(reconstructed_with_vmap)

print(jnp.sum(reconstructed_with_perms) / jnp.sum(reconstructed_with_vmap))


perms = jnp.stack([
    jnp.array(list(permutations(idx)))
    for idx in unique_indices    
])

n_perm = perms.shape[1]

SHAPE = (3,)*nu

@partial(jax.pmap, axis_name='i')
def rec_pmap(vals, idxs):
    zeros = jnp.zeros(SHAPE, dtype=vals.dtype)
    flat_idx  = idxs.reshape(-1, idxs.shape[-1])
    flat_vals = jnp.repeat(vals, idxs.shape[1])
    out = zeros.at[tuple(flat_idx.T)].set(flat_vals)
    return lax.psum(out, axis_name='i')


n_dev = jax.local_device_count()
batch_per_dev = unique_elements.shape[0] // n_dev


vals_sharded = unique_elements.reshape(n_dev, batch_per_dev)
perms_sharded = perms.reshape(n_dev, batch_per_dev, n_perm, nu)


start_time = time.time()
out = rec_pmap(vals_sharded, perms_sharded)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

print(out[0].shape)
print(out[0])

