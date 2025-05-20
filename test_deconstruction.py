import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from itertools import combinations_with_replacement, permutations
from functools import partial
import pickle
import time
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", False)


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
    
def pack_symmetric(tensor):
    """
    tensor: Array of shape (batch, n, n), assumed symmetric along last two axes.
    Returns: Array of shape (batch, K), where K = n*(n+1)//2.
    """
    n = tensor.shape[-1]
    i, j = jnp.triu_indices(n)  
    return tensor[..., i, j]

import itertools

def symmetric_indices(n, nu):
    """
    Compute sorted index-tuples for a fully symmetric tensor of rank nu and dimension n.

    Returns:
      idx: jnp.ndarray of shape (nu, K) where K = C(n + nu - 1, nu).
    """
    # combinations_with_replacement yields sorted tuples of length nu
    comb = itertools.combinations_with_replacement(range(n), nu)
    idx = jnp.array(list(comb)).T  # shape (nu, K)
    return idx


def pack_full_symmetric(tensor, nu):
    """
    Pack a batch of fully symmetric tensors into their unique entries.

    Args:
      tensor: jnp.ndarray of shape (batch, n, ..., n) with nu trailing axes.
      nu: int, rank of the symmetric tensor (number of trailing axes).

    Returns:
      packed: jnp.ndarray of shape (batch, K) where K = C(n + nu - 1, nu).
    """
    # batch dimension
    batch = tensor.shape[0]
    # dimension size
    n = tensor.shape[-1]
    # get unique index tuples
    idx = symmetric_indices(n, nu)  # (nu, K)
    # gather with advanced indexing
    # tensor[:, idx[0], idx[1], ..., idx[nu-1]] -> shape (batch, K)
    packed = tensor[(slice(None),) + tuple(idx)]
    return packed


def unpack_full_symmetric(packed, nu, n=3):
    """
    Reconstruct full symmetric tensors from packed unique entries.

    Args:
      packed: jnp.ndarray of shape (batch, K), as returned by pack_full_symmetric.
      nu: int, rank of the symmetric tensor.
      n: int, dimension size of each axis.

    Returns:
      full: jnp.ndarray of shape (batch, n, ..., n), fully symmetric.
    """
    # batch dimension
    batch = packed.shape[0]
    # get unique index tuples
    idx = symmetric_indices(n, nu)  # (nu, K)
    # prepare output array
    full_shape = (batch,) + (n,) * nu
    full = jnp.zeros(full_shape, dtype=packed.dtype)
    # scatter into all permutations
    perms = list(itertools.permutations(range(nu)))
    for perm in perms:
        perm_idx = tuple(idx[i] for i in perm)
        full = full.at[(slice(None),) + perm_idx].set(packed)
    return full

jax_images = load_data_pickle('training_data/jax_images_data_subset.pkl')
itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, 0)

@jax.jit
def extract_unique_optimized(sym_tensor):
    order = sym_tensor.ndim
    dim = sym_tensor.shape[0]
    unique_indices = jnp.array(list(itertools.combinations_with_replacement(range(dim), order)))
    unique_elements = sym_tensor[tuple(unique_indices.T)]
    return unique_elements, unique_indices


@partial(jax.jit, static_argnums=(2))
def reconstruct_symmetric_tensor_scan(unique_elements, unique_indices, nu):

    shape = tuple([3]*nu)

    def body(tensor, carry):
        idx, val = carry
        perms = jnp.array(list(permutations(idx)))    
        tensor = tensor.at[tuple(perms.T)].set(val)
        return tensor, None

    tensor, _ = lax.scan(body,
                         jnp.zeros(shape),
                         (unique_indices, unique_elements))
    return tensor


r = all_rijs[0]
nus = np.arange(2,8)

insitu_indices_times = []

jax.clear_caches()
print('reset cache')

nus = np.arange(2,10)
counter_insitu = 0
for nu in nus:
    print(f'starting insitu timing: {nu}')
    tensors = _jax_make_tensor(r,nu)
    tensor = tensors[0]

    start_time_total = time.time()

    start_time_el = time.time()
    unique_elements, unique_indices = extract_unique_optimized(tensor)
    end_time_el = time.time()
    elapsed_time_el = end_time_el - start_time_el
    print(elapsed_time_el)

    start_time_rec = time.time()
    reconstructed_tensor = reconstruct_symmetric_tensor_scan(unique_elements, unique_indices, nu)
    end_time_rec = time.time()
    elapsed_time_rec = end_time_rec - start_time_rec
    print(elapsed_time_rec)

    end_time_total = time.time()
    elapsed_time_total = end_time_total - start_time_total
    print(elapsed_time_total)

    insitu_indices_times.append(elapsed_time_total)

    counter_insitu += 1


nus = np.arange(2,10)
insitu_indices_times_cached = []
counter_insitu_cache = 0
for nu in nus:
    print(f'starting cached insitu timing: {nu}')
    tensors = _jax_make_tensor(r,nu)
    tensor = tensors[0]

    start_time_total = time.time()

    start_time_el = time.time()
    unique_elements, unique_indices = extract_unique_optimized(tensor)
    end_time_el = time.time()
    elapsed_time_el = end_time_el - start_time_el
    print(elapsed_time_el)

    start_time_rec = time.time()
    reconstructed_tensor = reconstruct_symmetric_tensor_scan(unique_elements, unique_indices, nu)
    end_time_rec = time.time()
    elapsed_time_rec = end_time_rec - start_time_rec
    print(elapsed_time_rec)

    end_time_total = time.time()
    elapsed_time_total = end_time_total - start_time_total
    print(elapsed_time_total)

    insitu_indices_times_cached.append(elapsed_time_total)

    counter_insitu_cache += 1

print('finished with all timings')
print(insitu_indices_times)
print(insitu_indices_times_cached)


fig, ax = plt.subplots()

ax.plot(nus[0:counter_insitu], insitu_indices_times, 'x-', c='b', label='insitu')

ax.plot(nus[0:counter_insitu_cache], insitu_indices_times_cached, 'x-', c='r', label='insitu cached')

plt.legend()
plt.savefig('deconstruction_timing.pdf')
plt.show()

#%%
import itertools
from itertools import permutations
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from itertools import combinations_with_replacement, permutations
import pickle


jax.config.update("jax_enable_x64", False)


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
nu = 6
tensors = _jax_make_tensor(r,nu)
tensor = tensors[0]
unique_elements, unique_indices = extract_unique_optimized(tensor)

def precompute_permutations(unique_indices):
    nu = unique_indices.shape[1]
    perms = []
    for idx in unique_indices:
        perms.extend(permutations(idx))
    return jnp.array(perms, dtype=jnp.int32) 

@partial(jax.jit, static_argnums=(2,))
def reconstruct_symmetric_tensor_scatter(unique_elements, all_idx, nu):
    shape = (3,) * nu

    total_perms = math.factorial(nu)
    all_vals = jnp.repeat(unique_elements, total_perms)

    tensor = jnp.zeros(shape, dtype=unique_elements.dtype)

    idx_tuple = tuple(all_idx.T.tolist())

    return tensor.at[idx_tuple].set(all_vals)

reconstructed_tensor = reconstruct_symmetric_tensor_scatter(unique_elements, unique_indices, nu)
print(reconstructed_tensor.shape)
