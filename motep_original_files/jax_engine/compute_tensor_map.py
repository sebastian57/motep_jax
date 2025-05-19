from functools import partial, lru_cache
from itertools import combinations_with_replacement, product
from math import comb as math_comb
import numpy as np
import pickle
import jax
import jax.numpy as jnp
from typing import List, Tuple # Added for type hinting

D_FIXED = 3

# Corrected function
def get_precomputed_full_to_compact_map(nu: int, D: int = D_FIXED) -> Tuple[jax.Array, List[Tuple[int, ...]]]:
    """Map each full-index tuple to a compact index for a symmetric nu-tensor."""
    if nu < 2:
        raise ValueError(f"Map only for nu >= 2, got nu={nu}")

    # This block is now correctly indented to execute when nu >= 2
    canon_to_idx = {
        idx_tuple: i
        for i, idx_tuple in enumerate(combinations_with_replacement(range(D), nu))
    }

    idx_map_np = np.empty((D,) * nu, dtype=np.int32) # Use a different name before converting to JAX array
    for full_idx in product(range(D), repeat=nu):
        canon = tuple(sorted(full_idx))
        idx_map_np[full_idx] = canon_to_idx[canon]

    unique_indices = list(combinations_with_replacement(range(D), nu))

    return jnp.array(idx_map_np), unique_indices


nus = np.arange(2, 14)
out_dict = {}

# This part of your script remains the same
for nu_val in nus: # Renamed nu to nu_val to avoid conflict with module name if you had one
    filename = f'tensor_maps/{nu_val}_map.pkl'
    idx_map, unique_indices_list = get_precomputed_full_to_compact_map(nu_val, D_FIXED)
    # Using your list structure for out_dict
    out_dict[nu_val] = [idx_map, unique_indices_list]

    with open(filename, 'wb') as f:
        pickle.dump(out_dict, f)
    print(f'Saved map for nu = {nu_val}')

print(f"Successfully created and saved maps in tensor_maps/")
