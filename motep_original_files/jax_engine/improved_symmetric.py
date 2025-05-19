import numpy as np
import pickle
import os
from itertools import combinations_with_replacement, product
from functools import lru_cache
import jax
import jax.numpy as jnp

D_FIXED = 3
MAX_NU = 14 

def precompute_maps(max_nu=MAX_NU, D=D_FIXED, save_path="tensor_maps.pickle"):
    """
    Precomputes mapping data for symmetric tensors up to max_nu and saves to a pickle file.
    
    Args:
        max_nu: Maximum tensor order to precompute
        D: Dimensionality of the tensors (typically 3 for 3D space)
        save_path: Path to save the pickle file
        
    Returns:
        Dictionary containing the precomputed maps
    """
    maps = {}
    
    # Precompute full_to_compact maps for each nu
    for nu in range(2, max_nu + 1):
        print(f"Precomputing maps for nu={nu}...")
        # Create mapping from full indices to compact indices
        canon_to_idx = {
            idx_tuple: i
            for i, idx_tuple in enumerate(combinations_with_replacement(range(D), nu))
        }
        
        # Create the full mapping
        idx_map = np.empty((D,) * nu, dtype=np.int32)
        for full_idx in product(range(D), repeat=nu):
            canon = tuple(sorted(full_idx))
            idx_map[full_idx] = canon_to_idx[canon]
        
        # Store the unique indices and their compact indices
        unique_indices = list(combinations_with_replacement(range(D), nu))
        
        # Store the maps
        maps[nu] = {
            'full_to_compact': idx_map,
            'unique_indices': unique_indices,
            'compact_size': len(unique_indices)
        }
    
    # Save to pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(maps, f)
    
    print(f"Maps saved to {save_path}")
    return maps

def load_maps(file_path="tensor_maps.pickle"):
    """
    Loads precomputed maps from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the precomputed maps
    """
    if not os.path.exists(file_path):
        print(f"Map file {file_path} not found. Precomputing...")
        return precompute_maps(save_path=file_path)
    
    with open(file_path, 'rb') as f:
        maps = pickle.load(f)
    
    return maps

# Global variable to store the loaded maps
_TENSOR_MAPS = None

def get_maps():
    """
    Get the tensor maps, loading them if necessary.
    """
    global _TENSOR_MAPS
    if _TENSOR_MAPS is None:
        _TENSOR_MAPS = load_maps()
    return _TENSOR_MAPS

def pack_symmetric_single(dense, nu, D=D_FIXED):
    """
    Extract the compact representation of a single symmetric nu-tensor
    using precomputed maps.
    """
    if nu == 0 or nu == 1:
        return dense
    
    maps = get_maps()
    if nu not in maps:
        raise ValueError(f"No precomputed maps for nu={nu}, maximum is {max(maps.keys())}")
    
    # Get the unique indices for this nu
    unique_indices = maps[nu]['unique_indices']
    
    # Extract values at unique indices
    idxs = tuple(jnp.array(cols) for cols in zip(*unique_indices))
    return dense[idxs]

def unpack_symmetric_single(compact, nu, D=D_FIXED):
    """
    Reconstruct the full dense nu-tensor from its compact storage
    using precomputed maps.
    """
    if nu == 0 or nu == 1:
        return compact
    
    maps = get_maps()
    if nu not in maps:
        raise ValueError(f"No precomputed maps for nu={nu}, maximum is {max(maps.keys())}")
    
    # Use the precomputed mapping to reconstruct the full tensor
    idx_map = jnp.array(maps[nu]['full_to_compact'])
    return compact[idx_map]

# Define vectorized versions for batch processing
pack_symmetric = jax.vmap(pack_symmetric_single, in_axes=(0, None, None), out_axes=0)
unpack_symmetric = jax.vmap(unpack_symmetric_single, in_axes=(0, None, None), out_axes=0)

def _jax_contract_over_axes(m1, m2, nu1, nu2, axes, D=D_FIXED):
    """
    Contract tensors efficiently using precomputed maps.
    """
    # Unpack symmetric tensors if needed
    if nu1 >= 2:
        m1 = unpack_symmetric_single(m1, nu1, D)
    if nu2 >= 2:
        m2 = unpack_symmetric_single(m2, nu2, D)
    
    # Perform contraction
    calculated_contraction = jnp.tensordot(m1, m2, axes=axes)
    
    # Calculate new tensor order
    if isinstance(axes, int):
        new_nu = m1.ndim + m2.ndim - 2*axes - 2
    else:
        new_nu = m1.ndim + m2.ndim - 2*len(axes[0]) - 2
    
    new_nu = max(0, new_nu)  # Ensure nu is non-negative
    
    # Pack the result if symmetric
    if new_nu >= 2:
        calculated_contraction = pack_symmetric_single(calculated_contraction, new_nu, D)
    
    return calculated_contraction, new_nu

def custom_symmetric_tensordot(m1, m2, nu1, nu2, axes):
    """
    Improved version of custom_symmetric_tensordot using precomputed maps.
    
    Args:
        m1: First tensor (compact representation)
        m2: Second tensor (compact representation)
        nu1: Order of first tensor
        nu2: Order of second tensor
        axes: Axes to contract over
        
    Returns:
        Contracted tensor and its order
    """
    result, result_nu = _jax_contract_over_axes(m1, m2, nu1, nu2, axes, D_FIXED)
    return result, result_nu

# Example of usage
if __name__ == "__main__":
    # Precompute maps if they don't exist
    if not os.path.exists("tensor_maps.pickle"):
        precompute_maps()
    
    # Load the maps
    maps = load_maps()
    
    print(f"Precomputed maps for nu values: {sorted(maps.keys())}")
    for nu, map_data in sorted(maps.items()):
        print(f"nu={nu}: compact size={map_data['compact_size']}")