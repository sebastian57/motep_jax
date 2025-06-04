import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from functools import partial
from itertools import combinations_with_replacement, permutations, product
import pickle
import time

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

pair_contractions = (((0, 0, 0), (0, 0, 0), 0, ((), ())), ((0, 0, 0), (1, 0, 0), 0, ((), ())), ((0, 0, 0), (2, 0, 0), 0, ((), ())), 
                     ((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), ((0, 1, 1), (1, 1, 1), 0, ((0,), (0,))), 
                     ((0, 2, 2), (0, 2, 2), 0, ((0, 1), (0, 1))), ((0, 2, 2), (1, 2, 2), 0, ((0, 1), (0, 1))), 
                     ((0, 3, 3), (0, 3, 3), 0, ((0, 1, 2), (0, 1, 2))), ((0, 4, 4), (0, 4, 4), 0, ((0, 1, 2, 3), (0, 1, 2, 3))), 
                     ((1, 0, 0), (1, 0, 0), 0, ((), ())), ((0, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 
                     ((1, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 
                     ((0, 0, 0), ((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), 0, ((), ())), 
                     ((0, 0, 0), ((0, 1, 1), (1, 1, 1), 0, ((0,), (0,))), 0, ((), ())), 
                     ((0, 0, 0), ((0, 2, 2), (0, 2, 2), 0, ((0, 1), (0, 1))), 0, ((), ())), 
                     ((0, 0, 0), ((0, 3, 3), (0, 3, 3), 0, ((0, 1, 2), (0, 1, 2))), 0, ((), ())), 
                     ((0, 1, 1), (0, 2, 2), 1, ((0,), (0,))), 
                     ((0, 1, 1), ((0, 1, 1), (0, 2, 2), 1, ((0,), (0,))), 0, ((0,), (0,))), 
                     ((1, 0, 0), ((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), 0, ((), ())), 
                     ((0, 2, 2), (0, 3, 3), 1, ((0, 1), (0, 1))), 
                     ((0, 1, 1), ((0, 2, 2), (0, 3, 3), 1, ((0, 1), (0, 1))), 0, ((0,), (0,))), 
                     ((0, 2, 2), (0, 2, 2), 2, ((0,), (0,))), 
                     ((0, 2, 2), ((0, 2, 2), (0, 2, 2), 2, ((0,), (0,))), 0, ((0, 1), (0, 1))), 
                     ((0, 0, 0), ((0, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 0, ((), ())), 
                     ((1, 0, 0), ((0, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 0, ((), ())), 
                     (((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 
                     (((0, 2, 2), (0, 2, 2), 0, ((0, 1), (0, 1))), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 
                     ((0, 0, 0), ((0, 1, 1), ((0, 1, 1), (0, 2, 2), 1, ((0,), (0,))), 0, ((0,), (0,))), 0, ((), ())), 
                     (((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), ((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), 0, ((), ())))
scalar_contractions = ((0, 0, 0), (1, 0, 0), (2, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), ((0, 0, 0), (1, 0, 0), 0, ((), ())), 
                       ((0, 0, 0), (2, 0, 0), 0, ((), ())), ((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), 
                       ((0, 1, 1), (1, 1, 1), 0, ((0,), (0,))), ((0, 2, 2), (0, 2, 2), 0, ((0, 1), (0, 1))), 
                       ((0, 2, 2), (1, 2, 2), 0, ((0, 1), (0, 1))), ((0, 3, 3), (0, 3, 3), 0, ((0, 1, 2), (0, 1, 2))), 
                       ((0, 4, 4), (0, 4, 4), 0, ((0, 1, 2, 3), (0, 1, 2, 3))), 
                       ((1, 0, 0), (1, 0, 0), 0, ((), ())), ((0, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 
                       ((1, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), ((0, 0, 0), ((0, 1, 1), (0, 1, 1), 0, ((0,), 
                       (0,))), 0, ((), ())), ((0, 0, 0), ((0, 1, 1), (1, 1, 1), 0, ((0,), (0,))), 0, ((), ())), 
                       ((0, 0, 0), ((0, 2, 2), (0, 2, 2), 0, ((0, 1), (0, 1))), 0, ((), ())), 
                       ((0, 0, 0), ((0, 3, 3), (0, 3, 3), 0, ((0, 1, 2), (0, 1, 2))), 0, ((), ())), 
                       ((0, 1, 1), ((0, 1, 1), (0, 2, 2), 1, ((0,), (0,))), 0, ((0,), (0,))), 
                       ((1, 0, 0), ((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), 0, ((), ())), ((0, 1, 1), ((0, 2, 2), (0, 3, 3), 1, ((0, 1), 
                       (0, 1))), 0, ((0,), (0,))), ((0, 2, 2), ((0, 2, 2), (0, 2, 2), 2, ((0,), (0,))), 0, ((0, 1), (0, 1))), 
                       ((0, 0, 0), ((0, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 0, ((), ())), 
                       ((1, 0, 0), ((0, 0, 0), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 0, ((), ())), 
                       (((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 
                       (((0, 2, 2), (0, 2, 2), 0, ((0, 1), (0, 1))), ((0, 0, 0), (0, 0, 0), 0, ((), ())), 0, ((), ())), 
                       ((0, 0, 0), ((0, 1, 1), ((0, 1, 1), (0, 2, 2), 1, ((0,), (0,))), 0, ((0,), (0,))), 0, ((), ())), 
                       (((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), ((0, 1, 1), (0, 1, 1), 0, ((0,), (0,))), 0, ((), ())))
basic_moments = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 1), (1, 1, 1), (0, 2, 2), (1, 2, 2), (0, 3, 3), (0, 4, 4))

jax_images = load_data_pickle('motep_jax/training_data/jax_images_data_subset.pkl')
itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, 0)
r = all_rijs

@partial(jax.vmap, in_axes=[0, None], out_axes=0)
def _jax_make_tensor(r, nu):
    m = 1
    for _ in range(nu):
        m = jnp.tensordot(r, m, axes=0)

    return m

def _jax_contract_over_axes(m1, m2, axes):
    #jax.debug.print('axes: {}', axes)
    calculated_contraction = jnp.tensordot(m1, m2, axes=axes)
    return calculated_contraction

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

print("All nodes created:")
for k, n in nodes.items():
    print("  ", n)
print("Roots:", root_nodes)

def clear_all_results(nodes_dict):
    for node in nodes_dict.values():
        node.result = None

def evaluate_node(node, r):
    if node.result is not None:
        return node.result

    if node.kind == 'basic':
        _, nu, _ = node.key

        full_tensor = _jax_make_tensor(r, nu)

        small_tensor = (full_tensor.T).sum(axis=-1)

        node.result = small_tensor
        return small_tensor

    if node.kind == 'contract':
        left_val = evaluate_node(node.left, r)
        right_val = evaluate_node(node.right, r)

        axes_left, axes_right = node.axes

        out = _jax_contract_over_axes(left_val, right_val, (axes_left, axes_right))

        node.result = out
        return out

    raise ValueError(f"Unknown node.kind = {node.kind!r} for key = {node.key!r}")

def compute_basis_for_atom(r, nodes_dict, root_nodes):
   
    clear_all_results(nodes_dict)

    basis_vals = []
    for root in root_nodes:
        scalar_val = evaluate_node(root, r)
        basis_vals.append(scalar_val)
    return jnp.stack(basis_vals) 

@jax.jit
def basis_for_all_atoms(R_all):
    def per_atom_eval(r_single):
        return compute_basis_for_atom(r_single, nodes, root_nodes)
    return jax.vmap(per_atom_eval)(R_all)


basises = basis_for_all_atoms(r)
print(basises)
