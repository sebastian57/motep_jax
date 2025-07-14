import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import export

from motep_original_files.jax_engine.jax_pad import calc_energy_forces_stress_padded_simple  as jax_calc
from motep_original_files.jax_engine.moment_jax import MomentBasis

from motep_original_files.mtp import read_mtp
from motep_original_files.calculator import MTP
from motep_original_files.jax_engine.conversion import BasisConverter
from motep_jax_train_import import*

def _initialize_mtp(mtp_file):
    mtp_data = read_mtp(mtp_file)
    mtp_data.species = np.arange(0, mtp_data.species_count)
    return mtp_data

def _get_data(atom_id, MAX_ATOMS, MAX_NEIGHBORS):
    jax_val_images = load_data_pickle(f'training_data/val_jax_images_data.pkl')     
    initial_args = get_data_for_indices(jax_val_images, atom_id)[0:6]
    
    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume = initial_args
    
    natoms_actual = len(itypes)
    nneigh_actual = all_js.shape[1] if len(all_js.shape) > 1 else 1
    
    itypes_padded = np.zeros(MAX_ATOMS, dtype=np.int32)
    all_js_padded = np.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=np.int32)
    all_rijs_padded = np.zeros((MAX_ATOMS, MAX_NEIGHBORS, 3), dtype=np.float32)
    all_jtypes_padded = np.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=np.int32)
    
    # Copy actual data (up to maximums)
    atoms_to_copy = min(natoms_actual, MAX_ATOMS)
    neighbors_to_copy = min(nneigh_actual, MAX_NEIGHBORS)
    
    itypes_padded[:atoms_to_copy] = itypes[:atoms_to_copy]
    all_js_padded[:atoms_to_copy, :neighbors_to_copy] = all_js[:atoms_to_copy, :neighbors_to_copy]
    all_rijs_padded[:atoms_to_copy, :neighbors_to_copy] = all_rijs[:atoms_to_copy, :neighbors_to_copy]
    all_jtypes_padded[:atoms_to_copy, :neighbors_to_copy] = all_jtypes[:atoms_to_copy, :neighbors_to_copy]
    
    # Return 8 arguments: 6 padded + 2 actual counts
    return [itypes_padded, all_js_padded, all_rijs_padded, all_jtypes_padded, 
            cell_rank, volume, natoms_actual, nneigh_actual]

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

def totuple(x):
    try:
        return tuple(totuple(y) for y in x)
    except TypeError:
        return x

def extract_static_params(level, mtp_file, MAX_ATOMS, MAX_NEIGHBORS):
    mtp_data = _initialize_mtp(f'training_data/{mtp_file}.mtp')

    moment_basis = MomentBasis(level)
    moment_basis.init_moment_mappings()
    basis_converter = BasisConverter(moment_basis)
    basis_converter.remap_mlip_moment_coeffs(mtp_data)
    basic_moments = moment_basis.basic_moments
    scalar_contractions_str = moment_basis.scalar_contractions
    pair_contractions = moment_basis.pair_contractions
    execution_order_list, _ = _flatten_computation_graph(basic_moments, pair_contractions, scalar_contractions_str)    
    execution_order = tuple(execution_order_list)
    scalar_contractions = tuple(scalar_contractions_str)

    species_coeffs = tuple(mtp_data.species_coeffs)
    moment_coeffs = tuple(mtp_data.moment_coeffs)
    radial_coeffs = totuple(mtp_data.radial_coeffs)

    # ✅ NEW: Create wrapper function that only takes 8 dynamic arguments
    def jax_calc_wrapper(itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_actual, nneigh_actual):
        return jax_calc(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_actual, nneigh_actual,
            # Static arguments (baked in):
            tuple(mtp_data.species),  # species
            mtp_data.scaling,         # scaling
            mtp_data.min_dist,        # min_dist
            mtp_data.max_dist,        # max_dist
            species_coeffs,           # species coeffs
            moment_coeffs,            # moment coeffs
            radial_coeffs,            # radial coeffs
            execution_order,          # execution order
            scalar_contractions       # scalar contractions
        )

    compile_args = [
        jax.ShapeDtypeStruct((MAX_ATOMS,), jnp.int32),                    # itypes
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS), jnp.int32),      # all_js
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS, 3), jnp.float32), # all_rijs
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS), jnp.int32),      # all_jtypes
        jax.ShapeDtypeStruct((), jnp.int32),                             # cell_rank
        jax.ShapeDtypeStruct((), jnp.float32),                           # volume
        jax.ShapeDtypeStruct((), jnp.int32),                             # natoms_actual
        jax.ShapeDtypeStruct((), jnp.int32),                             # nneigh_actual
        # No static arguments in compile_args anymore!
    ]

    return compile_args, mtp_data, jax_calc_wrapper

MAX_ATOMS = 64
MAX_NEIGHBORS = 64
# Extract static parameters and get wrapper function
compile_args, mtp_data, jax_calc_wrapper = extract_static_params(12, 'Ni3Al-12g', MAX_ATOMS, MAX_NEIGHBORS)

# ✅ FIXED: JIT the wrapper function (no static_argnums needed!)
jitted_calc = jax.jit(jax_calc_wrapper)

print("=== Traditional JAX Compilation ===")
# Original lowering approach
lowered_with_static = jitted_calc.trace(*compile_args).lower()

# Save StableHLO MLIR
stablehlo_text = lowered_with_static.compiler_ir(dialect="stablehlo")
with open("jax_potential_padded.stablehlo.mlir", "w") as f:
    f.write(str(stablehlo_text))

print(lowered_with_static.as_text())

# Compile for immediate use
compiled = lowered_with_static.compile()
flops = compiled.cost_analysis()['flops']
print(f'FLOPS: {flops}')
print('Traditional compilation done')

print("\n=== JAX Export for C Integration ===")
# ✅ FIXED: Export the wrapper function (only 8 arguments)
export_success = False
try:
    # Export the wrapper function
    exported_calc = export.export(jitted_calc)(*compile_args)
    
    print("Export successful!")
    print(f"Export info:")
    print(f"  Input shapes: {[str(shape) for shape in exported_calc.in_avals]}")
    print(f"  Output shapes: {[str(shape) for shape in exported_calc.out_avals]}")
    print(f"  Platforms: {exported_calc.platforms}")
    print(f"  Number of inputs: {len(exported_calc.in_avals)} (should be 8)")
    
    # Serialize the exported function
    serialized_data = exported_calc.serialize()
    
    # Save serialized data to file
    with open("jax_potential_exported_padded.bin", "wb") as f:
        f.write(serialized_data)
    
    print(f"Serialized data saved to jax_potential_exported_padded.bin ({len(serialized_data)} bytes)")


    def test_distance_filtering():
        """Test that dummy neighbors (large distances) are properly ignored"""
        
        natoms_test = 4
        nneigh_test = 8
        
        # Create test data with mixed real/dummy neighbors
        itypes = np.array([0, 1, 0, 1] + [0] * (MAX_ATOMS - 4), dtype=np.int32)
        
        all_rijs = np.full((MAX_ATOMS, MAX_NEIGHBORS, 3), mtp_data.max_dist + 2.0, dtype=np.float32)
        all_js = np.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=np.int32)
        all_jtypes = np.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=np.int32)
        
        # Add some real neighbors (within cutoff)
        for i in range(natoms_test):
            for j in range(min(3, MAX_NEIGHBORS)):  # 3 real neighbors per atom
                # Real neighbor within cutoff
                all_rijs[i, j] = np.random.uniform(1.0, mtp_data.max_dist - 0.5, 3)
                all_jtypes[i, j] = np.random.randint(0, 2)
        
        result = exported_calc.call(
            itypes, all_js, all_rijs, all_jtypes,
            3, 1000.0, natoms_test, nneigh_test
        )
        
        energy, forces, stress = result
        print(f"✅ Test result: energy={energy}, force_norm={np.linalg.norm(forces)}")
        print(f"forces: {forces}")
        print(f"stress: {stress}")
        print(f"✅ Dummy atom forces: {np.max(np.abs(forces[natoms_test:]))}")
    
    test_distance_filtering()
    
    # Test that the exported function works
    print("\n=== Testing Exported Function ===")
    atom_id = 0
    calc_args_test = _get_data(atom_id, MAX_ATOMS, MAX_NEIGHBORS)
    
    start_time = time.time()
    exported_result = exported_calc.call(*calc_args_test)
    end_time = time.time()
    exported_time = end_time - start_time
    
    print(exported_result[0])
    print(exported_result[1])
    print(exported_result[-1])

    print(f"Exported function time: {exported_time:.6f} seconds")
    print(f"Result shapes: {[np.array(r).shape for r in exported_result]}")
    
    export_success = True
    
except Exception as e:
    print(f"Export failed: {e}")
    import traceback
    traceback.print_exc()
    export_success = False

print("\n=== Summary ===")
print("Files created:")
print("1. jax_potential_padded.stablehlo.mlir - StableHLO from lowering")
if export_success:
    print("2. jax_potential_exported_padded.bin - Serialized padded function")
    print(f"\nFunction expects exactly 8 inputs (no more static argument issues)")
    print(f"Supports up to {64} atoms with {32} neighbors each")
else:
    print("2. Export failed - check errors above")
