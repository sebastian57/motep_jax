import os

# Force JAX to use GPU and optimize for GPU compilation
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Use CUDA explicitly for your RTX 3060 Ti
# Remove problematic XLA flags for JAX 0.6.2
# os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True'
os.environ['JAX_ENABLE_X64'] = 'False'  # Use float32 for better GPU performance

import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import export, device_put

# Verify GPU availability before compilation
print("=== GPU Setup Check ===")
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

gpu_devices = jax.devices('gpu')
cuda_devices = [d for d in jax.devices() if 'cuda' in str(d).lower()]

if len(gpu_devices) == 0 and len(cuda_devices) == 0:
    print("❌ WARNING: No GPU devices found! Falling back to CPU")
    print("   Make sure you installed JAX with CUDA support:")
    print("   pip install --upgrade 'jax[cuda12]'")
else:
    if len(cuda_devices) > 0:
        print(f"✅ Found {len(cuda_devices)} CUDA device(s): {cuda_devices}")
        # Set default device to GPU
        jax.config.update('jax_default_device', cuda_devices[0])
    else:
        print(f"✅ Found {len(gpu_devices)} GPU device(s): {gpu_devices}")
        jax.config.update('jax_default_device', gpu_devices[0])

from motep_original_files.jax_engine.jax_pad import calc_energy_forces_stress_padded_simple as jax_calc
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
    
    atoms_to_copy = min(natoms_actual, MAX_ATOMS)
    neighbors_to_copy = min(nneigh_actual, MAX_NEIGHBORS)
    
    itypes_padded[:atoms_to_copy] = itypes[:atoms_to_copy]
    all_js_padded[:atoms_to_copy, :neighbors_to_copy] = all_js[:atoms_to_copy, :neighbors_to_copy]
    all_rijs_padded[:atoms_to_copy, :neighbors_to_copy] = all_rijs[:atoms_to_copy, :neighbors_to_copy]
    all_jtypes_padded[:atoms_to_copy, :neighbors_to_copy] = all_jtypes[:atoms_to_copy, :neighbors_to_copy]
    
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

    def jax_calc_wrapper(itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_actual, nneigh_actual):
        return jax_calc(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_actual, nneigh_actual,
            tuple(mtp_data.species),
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            species_coeffs,
            moment_coeffs,
            radial_coeffs,
            execution_order,
            scalar_contractions
        )

    compile_args = [
        jax.ShapeDtypeStruct((MAX_ATOMS,), jnp.int32),
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS), jnp.int32),
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS, 3), jnp.float32),
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    ]

    return compile_args, mtp_data, jax_calc_wrapper

MAX_ATOMS = 64
MAX_NEIGHBORS = 64

print("\n=== Extracting MTP Parameters ===")
compile_args, mtp_data, jax_calc_wrapper = extract_static_params(12, 'Ni3Al-12g', MAX_ATOMS, MAX_NEIGHBORS)

print("\n=== JIT Compilation for GPU ===")
# Force compilation on GPU if available
available_gpus = jax.devices('gpu') + [d for d in jax.devices() if 'cuda' in str(d).lower()]

if len(available_gpus) > 0:
    with jax.default_device(available_gpus[0]):
        jitted_calc = jax.jit(jax_calc_wrapper)
        print(f"✅ Compiling on GPU device: {available_gpus[0]}")
else:
    jitted_calc = jax.jit(jax_calc_wrapper)
    print("⚠️  Compiling on CPU (no GPU available)")

print("\n=== Traditional JAX Compilation ===")
lowered_with_static = jitted_calc.trace(*compile_args).lower()

# Check compilation target (compatible with JAX 0.6.2)
try:
    if hasattr(lowered_with_static, 'compile_args') and hasattr(lowered_with_static.compile_args, 'target_config'):
        compilation_platform = lowered_with_static.compile_args.target_config.platform_name
    else:
        # Fallback for JAX 0.6.2 - infer from devices
        if len(available_gpus) > 0:
            compilation_platform = "gpu"
        else:
            compilation_platform = "cpu"
except:
    compilation_platform = "unknown"
    
print(f"Compilation target: {compilation_platform}")

# Save StableHLO MLIR
stablehlo_text = lowered_with_static.compiler_ir(dialect="stablehlo")
mlir_filename = f"jax_potential_gpu.stablehlo.mlir"
with open(mlir_filename, "w") as f:
    f.write(str(stablehlo_text))
print(f"Saved StableHLO to: {mlir_filename}")

compiled = lowered_with_static.compile()
flops = compiled.cost_analysis()['flops']
print(f'FLOPS: {flops}')

print("\n=== JAX Export for GPU Integration ===")
export_success = False
try:
    # Move test data to GPU if available
    test_data = _get_data(0, MAX_ATOMS, MAX_NEIGHBORS)
    available_gpus = jax.devices('gpu') + [d for d in jax.devices() if 'cuda' in str(d).lower()]
    
    if len(available_gpus) > 0:
        # Put test data on GPU
        test_data_gpu = [device_put(arr, available_gpus[0]) for arr in test_data]
        print(f"✅ Test data moved to GPU: {available_gpus[0]}")
    else:
        test_data_gpu = test_data
        print("⚠️  Using CPU for test data")
    
    # Export the function - this preserves the compilation target
    exported_calc = export.export(jitted_calc)(*compile_args)
    
    print("✅ Export successful!")
    print(f"Export info:")
    print(f"  Input shapes: {[str(shape) for shape in exported_calc.in_avals]}")
    print(f"  Output shapes: {[str(shape) for shape in exported_calc.out_avals]}")
    print(f"  Platforms: {exported_calc.platforms}")
    print(f"  Number of inputs: {len(exported_calc.in_avals)}")
    
    # Test the exported function
    print("\n=== Testing Exported Function ===")
    start_time = time.time()
    exported_result = exported_calc.call(*test_data_gpu)
    end_time = time.time()
    exported_time = end_time - start_time
    
    print(f"✅ Exported function test successful!")
    print(f"Execution time: {exported_time:.6f} seconds")
    print(f"Result shapes: {[np.array(r).shape for r in exported_result]}")
    print(f"Energy: {exported_result[0]}")
    
    # Serialize for C++ integration
    serialized_data = exported_calc.serialize()
    bin_filename = "jax_potential_gpu.bin"
    with open(bin_filename, "wb") as f:
        f.write(serialized_data)
    
    print(f"✅ Serialized GPU function saved to: {bin_filename} ({len(serialized_data)} bytes)")
    
    export_success = True
    
except Exception as e:
    print(f"❌ Export failed: {e}")
    import traceback
    traceback.print_exc()
    export_success = False

print("\n=== Summary ===")
print("Files created:")
print(f"1. {mlir_filename} - StableHLO from lowering")
if export_success:
    print(f"2. {bin_filename} - Serialized GPU function")
    print(f"✅ Function compiled for: {compilation_platform}")
    print(f"✅ Supports up to {MAX_ATOMS} atoms with {MAX_NEIGHBORS} neighbors each")
    print(f"✅ Ready for LAMMPS integration!")
else:
    print("❌ Export failed - check errors above")

# Final verification
print(f"\n=== Final Verification ===")
print(f"Current JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

available_gpus = jax.devices('gpu') + [d for d in jax.devices() if 'cuda' in str(d).lower()]
if len(available_gpus) > 0:
    print(f"✅ GPU compilation successful - use {bin_filename} in LAMMPS")
    print(f"✅ Target GPU: {available_gpus[0]} (RTX 3060 Ti)")
else:
    print(f"⚠️  CPU-only compilation - install JAX with CUDA support for GPU acceleration")
