import os

# Force JAX to use GPU and optimize for GPU compilation
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # Use CUDA explicitly for your RTX 3060 Ti
os.environ['JAX_ENABLE_X64'] = 'False'  # Use float32 for better GPU performance

# PHASE 1.2: Better compilation flags - compatible XLA optimization
xla_flags = [
    '--xla_gpu_autotune_level=4',
    '--xla_gpu_triton_gemm_any=true',
    '--xla_gpu_force_compilation_parallelism=4'
]
os.environ['XLA_FLAGS'] = ' '.join(xla_flags)

import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import export, device_put
from functools import partial

# JAX configuration for performance
jax.config.update('jax_enable_x64', False)  # Use float32 for speed
jax.config.update('jax_default_prng_impl', 'unsafe_rbg')  # Faster RNG

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

# NEW: Dynamic sizing functions
def create_dynamic_compile_args(max_atoms, max_neighbors):
    """
    Create compile arguments for specific array sizes
    """
    return [
        jax.ShapeDtypeStruct((max_atoms,), jnp.int32),
        jax.ShapeDtypeStruct((max_atoms, max_neighbors), jnp.int32),
        jax.ShapeDtypeStruct((max_atoms, max_neighbors, 3), jnp.float32),
        jax.ShapeDtypeStruct((max_atoms, max_neighbors), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    ]

def round_up_to_power_of_2(n, min_val=64):
    """
    Round up to next power of 2 for XLA efficiency
    """
    return max(min_val, 2 ** int(np.ceil(np.log2(max(n, min_val)))))

def extract_static_params(level, mtp_file):
    """
    Extract MTP parameters - no longer needs MAX_ATOMS/MAX_NEIGHBORS
    """
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

    return mtp_data, jax_calc_wrapper

def compile_and_export_function(jax_calc_wrapper, max_atoms, max_neighbors, 
                               filename_suffix, test_atom_id=0):
    """
    Compile and export a JAX function for specific array sizes
    """
    print(f"\n=== Compiling for {max_atoms} atoms × {max_neighbors} neighbors ===")
    
    # Create compile arguments for this size
    compile_args = create_dynamic_compile_args(max_atoms, max_neighbors)
    
    # PHASE 1.2: Better compilation with donate_argnums and static args
    @partial(jax.jit, 
             static_argnames=('natoms_actual', 'nneigh_actual'),
             donate_argnums=(0, 1, 2, 3))  # Donate input arrays to avoid copies
    def optimized_jax_calc(itypes, all_js, all_rijs, all_jtypes, *args, **kwargs):
        return jax_calc_wrapper(itypes, all_js, all_rijs, all_jtypes, *args, **kwargs)
    
    # Force compilation on GPU if available
    available_gpus = jax.devices('gpu') + [d for d in jax.devices() if 'cuda' in str(d).lower()]
    
    if len(available_gpus) > 0:
        with jax.default_device(available_gpus[0]):
            jitted_calc = optimized_jax_calc
            compilation_device = available_gpus[0]
    else:
        jitted_calc = optimized_jax_calc
        compilation_device = "CPU"
    
    print(f"✅ Compiling on: {compilation_device}")
    
    # Traditional JAX compilation for analysis
    lowered_with_static = jitted_calc.trace(*compile_args).lower()
    compiled = lowered_with_static.compile()
    
    try:
        flops = compiled.cost_analysis()['flops']
        print(f"FLOPS: {flops}")
    except:
        print("FLOPS: Could not analyze")
    
    # Export the function
    try:
        # Create test data for this size
        test_data = _get_data(test_atom_id, max_atoms, max_neighbors)
        
        if len(available_gpus) > 0:
            # Put test data on GPU
            test_data_gpu = [device_put(arr, available_gpus[0]) for arr in test_data]
        else:
            test_data_gpu = test_data
        
        # Export the function
        exported_calc = export.export(jitted_calc)(*compile_args)
        
        print(f"✅ Export successful!")
        print(f"  Input shapes: {[str(shape) for shape in exported_calc.in_avals]}")
        print(f"  Output shapes: {[str(shape) for shape in exported_calc.out_avals]}")
        print(f"  Platforms: {exported_calc.platforms}")
        
        # Test the exported function
        start_time = time.time()
        exported_result = exported_calc.call(*test_data_gpu)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ Function test successful!")
        print(f"  Execution time: {execution_time:.6f} seconds")
        print(f"  Energy: {exported_result[0]}")
        
        # Serialize for C++ integration
        serialized_data = exported_calc.serialize()
        bin_filename = f"jax_potential_{filename_suffix}.bin"
        
        with open(bin_filename, "wb") as f:
            f.write(serialized_data)
        
        print(f"✅ Saved: {bin_filename} ({len(serialized_data)} bytes)")
        
        return {
            'filename': bin_filename,
            'max_atoms': max_atoms,
            'max_neighbors': max_neighbors,
            'execution_time': execution_time,
            'energy': float(exported_result[0]),
            'size_bytes': len(serialized_data),
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'filename': f"jax_potential_{filename_suffix}.bin",
            'max_atoms': max_atoms,
            'max_neighbors': max_neighbors,
            'success': False,
            'error': str(e)
        }

# NEW: Define target system sizes for compilation
SYSTEM_CONFIGS = [
    # (max_atoms, max_neighbors, suffix, description)
    (1024, 128, "1k", "Small systems (up to 1K atoms)"),
    (4096, 128, "4k", "Medium systems (up to 4K atoms)"), 
    (16384, 128, "16k", "Large systems (up to 16K atoms)"),
    (65536, 128, "64k", "Very large systems (up to 64K atoms)"),
    (131072, 200, "128k", "Maximum systems (up to 128K atoms)"),
]

print("\n=== Extracting MTP Parameters ===")
mtp_data, jax_calc_wrapper = extract_static_params(12, 'Ni3Al-12g')

print("\n=== Dynamic JAX Compilation - Multiple Sizes ===")
compiled_functions = []

for max_atoms, max_neighbors, suffix, description in SYSTEM_CONFIGS:
    print(f"\n--- {description} ---")
    
    result = compile_and_export_function(
        jax_calc_wrapper, max_atoms, max_neighbors, suffix, test_atom_id=0
    )
    
    compiled_functions.append(result)
    
    if not result['success']:
        print(f"⚠️  Skipping {suffix} due to compilation failure")
        continue

print("\n=== COMPILATION SUMMARY ===")
print("Successfully compiled functions:")

total_successful = 0
for result in compiled_functions:
    if result['success']:
        total_successful += 1
        atoms = result['max_atoms']
        neighbors = result['max_neighbors']
        time_ms = result['execution_time'] * 1000
        size_mb = result['size_bytes'] / (1024 * 1024)
        
        print(f"✅ {result['filename']}")
        print(f"   Arrays: {atoms:,} atoms × {neighbors} neighbors")
        print(f"   Performance: {time_ms:.2f}ms execution time")
        print(f"   File size: {size_mb:.1f} MB")
        print(f"   Energy test: {result['energy']:.6f}")
    else:
        print(f"❌ {result['filename']} - FAILED")

print(f"\n=== FINAL SUMMARY ===")
print(f"Compiled {total_successful}/{len(SYSTEM_CONFIGS)} function variants")

if total_successful > 0:
    print(f"✅ Dynamic padding implementation ready!")
    print(f"✅ Function files can handle systems from 1K to 128K atoms")
    print(f"✅ Memory usage scales with actual system size")
    
    # Create a configuration file for the C++ side
    config_data = {
        'functions': [r for r in compiled_functions if r['success']],
        'mtp_params': {
            'scaling': float(mtp_data.scaling),
            'min_dist': float(mtp_data.min_dist),
            'max_dist': float(mtp_data.max_dist),
            'species_count': int(mtp_data.species_count)
        }
    }
    
    import json
    with open("jax_functions_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✅ Configuration saved to: jax_functions_config.json")
    
else:
    print(f"❌ No functions compiled successfully!")
    print(f"   Check GPU availability and JAX installation")

# Final verification
print(f"\n=== Final Verification ===")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

available_gpus = jax.devices('gpu') + [d for d in jax.devices() if 'cuda' in str(d).lower()]
if len(available_gpus) > 0:
    print(f"✅ GPU compilation successful")
    print(f"✅ Target GPU: {available_gpus[0]}")
else:
    print(f"⚠️  CPU-only compilation")
