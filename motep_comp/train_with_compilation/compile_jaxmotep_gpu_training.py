import os

# Force JAX to use GPU and optimize for GPU compilation
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
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
        jax.config.update('jax_default_device', cuda_devices[0])
    else:
        print(f"✅ Found {len(gpu_devices)} GPU device(s): {gpu_devices}")
        jax.config.update('jax_default_device', gpu_devices[0])

from motep_original_files.jax_engine.jax_pad import calc_energy_forces_stress_padded_simple_trainable as jax_calc_trainable
from motep_original_files.jax_engine.moment_jax import MomentBasis
from motep_original_files.mtp import read_mtp
from motep_original_files.calculator import MTP
from motep_original_files.jax_engine.conversion import BasisConverter
from motep_jax_train_import import*

def _initialize_mtp(mtp_file):
    rng = np.random.default_rng(10)
    mtp_data = read_mtp(mtp_file)
    mtp_data.initialize(rng)
    mtp_data.species = np.arange(0, mtp_data.species_count)
    return mtp_data

def _get_test_data_and_params(atom_id, level, mtp_file, MAX_ATOMS, MAX_NEIGHBORS):
    """Get test data and extract parameter shapes from MTP"""
    
    # Load MTP to get parameter shapes
    mtp_data = _initialize_mtp(f'{mtp_file}')
    
    # Get test atomic data
    jax_val_images = load_data_pickle(f'training_data/val_jax_images_data.pkl')     
    initial_args = get_data_for_indices(jax_val_images, atom_id)[0:6]
    
    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume = initial_args
    
    natoms_actual = len(itypes)
    nneigh_actual = all_js.shape[1] if len(all_js.shape) > 1 else 1
    
    # Pad atomic data
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
    
    atomic_data = [itypes_padded, all_js_padded, all_rijs_padded, all_jtypes_padded, 
                   cell_rank, volume, natoms_actual, nneigh_actual]
    
    # Create test parameters with correct shapes
    test_params = {
        'species': mtp_data.species_coeffs.astype(np.float32),
        'basis': mtp_data.moment_coeffs.astype(np.float32),
        'radial': mtp_data.radial_coeffs.astype(np.float32)
    }
    
    return atomic_data, test_params, mtp_data

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

def extract_static_params_trainable(level, mtp_file, MAX_ATOMS, MAX_NEIGHBORS):
    """Extract static parameters and create trainable compilation function"""
    
    mtp_data = _initialize_mtp(f'{mtp_file}')

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

    def trainable_jax_calc_wrapper(params, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_actual, nneigh_actual):
        """
        Trainable wrapper function that takes parameters as first argument
        
        Args:
            params: Dict with keys ['species', 'basis', 'radial'] containing trainable parameters
            ... other args same as before
        """
        return jax_calc_trainable(
            params,  # NEW: trainable parameters as first argument
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, natoms_actual, nneigh_actual,
            tuple(mtp_data.species),  # Static
            mtp_data.scaling,         # Static
            mtp_data.min_dist,        # Static  
            mtp_data.max_dist,        # Static
            execution_order,          # Static
            scalar_contractions       # Static
        )

    # Get parameter shapes from MTP data
    param_shapes = {
        'species': jax.ShapeDtypeStruct(mtp_data.species_coeffs.shape, jnp.float32),
        'basis': jax.ShapeDtypeStruct(mtp_data.moment_coeffs.shape, jnp.float32),
        'radial': jax.ShapeDtypeStruct(mtp_data.radial_coeffs.shape, jnp.float32)
    }
    
    # Atomic data shapes (same as before)
    atomic_data_shapes = [
        jax.ShapeDtypeStruct((MAX_ATOMS,), jnp.int32),
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS), jnp.int32),
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS, 3), jnp.float32),
        jax.ShapeDtypeStruct((MAX_ATOMS, MAX_NEIGHBORS), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    ]

    # Combine parameter and atomic data shapes
    compile_args = [param_shapes] + atomic_data_shapes

    return compile_args, mtp_data, trainable_jax_calc_wrapper, param_shapes

def test_vjp_support_rehydrated(bin_filename, test_args):
    """Test if a .bin file supports VJP after rehydration"""
    try:
        # Load and deserialize
        with open(bin_filename, "rb") as f:
            serialized_data = f.read()
        
        exported_fn = export.deserialize(serialized_data)
        
        # Test basic VJP availability
        vjp_fn = exported_fn.vjp()
        if vjp_fn is None:
            return False, "No VJP function available"
        
        # Test gradient computation
        def simple_loss(*args):
            result = exported_fn.call(*args)
            return result[0]  # Use energy as loss
        
        grad_fn = jax.grad(simple_loss)
        grads = grad_fn(*test_args)
        
        return True, f"VJP working - gradients computed successfully"
        
    except Exception as e:
        return False, f"VJP test failed: {e}"

def compile_trainable_mtp_gpu(level, mtp_file, MAX_ATOMS=64, MAX_NEIGHBORS=64):
    """Compile trainable MTP for GPU execution WITH VJP support (JAX 0.6.2)"""
    
    print(f"\n=== Extracting MTP Parameters for Trainable Compilation ===")
    compile_args, mtp_data, trainable_wrapper, param_shapes = extract_static_params_trainable(level, mtp_file, MAX_ATOMS, MAX_NEIGHBORS)
    
    print(f"Parameter shapes:")
    for key, shape in param_shapes.items():
        print(f"  {key}: {shape.shape} ({shape.dtype})")

    print("\n=== JIT Compilation for GPU (Trainable) ===")
    # Force compilation on GPU if available
    available_gpus = jax.devices('gpu') + [d for d in jax.devices() if 'cuda' in str(d).lower()]

    if len(available_gpus) > 0:
        with jax.default_device(available_gpus[0]):
            jitted_calc = jax.jit(trainable_wrapper)
            print(f"✅ Compiling trainable function on GPU device: {available_gpus[0]}")
    else:
        jitted_calc = jax.jit(trainable_wrapper)
        print("⚠️  Compiling trainable function on CPU (no GPU available)")

    print("\n=== JAX Export with VJP for Training (JAX 0.6.2 API) ===")
    export_success = False
    try:
        # Get test data and parameters
        test_atomic_data, test_params, _ = _get_test_data_and_params(0, level, mtp_file, MAX_ATOMS, MAX_NEIGHBORS)
        
        available_gpus = jax.devices('gpu') + [d for d in jax.devices() if 'cuda' in str(d).lower()]
        
        if len(available_gpus) > 0:
            # Put test data on GPU
            test_atomic_data_gpu = [device_put(arr, available_gpus[0]) for arr in test_atomic_data]
            test_params_gpu = {k: device_put(v, available_gpus[0]) for k, v in test_params.items()}
            print(f"✅ Test data moved to GPU: {available_gpus[0]}")
        else:
            test_atomic_data_gpu = test_atomic_data
            test_params_gpu = test_params
            print("⚠️  Using CPU for test data")
        
        # 🔥 CRITICAL FIX: Export normally, then serialize with VJP (JAX 0.6.2 API)
        print("🧮 Exporting function...")
        exported_calc = export.export(jitted_calc)(*compile_args)
        
        print("✅ Export successful!")
        print(f"Export info:")
        print(f"  Input shapes: {[str(shape) for shape in exported_calc.in_avals]}")
        print(f"  Output shapes: {[str(shape) for shape in exported_calc.out_avals]}")
        print(f"  Platforms: {exported_calc.platforms}")
        
        # Test the exported function (forward pass)
        print("\n=== Testing Forward Pass ===")
        #test_args = [test_params_gpu] + test_atomic_data_gpu
        
        #result = exported_calc.call(*test_args)
        #print(f"✅ Forward pass successful: {[r.shape for r in result]}")
        
        # 🔥 CRITICAL: Serialize with VJP support (JAX 0.6.2 API)
        print("\n=== Serializing with VJP Support ===")
        print("🧮 Serializing with VJP (gradient) support for training...")
        serialized_data = exported_calc.serialize(vjp_order=1)  # 🔥 THIS IS THE CORRECT API
        
        bin_filename = f"trainable_mtp_gpu_{level}_with_vjp.bin"
        with open(bin_filename, "wb") as f:
            f.write(serialized_data)
        
        print(f"✅ Serialized with VJP support: {bin_filename}")
        print(f"   File size: {len(serialized_data)} bytes")
        
        return jitted_calc, bin_filename, param_shapes
        
    except Exception as e:
        print(f"❌ Export with VJP failed: {e}")
        import traceback
        traceback.print_exc()
        export_success = False
        return None, None, None

    print("\n=== Summary ===")
    print("Files created:")
    if export_success:
        print(f"✅ {bin_filename} - Serialized trainable GPU function with VJP")
        print(f"✅ Function compiled for: GPU")
        print(f"✅ Supports up to {MAX_ATOMS} atoms with {MAX_NEIGHBORS} neighbors each")
        print(f"✅ Ready for training with gradients!")
    else:
        print("❌ Export failed - check errors above")

if __name__ == "__main__":
    MAX_ATOMS = 2
    MAX_NEIGHBORS = 1
    level = 12

    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'
    
    compiled_fn, bin_file, param_shapes = compile_trainable_mtp_gpu(
        level=level, 
        mtp_file=untrained_mtp, 
        MAX_ATOMS=MAX_ATOMS, 
        MAX_NEIGHBORS=MAX_NEIGHBORS
    )
    
    if compiled_fn is not None:
        print(f"\n🎉 SUCCESS! Trainable MTP with VJP compiled and ready!")
        print(f"📁 Binary file: {bin_file}")
        print(f"🏗️  Parameter shapes: {param_shapes}")
        print(f"🚀 Ready for training with dictionary parameters and gradients!")
    else:
        print(f"\n❌ Compilation failed - check errors above")