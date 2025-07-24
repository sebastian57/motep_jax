import numpy as np
from motep_original_files.mtp import read_mtp
from motep_original_files.setting import parse_setting
from motep_original_files.calculator import MTP

import jax
import jax.numpy as jnp
from jax import lax, export
import optax
from functools import partial

# import auxiliary functions
from motep_jax_train_import import*
jax.config.update('jax_enable_x64', False)

def load_compiled_mtp_function(bin_filename):
    """Load compiled trainable MTP function from .bin file"""
    with open(bin_filename, "rb") as f:
        serialized_data = f.read()
    
    exported_fn = export.deserialize(serialized_data)
    print(f"✅ Loaded compiled MTP from {bin_filename}")
    print(f"   Input shapes: {[str(shape) for shape in exported_fn.in_avals]}")
    print(f"   Output shapes: {[str(shape) for shape in exported_fn.out_avals]}")
    print(f"   Platforms: {exported_fn.platforms}")
    
    return exported_fn

def train4_with_compiled_mtp(
    bin_filename,  # NEW: Path to compiled .bin file
    training_cfg, 
    level, 
    steps_lbfgs, 
    threshold_loss, 
    min_steps, 
    lr_start, 
    transition_steps, 
    decay_rate, 
    global_norm_clip, 
    min_dist=0.5, 
    max_dist=5.0, 
    scaling=1.0, 
    species=None, 
    pkl_file='jax_images_data', 
    pkl_file_val='val_jax_images_data'
):
    """
    Updated train4 function using compiled .bin MTP function
    
    Key differences from original:
    - Loads compiled function instead of using mtp_instance.calculate_jax
    - Parameters passed as dictionary to compiled function
    - No more fromtuple conversions needed
    """
    
    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'

    # Load data (same as original)
    images_total = read_images([f'training_data/{training_cfg}'], species=species)    
    data_split = int(3/4*len(images_total))
    images = images_total[0:data_split][0:5]
    images_val = images_total[data_split:][0:1]
    
    rng = np.random.default_rng(10)
    
    mtp_data = read_mtp(untrained_mtp)
    mtp_data.species = species
    if species == None:
        mtp_data.species_count = 1
    else:
        mtp_data.species_count = len(species)
    mtp_data.min_dist = min_dist
    mtp_data.max_dist = max_dist
    mtp_data.scaling = scaling
    mtp_data.initialize(rng)    

    # Extract and save training data (same as original)
    extract_and_save_img_data(images, species, mtp_data, name=pkl_file)
    extract_and_save_img_data(images_val, species, mtp_data, name=pkl_file_val)
    
    jax_images = load_data_pickle(f'training_data/{pkl_file}.pkl')
    jax_val_images = load_data_pickle(f'training_data/{pkl_file_val}.pkl')
    
    # Load compiled MTP function
    compiled_mtp_fn = load_compiled_mtp_function(bin_filename)
    
    # Create wrapper for LLS compatibility
    def compiled_mtp_wrapper_for_lls(itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params):
        """
        Wrapper to make compiled function compatible with LLS solver
        
        Args:
            atomic_data... (original order)
            params: parameter dictionary
            
        Returns:
            dict with keys ['energy', 'forces', 'stress']
        """
        # Pad atomic data to expected shapes
        MAX_ATOMS = 2
        MAX_NEIGHBORS = 1
        
        natoms_actual = len(itypes)
        nneigh_actual = all_js.shape[1] if len(all_js.shape) > 1 else 1
        
        # Pad arrays if needed
        current_atoms = itypes.shape[0]
        current_neighbors = all_js.shape[1] if len(all_js.shape) > 1 else 0
        
        if current_atoms < MAX_ATOMS or current_neighbors < MAX_NEIGHBORS:
            itypes_padded = jnp.zeros(MAX_ATOMS, dtype=jnp.int32)
            all_js_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=jnp.int32)
            all_rijs_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS, 3), dtype=jnp.float32)
            all_jtypes_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=jnp.int32)
            
            atoms_to_copy = min(natoms_actual, MAX_ATOMS)
            neighbors_to_copy = min(nneigh_actual, MAX_NEIGHBORS)
            
            itypes_padded = itypes_padded.at[:atoms_to_copy].set(itypes[:atoms_to_copy])
            all_js_padded = all_js_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_js[:atoms_to_copy, :neighbors_to_copy])
            all_rijs_padded = all_rijs_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_rijs[:atoms_to_copy, :neighbors_to_copy])
            all_jtypes_padded = all_jtypes_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_jtypes[:atoms_to_copy, :neighbors_to_copy])
        else:
            itypes_padded = itypes
            all_js_padded = all_js
            all_rijs_padded = all_rijs
            all_jtypes_padded = all_jtypes
        
        # Call compiled function
        energy, forces, stress = compiled_mtp_fn.call(
            params,
            itypes_padded, all_js_padded, all_rijs_padded, all_jtypes_padded,
            cell_rank, volume, natoms_actual, nneigh_actual
        )
        
        # Return in format expected by LLS
        return {
            'energy': energy,
            'forces': forces,
            'stress': stress
        }
    
    def solve_lls_for_basis_compiled(
        compiled_prediction_fn,
        params,
        jax_images,
        training_ids,
        weight_e, 
        weight_f,
        weight_s,
        num_basis_params,
        num_targets_per_config,
        num_f_components_per_config, 
        num_s_components_per_config,
        num_configs
    ):
        """
        Linear least squares solver adapted for compiled MTP function
        """
        
        def flatten_targets(E, F, sigma):
            E_arr = jnp.atleast_1d(E)
            F_flat = F.reshape(-1)
            sigma_flat = sigma.reshape(-1) 
            return jnp.concatenate((E_arr, F_flat, sigma_flat))
       
        def predict_with_separated_params(basis_p, fixed_p, structure_inputs):
            current_params = {
                'species': fixed_p['species'], 
                'radial': fixed_p['radial'], 
                'basis': basis_p
            }
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume = structure_inputs
            targets = compiled_prediction_fn(
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, current_params
            )
            E_pred, F_pred, sigma_pred = targets['energy'], targets['forces'], targets['stress']
            return flatten_targets(E_pred, F_pred, sigma_pred)

        def calculate_offset_contribution(fixed_p, structure_inputs):
            zero_basis = jnp.zeros((num_basis_params,), dtype=params['basis'].dtype)
            return predict_with_separated_params(zero_basis, fixed_p, structure_inputs)

        get_basis_matrix_single = jax.jacfwd(predict_with_separated_params, argnums=0)

        def get_single_config_data_for_lls(atoms_id):
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E_true, F_true, sigma_true = get_data_for_indices(jax_images, atoms_id)
            structure_inputs = (itypes, all_js, all_rijs, all_jtypes, cell_rank, volume)
            true_targets_flat = flatten_targets(E_true, F_true, sigma_true)
            return structure_inputs, true_targets_flat

        all_structure_inputs, all_true_targets_flat = jax.lax.map(get_single_config_data_for_lls, training_ids)

        fixed_params = {'species': params['species'], 'radial': params['radial']}
        
        # Use lax.map instead of jax.vmap for exported function compatibility
        all_B_matrices = jax.lax.map(
            lambda structure_inputs: get_basis_matrix_single(params['basis'], fixed_params, structure_inputs),
            all_structure_inputs
        )
        all_offsets = jax.lax.map(
            lambda structure_inputs: calculate_offset_contribution(fixed_params, structure_inputs),
            all_structure_inputs
        )

        num_configs = len(training_ids)
        total_targets = num_configs * num_targets_per_config
        X = all_B_matrices.reshape(total_targets, num_basis_params)
        y_true_flat = all_true_targets_flat.reshape(total_targets)
        offsets_flat = all_offsets.reshape(total_targets)
        y_prime = y_true_flat - offsets_flat

        sqrt_we = jnp.sqrt(weight_e)
        sqrt_wf = jnp.sqrt(weight_f)
        sqrt_ws = jnp.sqrt(weight_s)

        weights_e_part = jnp.full((1,), sqrt_we)
        weights_f_part = jnp.full((num_f_components_per_config,), sqrt_wf)
        weights_s_part = jnp.full((num_s_components_per_config,), sqrt_ws)
        sqrt_weights_per_config = jnp.concatenate((weights_e_part, weights_f_part, weights_s_part))

        sqrt_weights = jnp.tile(sqrt_weights_per_config, num_configs)

        X_weighted = X * sqrt_weights[:, None]
        y_prime_weighted = y_prime * sqrt_weights

        w_basis_optimal, residuals, rank, s = jnp.linalg.lstsq(X_weighted, y_prime_weighted, rcond=None)

        return w_basis_optimal
    
    # Training configuration (same as original)
    num_basis_params = mtp_data.moment_coeffs.shape[0]
    n_atoms_representative = int(jax_images['n_atoms'][0])
    num_f_components_per_config = 3 * n_atoms_representative
    num_s_components_per_config = 6
    num_targets_per_config = 1 + num_f_components_per_config + num_s_components_per_config
    training_ids = np.arange(len(jax_images['E']))
    weight_e, weight_f, weight_s = 1.0, 0.01, 0.001
    num_configs = len(jax_images['E'])

    print(f"📊 Training configuration:")
    print(f"   Training configs: {len(jax_images['E'])}")
    print(f"   Validation configs: {len(jax_val_images['E'])}")
    print(f"   Basis parameters: {num_basis_params}")
    print(f"   Representative atoms: {n_atoms_representative}")
    
    @partial(jax.jit, static_argnames=("num_basis_params", "num_targets_per_config",
                                      "num_f_components_per_config", "num_s_components_per_config",
                                      "num_configs"))
    def fit(num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs):
        
        def loss_function(predictions, real_values, we=1.0, wf=0.01, ws=0.001):
            E_pred, F_pred, sigma_pred = predictions
            E_real, F_real, sigma_real = real_values
            loss_E = we * jnp.sum((E_pred - E_real)**2)
            loss_F = wf * jnp.sum((F_pred - real_values[1])**2)
            loss_sigma = ws * jnp.sum((sigma_pred - real_values[2])**2)
            return loss_E + loss_F + loss_sigma
    
        def loss_epoch_compiled(params, atoms_ids):
            """Loss function using compiled MTP - avoiding jax.vmap"""

            def predict_compiled(atoms_id):
                # Get atomic data (same as original)
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, atoms_id)
                
                # Pad atomic data to expected shapes
                MAX_ATOMS = 2
                MAX_NEIGHBORS = 1
                
                natoms_actual = len(itypes)
                nneigh_actual = all_js.shape[1] if len(all_js.shape) > 1 else 1
                
                # Pad arrays
                itypes_padded = jnp.zeros(MAX_ATOMS, dtype=jnp.int32)
                all_js_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=jnp.int32)
                all_rijs_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS, 3), dtype=jnp.float32)
                all_jtypes_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=jnp.int32)
                
                atoms_to_copy = min(natoms_actual, MAX_ATOMS)
                neighbors_to_copy = min(nneigh_actual, MAX_NEIGHBORS)
                
                itypes_padded = itypes_padded.at[:atoms_to_copy].set(itypes[:atoms_to_copy])
                all_js_padded = all_js_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_js[:atoms_to_copy, :neighbors_to_copy])
                all_rijs_padded = all_rijs_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_rijs[:atoms_to_copy, :neighbors_to_copy])
                all_jtypes_padded = all_jtypes_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_jtypes[:atoms_to_copy, :neighbors_to_copy])

                # Call compiled function with parameter dictionary
                pred_energy, pred_forces_padded, pred_stress = compiled_mtp_fn.call(
                    params,  # Dictionary of trainable parameters
                    itypes_padded, all_js_padded, all_rijs_padded, all_jtypes_padded,
                    cell_rank, volume, natoms_actual, nneigh_actual
                )
                
                # 🔥 CRITICAL FIX: Slice forces to match actual number of atoms
                pred_forces = pred_forces_padded[:natoms_actual]  # Shape: [natoms_actual, 3]
                
                # Now pred_forces matches F shape exactly
                return (pred_energy, pred_forces, pred_stress), (E, F, sigma)
            
            # Use lax.map instead of jax.vmap for exported functions
            predictions, real_values = jax.lax.map(predict_compiled, atoms_ids)
            return loss_function(predictions, real_values)
        
        def loss_epoch_val_compiled(params, atoms_ids):
            """Validation loss using compiled MTP - avoiding jax.vmap"""
            def predict_compiled(atoms_id):
                # Get atomic data (same as original)
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, atoms_id)
                
                # Pad atomic data to expected shapes
                MAX_ATOMS = 2
                MAX_NEIGHBORS = 1
                
                natoms_actual = len(itypes)
                nneigh_actual = all_js.shape[1] if len(all_js.shape) > 1 else 1
                
                # Pad arrays
                itypes_padded = jnp.zeros(MAX_ATOMS, dtype=jnp.int32)
                all_js_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=jnp.int32)
                all_rijs_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS, 3), dtype=jnp.float32)
                all_jtypes_padded = jnp.zeros((MAX_ATOMS, MAX_NEIGHBORS), dtype=jnp.int32)
                
                atoms_to_copy = min(natoms_actual, MAX_ATOMS)
                neighbors_to_copy = min(nneigh_actual, MAX_NEIGHBORS)
                
                itypes_padded = itypes_padded.at[:atoms_to_copy].set(itypes[:atoms_to_copy])
                all_js_padded = all_js_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_js[:atoms_to_copy, :neighbors_to_copy])
                all_rijs_padded = all_rijs_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_rijs[:atoms_to_copy, :neighbors_to_copy])
                all_jtypes_padded = all_jtypes_padded.at[:atoms_to_copy, :neighbors_to_copy].set(all_jtypes[:atoms_to_copy, :neighbors_to_copy])

                # Call compiled function with parameter dictionary
                pred_energy, pred_forces_padded, pred_stress = compiled_mtp_fn.call(
                    params,  # Dictionary of trainable parameters
                    itypes_padded, all_js_padded, all_rijs_padded, all_jtypes_padded,
                    cell_rank, volume, natoms_actual, nneigh_actual
                )
                
                # 🔥 CRITICAL FIX: Slice forces to match actual number of atoms
                pred_forces = pred_forces_padded[:natoms_actual]  # Shape: [natoms_actual, 3]
                
                # Now pred_forces matches F shape exactly
                return (pred_energy, pred_forces, pred_stress), (E, F, sigma)
            
            # Use lax.map instead of jax.vmap for exported functions
            predictions, real_values = jax.lax.map(predict_compiled, atoms_ids)
            return loss_function(predictions, real_values)
    
        def epoch_step_lbfgs(carry, step):
            params = carry['params']
            opt_state = carry['opt_state']
            key = carry['key']
            loss_history = carry['loss_history']
            val_loss_history = carry['val_loss_history']
    
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, grads = loss_and_grads(params, atoms_ids)
            clipped_grads, _ = optax.clip_by_global_norm(global_norm_clip).update(grads, None)
            updates, new_opt_state = optimizer_lbfgs.update(
                clipped_grads, opt_state, params,
                value=loss,
                grad=clipped_grads,
                value_fn=lambda p: loss_epoch_compiled(p, atoms_ids)
            )
            
            key, subkey = jax.random.split(key)
            atoms_ids_val = jax.random.permutation(subkey, len(images_val))
            val_loss = loss_epoch_val_compiled(params, atoms_ids_val)
            new_val_loss_history = val_loss_history.at[step].set(val_loss)

            new_params = optax.apply_updates(params, updates)
            new_loss_history = loss_history.at[step].set(loss)
            new_carry = carry.copy()
            
            new_carry.update({
                'params': new_params,
                'opt_state': new_opt_state,
                'key': key,
                'loss_history': new_loss_history,
                'val_loss_history': new_val_loss_history
            })
            
            return new_carry, loss
        
        def compute_init_loss(state):
            params = state['params']
            key = state['key']
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss = loss_epoch_compiled(params, atoms_ids)
            return loss, key
        
        # Initialize parameters (same structure as original)
        key = jax.random.PRNGKey(42)
        
        params_pre_lls = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': mtp_data.moment_coeffs  # Note: using 'basis' key
        }
        
        atoms_ids = jax.random.permutation(key, len(images))
        initial_loss = loss_epoch_compiled(params_pre_lls, atoms_ids)
        
        # TODO: Fix LLS to work with exported functions - using simple initialization for now
        print(f"🔧 Using simple parameter initialization (LLS temporarily disabled)")
        params = {
            'species': mtp_data.species_coeffs * 1e-2,
            'radial': mtp_data.radial_coeffs * 1e-2,
            'basis': mtp_data.moment_coeffs * 1e-2
        }
        
        # Linear least squares initialization using compiled function
        # print(f"🔍 Running linear least squares initialization...")
        # opt_basis_lls = solve_lls_for_basis_compiled(
        #     compiled_mtp_wrapper_for_lls,
        #     params_pre_lls, 
        #     jax_images, 
        #     training_ids, 
        #     weight_e, 
        #     weight_f, 
        #     weight_s, 
        #     num_basis_params, 
        #     num_targets_per_config, 
        #     num_f_components_per_config, 
        #     num_s_components_per_config, 
        #     num_configs
        # )
        
        # params = {
        #     'species': mtp_data.species_coeffs,
        #     'radial': mtp_data.radial_coeffs,
        #     'basis': opt_basis_lls  # Use LLS optimized basis
        # }
        
        atoms_ids = jax.random.permutation(key, len(images))
        loss_after_init = loss_epoch_compiled(params, atoms_ids)
        

        # Optimizer setup (same as original)
        lr_schedule_lbfgs = optax.exponential_decay(
            init_value=lr_start,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        
        optimizer_lbfgs = optax.lbfgs(learning_rate=lr_schedule_lbfgs)
        opt_state = optimizer_lbfgs.init(params)
    
        state = {
            'params': params, 
            'opt_state': opt_state, 
            'key': key, 
            'loss_history': jnp.full(steps_lbfgs, jnp.nan), 
            'val_loss_history': jnp.full(steps_lbfgs, jnp.nan)
        }
        
        loss_and_grads = jax.value_and_grad(loss_epoch_compiled)

        # Training loop (same convergence logic as original)
        init_loss, new_key = compute_init_loss(state)
        state = {**state, 'key': new_key}
        init = (0, state, init_loss, jnp.inf)
        
        def cond(carry):
            step, state, loss, prev_loss = carry
            converged_by_loss = jnp.logical_and(prev_loss > loss, (prev_loss - loss) <= threshold_loss) 
            is_less_than_min_steps = (step < min_steps)
            converged = jnp.where(is_less_than_min_steps, 
                                  jnp.array(False), 
                                  converged_by_loss)        
            continue_loop = jnp.logical_and(step < steps_lbfgs, jnp.logical_not(converged))
            return continue_loop
    
        def body(carry):
            step, state, loss, prev_loss = carry
            new_state, new_loss = epoch_step_lbfgs(state, step)
            return (step + 1, new_state, new_loss, loss)
    
        step, state, final_loss, prev_loss = lax.while_loop(cond, body, init)
        loss_history = state['loss_history']
        val_loss_history = state['val_loss_history']        
    
        steps_performed = [step]
    
        return state, jnp.array([final_loss]), steps_performed, loss_history, val_loss_history
    
    # Execute training
    print(f"\n🚀 Starting training with compiled MTP...")
    epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history = fit(
        num_basis_params, num_targets_per_config, num_f_components_per_config, 
        num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs
    )
        
    # Clean up results (same as original)
    nan_mask = ~np.isnan(loss_history)
    loss_history = loss_history[nan_mask]
    val_loss_history = val_loss_history[nan_mask]
    
    print(f"\n✅ Training completed!")
    print(f"   Steps performed: {steps_performed[0]}")
    print(f"   Final loss: {epoch_losses[0]:.6f}")
    print(f"   Loss history length: {len(loss_history)}")
    
    return epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history

# Example usage
def run_training_with_compiled_mtp():
    """Complete training pipeline with compiled MTP"""
    
    # Step 1: Compile trainable MTP (run compile_trainable_mtp_gpu.py first)
    bin_filename = "trainable_mtp_gpu_12_with_vjp.bin"
    
    # Step 2: Train using compiled function
    result = train4_with_compiled_mtp(
        bin_filename=bin_filename,
        training_cfg='training.cfg',
        level=12,
        steps_lbfgs=500,
        threshold_loss=1e-4,
        min_steps=20,
        lr_start=1e-2,
        transition_steps=50,
        decay_rate=0.95,
        global_norm_clip=0.1,
        species=[0]  
    )
    
    final_state, final_loss, steps, loss_hist, val_loss_hist = result
    
    # Step 3: Save trained parameters
    trained_params = final_state['params']
    print(f"\n💾 Saving trained parameters...")
    print(f"   Species coeffs shape: {trained_params['species'].shape}")
    print(f"   Basis coeffs shape: {trained_params['basis'].shape}")
    print(f"   Radial coeffs shape: {trained_params['radial'].shape}")
    
    # You can save to .mtp file using write_mtp_file function
    # write_mtp_file(level=12, species=[0, 1], params=trained_params, file='trained_mtp.mtp')
    
    print("🎉 Training pipeline completed!")
    return result

if __name__ == "__main__":
    import time 
    
    start_time = time.time()
    run_training_with_compiled_mtp()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)