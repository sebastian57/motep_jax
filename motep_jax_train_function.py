import numpy as np
from motep_original_files.mtp import read_mtp
from motep_original_files.setting import parse_setting
from motep_original_files.calculator import MTP

import jax
import jax.numpy as jnp
from jax import lax
import optax
from functools import partial

# import auxillary functions
from motep_jax_train_import import*
jax.config.update('jax_enable_x64', False)

    

def train(training_cfg, level, steps_opt1, steps_opt2, species=None, pkl_file='jax_images_data'):
    
    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'
        
    images = read_images([training_cfg], species=species)    
    rng = np.random.default_rng(10)
    
    mtp_data = read_mtp(untrained_mtp)
    mtp_data.species = species
    mtp_data.initialize(rng)
        
    mtp_data.radial_coeffs *= 1e-2
    mtp_data.moment_coeffs *= 1e-2
    mtp_data.species_coeffs *= 1e-2
    
    mtp_instance = MTP(mtp_data, engine="jax_new", is_trained=True)
    
    extract_and_save_img_data(images, species, mtp_data)
    jax_images = load_data_pickle(f'{pkl_file}.pkl')
    
    # do I need to pass this as static args??
    prediction_fn = mtp_instance.calculate_jax_new
    num_basis_params = mtp_data.moment_coeffs.shape[0]
    
    
    n_atoms_representative = 2
    num_f_components_per_config = 3 * n_atoms_representative
    num_s_components_per_config = 6
    num_targets_per_config = 1 + num_f_components_per_config + num_s_components_per_config
    training_ids = np.arange(len(jax_images))
    weight_e, weight_f, weight_s = 1.0, 0.01, 0.001
    num_configs = len(jax_images)
   
    
    #@jax.jit
    @partial(jax.jit, static_argnames=("prediction_fn","num_basis_params", "num_targets_per_config",
                                      "num_f_components_per_config", "num_s_components_per_config",
                                      "num_configs"))
    def fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs):
        key = jax.random.PRNGKey(42)
        
        params_pre_lls = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': mtp_data.moment_coeffs
        }
        
        #####
        opt_basis_lls = solve_lls_for_basis(prediction_fn, params_pre_lls, jax_images, training_ids, weight_e, weight_f, weight_s, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, num_configs)
        #####
        
        params = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': opt_basis_lls
        }
        
        
        lr_schedule_opt1 = optax.exponential_decay(
            init_value=1e-1,
            transition_steps=20,
            decay_rate=0.90,
            staircase=True
        )
        
        lr_schedule_opt2 = optax.exponential_decay(
            init_value=1e-2,
            transition_steps=20,
            decay_rate=0.90,
            staircase=True
        )
        

        optimizer1 = optax.novograd(learning_rate=lr_schedule_opt1, weight_decay=0.0)
        optimizer2 = optax.lbfgs(learning_rate=lr_schedule_opt2)
        
        opt1_state = optimizer1.init(params)
        opt2_state = optimizer2.init(params)
    
        state = {'params': params, 'opt_state': {'opt1': opt1_state, 'opt2': opt2_state}, 'key': key}
    
        def loss_function(predictions, real_values, we=1.0, wf=0.01, ws=0.001):
            E_pred = predictions['energy']
            F_pred = predictions['forces']
            sigma_pred = predictions['stress']
            loss_E = we * jnp.sum((E_pred - real_values[0])**2)
            loss_F = wf * jnp.sum((F_pred - real_values[1])**2)
            loss_sigma = ws * jnp.sum((sigma_pred - real_values[2])**2)
            return loss_E + loss_F + loss_sigma
    
        def loss_epoch(params, atoms_ids):
            def predict(atoms_id):
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, atoms_id)
                targets = mtp_instance.calculate_jax_new(
                    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params
                )
                return targets, [E, F, sigma]
            predictions, real_values = jax.vmap(predict)(atoms_ids)
            return loss_function(predictions, real_values)
    
        loss_and_grads = jax.value_and_grad(loss_epoch)
        
        def epoch_step_opt1(carry, step):
            params = carry['params']
            opt1_state = carry['opt_state']['opt1']
            opt2_state = carry['opt_state']['opt2']
            key = carry['key']
    
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, grads = loss_and_grads(params, atoms_ids)
            updates, new_opt1_state = optimizer1.update(grads, opt1_state, params)
            new_state = {'opt1': new_opt1_state, 'opt2': opt2_state}
            new_params = optax.apply_updates(params, updates)
            new_carry = {'params': new_params, 'opt_state': new_state, 'key': key}
            return new_carry, loss
    
        
        def epoch_step_opt2(carry, step):
            params = carry['params']
            opt1_state = carry['opt_state']['opt1']
            opt2_state = carry['opt_state']['opt2']
            key = carry['key']
    
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, grads = loss_and_grads(params, atoms_ids)
            clipped_grads, _ = optax.clip_by_global_norm(0.1).update(grads, None)
            updates, new_opt2_state = optimizer2.update(
                clipped_grads, opt2_state, params,
                value=loss,
                grad=clipped_grads,
                value_fn=lambda p: loss_epoch(p, atoms_ids)
            )
            new_state = {'opt1': opt1_state, 'opt2': new_opt2_state}
            new_params = optax.apply_updates(params, updates)
            new_carry = {'params': new_params, 'opt_state': new_state, 'key': key}
            return new_carry, loss
        
    
        def compute_loss(state):
            params = state['params']
            key = state['key']
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, _ = loss_and_grads(params, atoms_ids)
            return loss, key
        
        ##### opt1 #####
        threshold_1 = 1e-2
        init_loss, new_key = compute_loss(state)
        state = {**state, 'key': new_key}
        init_phase1 = (0, state, init_loss)
    
        def cond_phase1(carry):
            step, state, loss = carry
            return jnp.logical_and(step < steps_opt1, loss > threshold_1)
    
        def body_phase1(carry):
            step, state, loss = carry
            new_state, new_loss = epoch_step_opt1(state, step)
            return (step + 1, new_state, new_loss)
    
        step_phase1, state_phase1, final_loss_phase1 = lax.while_loop(cond_phase1, body_phase1, init_phase1)
        ##########
    
        ##### opt2 #####
        threshold_2 = 1e-6
        init_loss2, new_key2 = compute_loss(state_phase1)
        state_phase1 = {**state_phase1, 'key': new_key2}
        init_phase2 = (0, state_phase1, init_loss2)
    
        def cond_phase2(carry):
            step, state, loss = carry
            return jnp.logical_and(step < steps_opt2, loss > threshold_2)
    
        def body_phase2(carry):
            step, state, loss = carry
            new_state, new_loss = epoch_step_opt2(state, step)
            return (step + 1, new_state, new_loss)
    
        step_phase2, state_phase2, final_loss_phase2 = lax.while_loop(cond_phase2, body_phase2, init_phase2)
        ##########
        
    
        steps_performed = [step_phase1, step_phase2]
    
        return state_phase2, jnp.array([final_loss_phase1, final_loss_phase2]), steps_performed
    
    epoch_carry, epoch_losses, steps_performed = fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs)
    return epoch_carry, epoch_losses, steps_performed



# notes for further steps
# test optax.partition for separately updating params
# test optax.masked for the same reason (might be easier if I really only want to iteratively udapte different subsets of params)


def train_2_lss_lbfgs(training_cfg, level, steps_opt1, species=None, pkl_file='jax_images_data'):
    
    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'
        
    images = read_images([training_cfg], species=species)    
    rng = np.random.default_rng(10)
    
    mtp_data = read_mtp(untrained_mtp)
    mtp_data.species = species
    mtp_data.initialize(rng)
        
    mtp_data.radial_coeffs *= 1e-2
    mtp_data.moment_coeffs *= 1e-2
    mtp_data.species_coeffs *= 1e-2
    
    mtp_instance = MTP(mtp_data, engine="jax_new", is_trained=True)
    
    extract_and_save_img_data(images, species, mtp_data)
    jax_images = load_data_pickle(f'{pkl_file}.pkl')
    
    prediction_fn = mtp_instance.calculate_jax_new
    num_basis_params = mtp_data.moment_coeffs.shape[0]
    n_atoms_representative = 2
    num_f_components_per_config = 3 * n_atoms_representative
    num_s_components_per_config = 6
    num_targets_per_config = 1 + num_f_components_per_config + num_s_components_per_config
    training_ids = np.arange(len(jax_images))
    weight_e, weight_f, weight_s = 1.0, 0.01, 0.001
    num_configs = len(jax_images)
   
    
    @partial(jax.jit, static_argnames=("prediction_fn","num_basis_params", "num_targets_per_config",
                                      "num_f_components_per_config", "num_s_components_per_config",
                                      "num_configs"))
    def fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs):
        key = jax.random.PRNGKey(42)
        
        params_pre_lls = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': mtp_data.moment_coeffs
        }
        
        #####
        opt_basis_lls = solve_lls_for_basis(prediction_fn, params_pre_lls, jax_images, training_ids, weight_e, weight_f, weight_s, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, num_configs)
        #####
        
        params = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': opt_basis_lls
        }
        
        
        lr_schedule_opt1 = optax.exponential_decay(
            init_value=1e-1,
            transition_steps=100,
            decay_rate=0.99,
            staircase=True
        )
        

        optimizer1 = optax.lbfgs(learning_rate=lr_schedule_opt1)
        
        opt1_state = optimizer1.init(params)
    
        state = {'params': params, 'opt_state': {'opt1': opt1_state}, 'key': key}
    
        def loss_function(predictions, real_values, we=1.0, wf=0.01, ws=0.001):
            E_pred = predictions['energy']
            F_pred = predictions['forces']
            sigma_pred = predictions['stress']
            loss_E = we * jnp.sum((E_pred - real_values[0])**2)
            loss_F = wf * jnp.sum((F_pred - real_values[1])**2)
            loss_sigma = ws * jnp.sum((sigma_pred - real_values[2])**2)
            return loss_E + loss_F + loss_sigma
    
        def loss_epoch(params, atoms_ids):
            def predict(atoms_id):
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, atoms_id)
                targets = mtp_instance.calculate_jax_new(
                    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params
                )
                return targets, [E, F, sigma]
            predictions, real_values = jax.vmap(predict)(atoms_ids)
            return loss_function(predictions, real_values)
    
        loss_and_grads = jax.value_and_grad(loss_epoch)
    
        
        def epoch_step_opt1(carry, step):
            params = carry['params']
            opt1_state = carry['opt_state']['opt1']
            key = carry['key']
    
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, grads = loss_and_grads(params, atoms_ids)
            clipped_grads, _ = optax.clip_by_global_norm(0.5).update(grads, None)
            updates, new_opt1_state = optimizer1.update(
                clipped_grads, opt1_state, params,
                value=loss,
                grad=clipped_grads,
                value_fn=lambda p: loss_epoch(p, atoms_ids)
            )
            new_state = {'opt1': opt1_state}
            new_params = optax.apply_updates(params, updates)
            new_carry = {'params': new_params, 'opt_state': new_state, 'key': key}
            return new_carry, loss
        
    
        def compute_loss(state):
            params = state['params']
            key = state['key']
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, _ = loss_and_grads(params, atoms_ids)
            return loss, key
        
        ##### opt1 #####
        threshold_1 = 1e-4
        init_loss, new_key = compute_loss(state)
        state = {**state, 'key': new_key}
        init_phase1 = (0, state, init_loss, jnp.inf)
    
        def cond_phase1(carry):
            step, state, loss, prev_loss = carry
            return jnp.logical_and(step < steps_opt1, loss - prev_loss < threshold_1)
    
        def body_phase1(carry):
            step, state, loss, prev_loss = carry
            new_state, new_loss = epoch_step_opt1(state, step)
            return (step + 1, new_state, new_loss, loss)
    
        step_phase1, state_phase1, final_loss_phase1, second_to_last_loss = lax.while_loop(cond_phase1, body_phase1, init_phase1)
        ##########
    
        steps_performed = [step_phase1]
    
        return state_phase1, jnp.array([init_loss, final_loss_phase1]), steps_performed
    
    epoch_carry, epoch_losses, steps_performed = fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs)
    return epoch_carry, epoch_losses, steps_performed



def train3(training_cfg, level, steps_opt1, steps_opt2, species=None, pkl_file='jax_images_data'):
    
    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'
        
    images = read_images([training_cfg], species=species)    
    rng = np.random.default_rng(10)
    
    mtp_data = read_mtp(untrained_mtp)
    mtp_data.species = species
    mtp_data.initialize(rng)
        
    mtp_data.radial_coeffs *= 1e-2
    mtp_data.moment_coeffs *= 1e-2
    mtp_data.species_coeffs *= 1e-2
    
    mtp_instance = MTP(mtp_data, engine="jax_new", is_trained=True)
    
    extract_and_save_img_data(images, species, mtp_data)
    jax_images = load_data_pickle(f'{pkl_file}.pkl')
    
    # do I need to pass this as static args??
    prediction_fn = mtp_instance.calculate_jax_new
    num_basis_params = mtp_data.moment_coeffs.shape[0]
    
    
    n_atoms_representative = 2
    num_f_components_per_config = 3 * n_atoms_representative
    num_s_components_per_config = 6
    num_targets_per_config = 1 + num_f_components_per_config + num_s_components_per_config
    training_ids = np.arange(len(jax_images))
    weight_e, weight_f, weight_s = 1.0, 0.01, 0.001
    num_configs = len(jax_images)
   
    
    #@jax.jit
    @partial(jax.jit, static_argnames=("prediction_fn","num_basis_params", "num_targets_per_config",
                                      "num_f_components_per_config", "num_s_components_per_config",
                                      "num_configs"))
    def fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs):
        key = jax.random.PRNGKey(42)
        
        params_pre_lls = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': mtp_data.moment_coeffs
        }
        
        #####
        opt_basis_lls = solve_lls_for_basis(prediction_fn, params_pre_lls, jax_images, training_ids, weight_e, weight_f, weight_s, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, num_configs)
        #####
        
        params = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': opt_basis_lls
        }
        
        
        lr_schedule_opt1 = optax.exponential_decay(
            init_value=1e-2,
            transition_steps=20,
            decay_rate=0.99,
            staircase=True
        )
        
        lr_schedule_opt2 = optax.exponential_decay(
            init_value=1e-1,
            transition_steps=20,
            decay_rate=0.95,
            staircase=True
        )
        

        optimizer1 = optax.novograd(learning_rate=lr_schedule_opt1, weight_decay=0.0)
        optimizer2 = optax.lbfgs(learning_rate=lr_schedule_opt2)
        
        opt1_state = optimizer1.init(params)
        opt2_state = optimizer2.init(params)
    
        state = {'params': params, 'opt_state': {'opt1': opt1_state, 'opt2': opt2_state}, 'key': key, 'loss_history': jnp.full(steps_opt1+steps_opt2, jnp.nan), 'steps_offset': 0}
    
        def loss_function(predictions, real_values, we=1.0, wf=0.01, ws=0.001):
            E_pred = predictions['energy']
            F_pred = predictions['forces']
            sigma_pred = predictions['stress']
            loss_E = we * jnp.sum((E_pred - real_values[0])**2)
            loss_F = wf * jnp.sum((F_pred - real_values[1])**2)
            loss_sigma = ws * jnp.sum((sigma_pred - real_values[2])**2)
            return loss_E + loss_F + loss_sigma
    
        def loss_epoch(params, atoms_ids):
            def predict(atoms_id):
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, atoms_id)
                targets = mtp_instance.calculate_jax_new(
                    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params
                )
                return targets, [E, F, sigma]
            predictions, real_values = jax.vmap(predict)(atoms_ids)
            return loss_function(predictions, real_values)
    
        loss_and_grads = jax.value_and_grad(loss_epoch)
        
        def epoch_step_opt1(carry, step):
            params = carry['params']
            opt1_state = carry['opt_state']['opt1']
            key = carry['key']
            loss_history = carry['loss_history']
    
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, grads = loss_and_grads(params, atoms_ids)
            clipped_grads, _ = optax.clip_by_global_norm(0.1).update(grads, None)
            updates, new_opt1_state = optimizer1.update(clipped_grads, opt1_state, params)
            new_params = optax.apply_updates(params, updates)
            new_loss_history = loss_history.at[step].set(loss)
            
            new_carry = carry.copy()
            new_carry.update({
                'params': new_params,
                'opt_state': {**carry['opt_state'], 'opt1': new_opt1_state},
                'key': key,
                'loss_history': new_loss_history
            })
            
            return new_carry, loss
    
        
        def epoch_step_opt2(carry, step):
            params = carry['params']
            opt2_state = carry['opt_state']['opt2']
            key = carry['key']
            loss_history = carry['loss_history']
            offset = carry['step_offset']
    
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, grads = loss_and_grads(params, atoms_ids)
            clipped_grads, _ = optax.clip_by_global_norm(0.1).update(grads, None)
            updates, new_opt2_state = optimizer2.update(
                clipped_grads, opt2_state, params,
                value=loss,
                grad=clipped_grads,
                value_fn=lambda p: loss_epoch(p, atoms_ids)
            )
            new_params = optax.apply_updates(params, updates)
            
            correct_index = offset + step 
            new_loss_history = loss_history.at[correct_index].set(loss)
            
            
            new_carry = carry.copy()
            new_carry.update({
                'params': new_params,
                'opt_state': {**carry['opt_state'], 'opt2': new_opt2_state},
                'key': key,
                'loss_history': new_loss_history
            })
            
            
            return new_carry, loss
        
        # does this actually give me the right loss? Do I not need to compute it using my functions
        def compute_loss(state):
            params = state['params']
            key = state['key']
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss, _ = loss_and_grads(params, atoms_ids)
            return loss, key
        
        # new function for init_loss from state
        def compute_init_loss(state):
            params = state['params']
            key = state['key']
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss = loss_epoch(params, atoms_ids)
            return loss, key
            
        
        ##### opt1 #####
        threshold_1 = 1e-2
        #init_loss, new_key = compute_loss(state)
        init_loss, new_key = compute_init_loss(state)
        jax.debug.print("init_loss: {}", init_loss) 
        state = {**state, 'key': new_key}
        init_phase1 = (0, state, init_loss, jnp.inf)
        
        def cond_phase1(carry):
            step, state, loss, prev_loss = carry
            converged = jnp.logical_and(prev_loss > loss, (prev_loss-loss) <= threshold_1) 
            return jnp.logical_and(step < steps_opt1, jnp.logical_not(converged)) 
    
        def body_phase1(carry):
            step, state, loss, prev_loss = carry
            new_state, new_loss = epoch_step_opt1(state, step)
            return (step + 1, new_state, new_loss, loss)
    
        step_phase1, state_phase1, final_loss_phase1, prev_loss_phase1 = lax.while_loop(cond_phase1, body_phase1, init_phase1)
        ##########
        
        # maybe need to add a condition that I do not converge after less and 5 steps or so?
        # not sure why I sometimes get converged == True right away and other times not?
        # need more debugging
        # but just using lls and lbfgs is the best
        ##### opt2 #####
        state_phase1['step_offset'] = step_phase1 
        threshold_2 = 1e-5
        step_threshold = 20
        #init_loss2, new_key2 = compute_loss(state_phase1)
        init_loss2, new_key2 = compute_init_loss(state_phase1)
        jax.debug.print("init_loss2: {}", init_loss2) 
        state_phase1 = {**state_phase1, 'key': new_key2}
        init_phase2 = (0, state_phase1, init_loss2, jnp.inf)
        
        def cond_phase2(carry):
            step, state, loss, prev_loss = carry
            converged_by_loss = jnp.logical_and(prev_loss > loss, (prev_loss - loss) <= threshold_2) 
            is_less_than_10_steps = (step < step_threshold)
            converged = jnp.where(is_less_than_10_steps, 
                                  jnp.array(False), 
                                  converged_by_loss)
            #jax.debug.print("Step={s}, Loss={l:.4f}, PrevLoss={p:.4f}, LossConverged={lc}, Step<10={s10}, FinalConverged={c}", 
                           # s=step, l=loss, p=prev_loss, lc=converged_by_loss, s10=is_less_than_10_steps, c=converged)
        
            continue_loop = jnp.logical_and(step < steps_opt2, jnp.logical_not(converged))
            return continue_loop
    
        def body_phase2(carry):
            step, state, loss, prev_loss = carry
            #jax.debug.print("loss in body: {}", loss)
            #jax.debug.print("prev_loss in body: {}", prev_loss) 
            new_state, new_loss = epoch_step_opt2(state, step)
            return (step + 1, new_state, new_loss, loss)
    
        step_phase2, state_phase2, final_loss_phase2, prev_loss_phase2 = lax.while_loop(cond_phase2, body_phase2, init_phase2)
        loss_history = state_phase2['loss_history']
        ##########
        
    
        steps_performed = [step_phase1, step_phase2]
    
        return state_phase2, jnp.array([final_loss_phase1, final_loss_phase2]), steps_performed, loss_history
    
    epoch_carry, epoch_losses, steps_performed, loss_history = fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs)
        
    nan_mask = ~np.isnan(loss_history)
    loss_history = loss_history[nan_mask]
    
    print(loss_history[-4:])
    print(len(loss_history))
    
    return epoch_carry, epoch_losses, steps_performed, loss_history


# make sure min dist is not wrong. Might have to add this to the arguments here. Also want to be able to write and read mtp files. 
# changed it for mtp12
def train4(training_cfg, level, steps_lbfgs, threshold_loss, min_steps, lr_start, transition_steps, decay_rate, global_norm_clip, min_dist=0.5, max_dist=5.0, scaling=1.0, species=None, pkl_file='jax_images_data', pkl_file_val='val_jax_images_data'):
    
    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'


    # something wrong here
    images_total = read_images([f'training_data/{training_cfg}'], species=species)    
    data_split = int(3/4*len(images_total))
    images = images_total[0:data_split][0:5]
    images_val = images_total[1-data_split:][0:1]
    
    
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
    
    #print(mtp_data)    
    

    
    mtp_instance = MTP(mtp_data, engine="jax_new", is_trained=True)
    
    extract_and_save_img_data(images, species, mtp_data, name=pkl_file)
    extract_and_save_img_data(images_val, species, mtp_data, name=pkl_file_val)
    
    jax_images = load_data_pickle(f'training_data/{pkl_file}.pkl')
    jax_val_images = load_data_pickle(f'training_data/{pkl_file_val}.pkl')
    
    
    #print(len(jax_images['E']))
    #print(len(jax_val_images['E']))
    
    prediction_fn = mtp_instance.calculate_jax_new
    num_basis_params = mtp_data.moment_coeffs.shape[0]
    n_atoms_representative = int(jax_images['n_atoms'][0])
    num_f_components_per_config = 3 * n_atoms_representative
    num_s_components_per_config = 6
    num_targets_per_config = 1 + num_f_components_per_config + num_s_components_per_config
    training_ids = np.arange(len(jax_images['E']))
    weight_e, weight_f, weight_s = 1.0, 0.01, 0.001
    num_configs = len(jax_images['E'])
    
    #print(n_atoms_representative)
    #print(num_f_components_per_config)
    #print(training_ids)
    #print(num_configs)
    
    @partial(jax.jit, static_argnames=("prediction_fn","num_basis_params", "num_targets_per_config",
                                      "num_f_components_per_config", "num_s_components_per_config",
                                      "num_configs"))
    def fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs):
        
        def loss_function(predictions, real_values, we=1.0, wf=0.01, ws=0.001):
            E_pred = predictions['energy']
            F_pred = predictions['forces']
            sigma_pred = predictions['stress']
            loss_E = we * jnp.sum((E_pred - real_values[0])**2)
            loss_F = wf * jnp.sum((F_pred - real_values[1])**2)
            loss_sigma = ws * jnp.sum((sigma_pred - real_values[2])**2)
            return loss_E + loss_F + loss_sigma
    
        def loss_epoch(params, atoms_ids):
            def predict(atoms_id):
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, atoms_id)
                
                #jax.debug.print("js: {}", all_js.shape)
                #jax.debug.print("rijs: {}", all_rijs.shape)
                #jax.debug.print("jtypes: {}", all_jtypes.shape)

                targets = mtp_instance.calculate_jax_new(
                    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params
                )
                return targets, [E, F, sigma]
            predictions, real_values = jax.vmap(predict)(atoms_ids)
            return loss_function(predictions, real_values)
        
        def loss_epoch_val(params, atoms_ids):
            def predict(atoms_id):
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_val_images, atoms_id)
                targets = mtp_instance.calculate_jax_new(
                    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params
                )
                return targets, [E, F, sigma]
            predictions, real_values = jax.vmap(predict)(atoms_ids)
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
                value_fn=lambda p: loss_epoch(p, atoms_ids)
            )
            
            key, subkey = jax.random.split(key)
            atoms_ids_val = jax.random.permutation(subkey, len(images_val))
            val_loss = loss_epoch_val(params, atoms_ids_val)
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
            
            
            
            #jax.debug.print("updated params at step: {}", step)
            #jax.debug.print("updated params at step: {}", new_loss_history[step])
            
            return new_carry, loss
        
        def compute_init_loss(state):
            params = state['params']
            key = state['key']
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss = loss_epoch(params, atoms_ids)
            return loss, key
        
        
        # start #
        key = jax.random.PRNGKey(42)
        
        params_pre_lls = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': mtp_data.moment_coeffs
        }
        
        atoms_ids = jax.random.permutation(key, len(images))
        initial_loss = loss_epoch(params_pre_lls, atoms_ids)
        #jax.debug.print("initial loss: {}", initial_loss)
        
        
        #####
        opt_basis_lls = solve_lls_for_basis(prediction_fn, params_pre_lls, jax_images, training_ids, weight_e, weight_f, weight_s, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, num_configs)
        #####
        
        params = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': opt_basis_lls
        }
        
        atoms_ids = jax.random.permutation(key, len(images))
        loss_after_lls = loss_epoch(params, atoms_ids)
        #jax.debug.print("loss after lls: {}", loss_after_lls)

        
        lr_schedule_lbfgs = optax.exponential_decay(
            init_value=lr_start,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        
        
        optimizer_lbfgs = optax.lbfgs(learning_rate=lr_schedule_lbfgs)
        opt_state = optimizer_lbfgs.init(params)
    
        state = {'params': params, 'opt_state': opt_state, 'key': key, 'loss_history': jnp.full(steps_lbfgs, jnp.nan), 'val_loss_history': jnp.full(steps_lbfgs, jnp.nan)}
        
        loss_and_grads = jax.value_and_grad(loss_epoch)

        ##### optimization cycle #####
        init_loss, new_key = compute_init_loss(state)
        state = {**state, 'key': new_key}
        init = (0, state, init_loss, jnp.inf)
        
        def cond(carry):
            step, state, loss, prev_loss = carry
            converged_by_loss = jnp.logical_and(prev_loss > loss, (prev_loss - loss) <= threshold_loss) 
            is_less_than_min_steps = (step < min_steps)
            #jax.debug.print("diff: {}", prev_loss - loss)
            #jax.debug.print("converged_by_loss: {}", converged_by_loss)
            #jax.debug.print("less than min step: {}", is_less_than_min_steps)
            converged = jnp.where(is_less_than_min_steps, 
                                  jnp.array(False), 
                                  converged_by_loss)        
            continue_loop = jnp.logical_and(step < steps_lbfgs, jnp.logical_not(converged))
            #jax.debug.print("Continue loop?: {}", continue_loop)
            return continue_loop
    
        def body(carry):
            step, state, loss, prev_loss = carry
            new_state, new_loss = epoch_step_lbfgs(state, step)
            return (step + 1, new_state, new_loss, loss)
    
        step, state, final_loss, prev_loss = lax.while_loop(cond, body, init)
        loss_history = state['loss_history']
        val_loss_history = state['val_loss_history']
        ##########
        
    
        steps_performed = [step]
    
        return state, jnp.array([final_loss]), steps_performed, loss_history, val_loss_history
    
    epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history = fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs)
        
    nan_mask = ~np.isnan(loss_history)
    loss_history = loss_history[nan_mask]
    val_loss_history = val_loss_history[nan_mask]
    
    return epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history

# train4 with mini batch. This should counteract the memory issue
def train5(training_cfg, level, batch_size, steps_lbfgs, threshold_loss, min_steps, lr_start, transition_steps, decay_rate, global_norm_clip, min_dist=0.5, max_dist=5.0, scaling=1.0, species=None, pkl_file='jax_images_data', pkl_file_val='val_jax_images_data'):
    
    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'

    
    # something wrong here
    images_total = read_images([f'training_data/{training_cfg}'], species=species)    
    data_split = int(3/4*len(images_total))
    images = images_total[0:data_split][0:20]
    images_val = images_total[1-data_split:][0:1]
    
    
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
    
    print(mtp_data)
    
    mtp_instance = MTP(mtp_data, engine="jax_new", is_trained=True)
    
    extract_and_save_img_data(images, species, mtp_data, name=pkl_file)
    extract_and_save_img_data(images_val, species, mtp_data, name=pkl_file_val)
    
    jax_images = load_data_pickle(f'training_data/{pkl_file}.pkl')
    jax_val_images = load_data_pickle(f'training_data/{pkl_file_val}.pkl')
    
    print(len(jax_images['E']))
    print(len(jax_val_images['E']))
    
    prediction_fn = mtp_instance.calculate_jax_new
    num_basis_params = mtp_data.moment_coeffs.shape[0]
    n_atoms_representative = int(jax_images['n_atoms'][0])
    num_f_components_per_config = 3 * n_atoms_representative
    num_s_components_per_config = 6
    num_targets_per_config = 1 + num_f_components_per_config + num_s_components_per_config
    training_ids = np.arange(len(jax_images['E']))
    weight_e, weight_f, weight_s = 1.0, 0.01, 0.001
    num_configs = len(jax_images['E'])    
    
    @partial(jax.jit, static_argnames=("prediction_fn","num_basis_params", "num_targets_per_config",
                                      "num_f_components_per_config", "num_s_components_per_config",
                                      "num_configs"))
    def fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs):
        
        def loss_function(predictions, real_values, we=1.0, wf=0.01, ws=0.001):
            E_pred = predictions['energy']
            F_pred = predictions['forces']
            sigma_pred = predictions['stress']
            loss_E = we * jnp.sum((E_pred - real_values[0])**2)
            loss_F = wf * jnp.sum((F_pred - real_values[1])**2)
            loss_sigma = ws * jnp.sum((sigma_pred - real_values[2])**2)
            return loss_E + loss_F + loss_sigma
    
        def loss_batch(params, atoms_ids):
            def predict(atoms_id):
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, atoms_id)
                targets = mtp_instance.calculate_jax_new(
                    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params
                )
                return targets, [E, F, sigma]
            predictions, real_values = jax.vmap(predict)(atoms_ids)
            return loss_function(predictions, real_values)
        
        def loss_epoch_val(params, atoms_ids):
            def predict(atoms_id):
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_val_images, atoms_id)
                targets = mtp_instance.calculate_jax_new(
                    itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params
                )
                return targets, [E, F, sigma]
            predictions, real_values = jax.vmap(predict)(atoms_ids)
            return loss_function(predictions, real_values)
        
        def epoch_step_lbfgs(carry, step): 
            params = carry['params']
            opt_state = carry['opt_state']
            key = carry['key']
            loss_history = carry['loss_history']
            val_loss_history = carry['val_loss_history']
        
            num_samples = len(images) 

            num_batches = (num_samples + batch_size - 1) // batch_size
            num_padding = num_batches * batch_size - num_samples
            padding_value = -1
        
            key, subkey = jax.random.split(key)
            base_indices = np.arange(num_samples)
            shuffled_base_indices = jax.random.permutation(subkey, base_indices)
        
            padding_indices = jnp.full((num_padding,), padding_value, dtype=shuffled_base_indices.dtype)
            padded_shuffled_indices = jnp.concatenate([shuffled_base_indices, padding_indices])
        
            batched_indices = padded_shuffled_indices.reshape((num_batches, batch_size))
        
            def scan_body(carry_scan, x_batch_indices):
                acc_grads, acc_loss_sum = carry_scan
                batch_loss_sum, batch_grads = loss_and_grads(params, x_batch_indices)
                new_acc_grads = jax.tree_map(lambda acc, batch: acc + batch, acc_grads, batch_grads)
                new_acc_loss_sum = acc_loss_sum + batch_loss_sum
                return (new_acc_grads, new_acc_loss_sum), None
        
            zero_grads = jax.tree_map(jnp.zeros_like, params)
            init_carry_scan = (zero_grads, 0.0)
        
            final_carry_scan, _ = lax.scan(scan_body, init_carry_scan, batched_indices)
        
            total_grads_accumulated, total_loss_sum_accumulated = final_carry_scan
        
            avg_loss = total_loss_sum_accumulated / jnp.maximum(num_samples, 1)
            avg_grads = jax.tree_map(lambda g: g / jnp.maximum(num_samples, 1), total_grads_accumulated)
        
            clipped_avg_grads, _ = optax.clip_by_global_norm(global_norm_clip).update(avg_grads, opt_state, params)
        
            def lbfgs_value_fn(p_eval):
                def value_scan_body(carry_loss_sum, x_batch_indices):
                    batch_loss_sum_eval = loss_batch(p_eval, x_batch_indices)
                    return carry_loss_sum + batch_loss_sum_eval, None
        
                total_loss_sum_eval, _ = lax.scan(value_scan_body, 0.0, batched_indices)
                avg_loss_eval = total_loss_sum_eval / jnp.maximum(num_samples, 1)
                return avg_loss_eval
        
            updates, new_opt_state = optimizer_lbfgs.update(
                clipped_avg_grads, opt_state, params,
                value=avg_loss,
                grad=clipped_avg_grads,
                value_fn=lbfgs_value_fn
            )
        
            new_params = optax.apply_updates(params, updates)
            
            key, subkey = jax.random.split(key)
            atoms_ids_val = jax.random.permutation(subkey, len(images_val))
            val_loss = loss_epoch_val(params, atoms_ids_val)
            new_val_loss_history = val_loss_history.at[step].set(val_loss)

            
            new_params = optax.apply_updates(params, updates)
            new_loss_history = loss_history.at[step].set(avg_loss)
            new_carry = carry.copy()
            
            new_carry.update({
                'params': new_params,
                'opt_state': new_opt_state,
                'key': key,
                'loss_history': new_loss_history,
                'val_loss_history': new_val_loss_history
            })
            
            
            #jax.debug.print("updated params at step: {}", step)
            #jax.debug.print("updated params at step: {}", new_loss_history[step])
            
            return new_carry, avg_loss
        
        def compute_init_loss(state):
            params = state['params']
            key = state['key']
            key, subkey = jax.random.split(key)
            atoms_ids = jax.random.permutation(subkey, len(images))
            loss = loss_batch(params, atoms_ids)
            return loss, key
        
        
        # start #
        key = jax.random.PRNGKey(42)
        
        params_pre_lls = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': mtp_data.moment_coeffs
        }
        
        #jax.debug.print("species_params: {}", params_pre_lls['species'])
        
        atoms_ids = jax.random.permutation(key, len(images))
        initial_loss = loss_batch(params_pre_lls, atoms_ids)
        #jax.debug.print("initial loss: {}", initial_loss)
        #jax.debug.print("initial loss: {}", type(initial_loss))
        
        
        
        #####
        opt_basis_lls = solve_lls_for_basis(prediction_fn, params_pre_lls, jax_images, training_ids, weight_e, weight_f, weight_s, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, num_configs)
        #####
        
        params = {
            'species': mtp_data.species_coeffs,
            'radial': mtp_data.radial_coeffs,
            'basis': opt_basis_lls
        }
        
        atoms_ids = jax.random.permutation(key, len(images))
        loss_after_lls = loss_batch(params, atoms_ids)
        #jax.debug.print("loss after lls: {}", loss_after_lls)

        
        lr_schedule_lbfgs = optax.exponential_decay(
            init_value=lr_start,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        
        
        optimizer_lbfgs = optax.lbfgs(learning_rate=lr_schedule_lbfgs)
        opt_state = optimizer_lbfgs.init(params)
    
        state = {'params': params, 'opt_state': opt_state, 'key': key, 'loss_history': jnp.full(steps_lbfgs, jnp.nan), 'val_loss_history': jnp.full(steps_lbfgs, jnp.nan)}
        
        loss_and_grads = jax.value_and_grad(loss_batch)

        ##### optimization cycle #####
        init_loss, new_key = compute_init_loss(state)
        state = {**state, 'key': new_key}
        init = (0, state, init_loss, jnp.inf)
        
        def cond(carry):
            step, state, loss, prev_loss = carry
            converged_by_loss = jnp.logical_and(prev_loss > loss, (prev_loss - loss) <= threshold_loss) 
            is_less_than_min_steps = (step < min_steps)
            #jax.debug.print("diff: {}", prev_loss - loss)
            #jax.debug.print("converged_by_loss: {}", converged_by_loss)
            #jax.debug.print("less than min step: {}", is_less_than_min_steps)
            converged = jnp.where(is_less_than_min_steps, 
                                  jnp.array(False), 
                                  converged_by_loss)        
            continue_loop = jnp.logical_and(step < steps_lbfgs, jnp.logical_not(converged))
            #jax.debug.print("Continue loop?: {}", continue_loop)
            return continue_loop
    
        def body(carry):
            step, state, loss, prev_loss = carry
            new_state, new_loss = epoch_step_lbfgs(state, step)
            return (step + 1, new_state, new_loss, loss)
    
        step, state, final_loss, prev_loss = lax.while_loop(cond, body, init)
        loss_history = state['loss_history']
        val_loss_history = state['val_loss_history']
        ##########
        
    
        steps_performed = [step]
    
        return state, jnp.array([final_loss]), steps_performed, loss_history, val_loss_history
    
    epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history = fit(prediction_fn, num_basis_params, num_targets_per_config, num_f_components_per_config, num_s_components_per_config, training_ids, weight_e, weight_f, weight_s, num_configs)
        
    nan_mask = ~np.isnan(loss_history)
    loss_history = loss_history[nan_mask]
    val_loss_history = val_loss_history[nan_mask]
    
    return epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history


# move to import file and add to train4
def write_mtp_data(training_cfg,species,pkl_loss='jax_images_data',pkl_val_loss='val_jax_images_data'):
    
    images_total = read_images([training_cfg], species=species)    
    data_split = int(3/4*len(images_total))
    images = images_total[0:data_split]
    images_val = images_total[1-data_split:]
    
    extract_and_save_img_data(images, species, mtp_data, name=pkl_loss)
    extract_and_save_img_data(images_val, species, mtp_data, name=pkl_val_loss)
    
    jax_images = load_data_pickle(f'{pkl_loss}.pkl')
    jax_val_images = load_data_pickle(f'{pkl_val_loss}.pkl')
    
    print(f'Data saved at: Loss:{pkl_loss}.pkl and Validation Loss: {pkl_val_loss}.pkl')


def write_mtp_file(level,species,params,file):
    
    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'
    
    rng = np.random.default_rng(10)
    mtp_data = read_mtp(untrained_mtp)
    mtp_data.species = species
    mtp_data.initialize(rng)
    
    mtp_data.species_coeffs = params['species'] 
    mtp_data.moment_coeffs = params['basis']
    mtp_data.radial_coeffs = params['radial']
    
    write_mtp(file, mtp_data)
    
    print(f'MTP saved at: {file}')

    


# later this will have to become a class
# For now I will init a untrained mtp and just set my params
def mtp(cfgs,level,params,min_dist=0.5,max_dist=5.0,scaling=1.0,species=None):
    
    level_str = str(level)
    level_formatted = level_str.zfill(2)
    untrained_mtp = f'untrained_mtps/{level_formatted}.mtp'
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
    
    
    mtp_data.species_coeffs = params['species']
    mtp_data.radial_coeffs = params['radial']
    mtp_data.moment_coeffs = params['basis']
    
    mtp_instance = MTP(mtp_data, engine="jax_new", is_trained=True)
    
    jax_images = load_data_pickle(f'training_data/{cfgs}.pkl') 
    
    params = {
        'species': mtp_data.species_coeffs,
        'radial': mtp_data.radial_coeffs,
        'basis': mtp_data.moment_coeffs
    }
    
    @jax.jit
    def calc(params):
        atoms_ids = jnp.arange(0,len(jax_images))
        def predict(atoms_id):
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(jax_images, atoms_id)
            targets = mtp_instance.calculate_jax_new(
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, params
            )
            return targets, [E,F,sigma]
        
        predictions, real_values = jax.vmap(predict)(atoms_ids)      
        E, F, sigma = predictions['energy'], predictions['forces'], predictions['stress']

        return E, F, sigma, real_values
    
    E, F, sigma, real_values = calc(params)
    
    return E, F, sigma, real_values





