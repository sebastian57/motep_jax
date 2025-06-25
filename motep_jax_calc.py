import jax.experimental
import numpy as np
from motep_original_files.mtp import read_mtp
from motep_original_files.setting import parse_setting
from motep_original_files.calculator import MTP

import jax
import jax.numpy as jnp
from jax import lax
import optax
from functools import partial
import time
import argparse

from motep_jax_train_import import*

   
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_default_matmul_precision', 'tensorfloat32')

class OptimizedMTPLammpsInterface:
    def __init__(self, mtp_file, pkl_file_val):  
        super().__init__()
        self.mtp_file = mtp_file   
        self.jax_val_images = load_data_pickle(f'training_data/{pkl_file_val}.pkl')     
        self._initialize_mtp()


    def _initialize_mtp(self):
        jax.clear_caches()
        self.mtp_data = read_mtp(self.mtp_file)
        self.mtp_data.species = [13, 28]  
        self.mtp_data.species_count = len(self.mtp_data.species)
        self.rcut = self.mtp_data.max_dist
        
        self.params = {
            'species': jnp.array(self.mtp_data.species_coeffs),
            'radial': jnp.array(self.mtp_data.radial_coeffs),
            'basis': jnp.array(self.mtp_data.moment_coeffs)
        }
        self.mtp_instance = MTP(self.mtp_data, engine="jax_new", is_trained=True)

    @partial(jax.jit, static_argnums=(0,1))
    def _calculate_mtp(self, atom_id):
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma = get_data_for_indices(self.jax_val_images, atom_id)
            targets = self.mtp_instance.calculate_jax(
                itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, self.params
            )
            return targets

def main():

    parser = argparse.ArgumentParser(description="Run MTP Jax calc timing.")
    parser.add_argument("--atom_ids", type=int, default=1,
                        help="number of config ids")
    args = parser.parse_args()
    
    mtp_interface = OptimizedMTPLammpsInterface(
        'training_data/Ni3Al-12g.mtp', 
        'val_jax_images_data_subset'
    )

    elapsed_times = []
    for i in range(0, args.atom_ids):
        start_time = time.time()
        targets = mtp_interface._calculate_mtp(i)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)

    
    elapsed_times = np.array(elapsed_times)

    print(f'Compilation time: {elapsed_times[0]}')
    print(f'Mean time after compilation: {np.mean(elapsed_times[1::])}')
    print(f'Total time: {np.sum(elapsed_times)}')


if __name__ == "__main__":
    main()



