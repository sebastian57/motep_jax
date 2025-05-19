import argparse
import time
import numpy as np
from motep_jax_train_function import train, train_2_lss_lbfgs, train3

def main():
    parser = argparse.ArgumentParser(description='Run MTP Jax training timing test.')
    parser.add_argument('--training_cfg', type=str, default=None,
                        help='String of the name of the .cfg file for training')
    parser.add_argument('--level', type=int, default=2,
                        help='Level for training (default: 2)')
    parser.add_argument('--steps1', type=int, default=1000,
                        help='Number of training steps (default: 1000)')
    parser.add_argument('--steps2', type=int, default=100,
                        help='Number of training steps (default: 100)')
    parser.add_argument('--steps3', type=int, default=100,
                        help='Number of training steps (default: 100)')
    parser.add_argument('--species', default=None,
                        help='Species contained in the dataset')
    parser.add_argument('--pkl_file', type=str, default='timing',
                        help='Pickle file for saving jnp atoms data (default: jax_images_data)')
    parser.add_argument('--name', type=str, default='timing',
                        help='Output file name without extension (default: timing)')
    parser.add_argument("--train1", type=str, default='true',
                        help="str (true or false) to decide if train1 is to be used.")
    parser.add_argument("--train2", type=str, default='false',
                        help="str (true or false) to decide if train2 is to be used.")
    parser.add_argument("--train3", type=str, default='false',
                        help="str (true or false) to decide if train2 is to be used.")
    args = parser.parse_args()

    if args.train1 == 'true':
        epoch_carry, epoch_losses, steps_performed = train(args.training_cfg, args.level, args.steps1, args.steps2, args.species, args.pkl_file)
    elif args.train2 == 'true':
        epoch_carry, epoch_losses, steps_performed = train_2_lss_lbfgs(args.training_cfg, args.level, args.steps1, args.species, args.pkl_file) 
    elif args.train3 == 'true':
        epoch_carry, epoch_losses, steps_performed = train3(args.training_cfg, args.level, args.steps1, args.steps2, args.species, args.pkl_file) 
    else:
        print('No training function specified')
    
    moment_coeffs = epoch_carry['params']['basis']
    radial_coeffs = epoch_carry['params']['radial']
    species_coeffs = epoch_carry['params']['species']
    
    output_filename = f'training_results/{args.name}_coeffs'
    
    print(f'Saving padded coefficients to {output_filename}')
    np.savez_compressed(
        output_filename, 
        basis=moment_coeffs,   
        radial=radial_coeffs,  
        species=species_coeffs  
    )

    print('Saving complete.')

if __name__ == '__main__':
    import os
    os.makedirs('training_results', exist_ok=True) 
    main()
