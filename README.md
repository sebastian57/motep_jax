# motep_jax
Training and timing code for the jax implementation of the motep (python MTP) package.<br>
Important packages: Jax, Optax, Pillow, absl_py, flatbuffers

# motep_comp
Aot compilation scripts and LAMMPS integration files for running large scale simulations efficiently.<br>

Preliminary performance (local computer with a RTX 3060 Ti graphics card):<br>
-------------<br>
Loop time of 8.5092 on 4 procs for 100 steps with 107000 atoms<br>
Performance: 0.102 ns/day, 236.367 hours/ns, 11.752 timesteps/s, 1.257 Matom-step/s<br>
92.2% CPU use with 4 MPI tasks x 1 OpenMP threads<br>
-------------<br>

Considering performance increase when switching to super computer --> production ready large scale (millions of atoms) molecular dynamics simulations

## Example usage for timing
./run_timing.sh or ./run_training.sh (variables inside these files can be changed to adjust levels etc.)


