# motep_jax
Training and timing code for the jax implementation of the motep (python MTP) package.<br>
Important packages: Jax, Optax, Pillow


## Example usage for timing
./run_timing.sh or ./run_training.sh (variables inside these files can be changed to adjust levels etc.)

## Preliminary thesis title
Optimized GPU Training For the MTP Machine Learning Interatomic Potential<br>
Comparison Of A GPU Trained, Python Implementation of the MTP Machine Learning Interatomic Potential With The Original MLIP3 Package<br>
Comparing An Inhouse Python Implementation of the Moment Tensor Potential With The Original MLIP3 Package<br>
(probably shouldn't have a abreviation in the title...)<br>

## Large-Scale Simulations with MLIPs on GPUs
LAMMPS runs on GPUs with either the GPU or KOKKOS Package
JAX runs on AMD GPUs with the ROCM installation (check AMD site)
### Plan: 
  - Talk to someone at Hunter and ask if LAMMPS can be installed (need to ask about Docker aswell). 
  - Either have them do it or do it myself and also install JAX using the ROCM method
  - Test a LAMMPS run using an existing EAM potential
  - Write a new LAMMPS Pair Style --> see how MLIP3 did it
  - Use this new Pair Style to run LAMMPS simulations on Hunters APU systems with a JAX-MTP using already trained values
  - If the pair style does not work --> only use EAM
  - MLIP is probaby not compatible at all
  - If everything works out --> test my training on Hunter aswell (unlikely in the given timeframe)
### Notes: 
  - ROCm works with jax and docker.
  - Need to have ROCm GPUs on Hunter, but the implementation should be pretty easy.
  - This means that I can very well use the current jax mtp on Hunter.
  - Still need to make the pair style though!!! (This could be tough maybe)
  - Still not clear if MLIP3 works on APU.
  - First step would be to install LAMMPS with ROCm support on Hunter.
  - Then talk with someone and get my Docker image on hunter. If Docker is not supported I can convert my image using Apptainer and then add it to Hunter using scp.
  - If all of this works, then I "simply" need the custom pairstyle and I can run MLIP large scale md simulations.
  - Also check if MLIP3 might be compatible with the APUs.
### Important sites:
 - https://www.lammps.org/bench.html
 - https://docs.lammps.org/Speed_packages.html
 - https://github.com/jax-ml/jax/blob/main/build/rocm/README.md
 - https://docs.lammps.org/Developer_write_pair.html
 - https://docs.lammps.org/Speed_gpu.html#
 - https://gitlab.com/ivannovikov/interface-lammps-mlip-3/-/blob/master/LAMMPS/USER-MLIP/pair_MLIP.cpp?ref_type=heads



## Notes/ToDo (Project)
- Can reduce memory usage by switching to bf.float16 (might have to do manually)
- Will look for more ways to leverage symmetry and be more memory efficient.
- Speed should not be impacted too much, so I think using the model might still work if the APUs have enough memory!!!
- Added a train5 function which does mini batch learning. Seems to work better at higher levels for both training sets (training.cfg and SubSet.cfg). Is also less memory intensive, which helps avoid memory isses on the gpu. Also switched to jnp.float32 (have to check if this worked or not though). 
- Hyper parameter opt now minimizes the validation loss that I added. Also looking at predictions on non training data. Works really well. Might be able to run hyper param opt on low levels and use values for higher levels.
- Reached the limit of this dataset. Need a new, bigger one!!!!
- Added hyper parameter optimization using optuna. Did initial tests using level 10 mtp. Will take best performers to give new ranges for hyper params and will do a test at level 16 (later on GPU at level 22).
- Create a prediction function (jitted and non-jitted (for single cfgs)). Code already exists, just a few syntax adjustements. Then split data into training and testing and do some runs like this. This way the optimized hyper parameters can be tested more thoroughly. Need good extrapolation!!!!
- Create a file that takes the obtained timing results and adds them all into one PDF file. Should be organized with titles and some extra info. plus with the option to edit and add some comments. (done)
- first real test using train3 on GPU: Got similar timings as for the runs shown in the presentation. Second loss measurement is always very large, however algorithm immediately corrects and moves towards min. Convergence works really well now. Most of the time the algorithm converges before the 20 steps (thus only doing the minimum number of lbfgs steps). Maybe I can thus lower the threshold. Also still need to see how the learning rate affects the results. Used 1e-1, 20, 0.95 for this test. 
- train3 is now a function that performs very well in terms of efficiency. Takes around 20 (min steps) to 40 epochs to converge (check if my condition is rigorous enough). Currently using train3 with 0 opt1 (novograd) steps. I think it makes sense to only use lls into lbfgs. Everything else seems to worsen the lls optimization and then making the convergence slower. Will create a train4 function shortly with all the correct changes. I also added much better visualization of the training process, which helps to optimize the hyper parameters (learning rate, etc.). Will do and save tests in the next few days. 
- Go over notes from presentation  
  Try to copy MLIP optimization (Pretraining (linear bfgs + rescaling)) --> so far giving good results. Will need to do more checks to see if I can add more things.<br>
  Test first with pre trained mlip potential (after analytical part) --> not necessary I think, since I got lls to work.<br>
  Do not optimize everything at once --> see lls<br>
  Start with lbfgs --> have decided to test using only lbfgs from optax.<br>
  Test on larger datasets --> yes, do this as soon as possible.<br>
  Ask Justus people how jax integration works --> open<br>
  Maybe use pretrained low level mtps to train higher level ones (quality scheme??) --> open<br>
- Start to work on a framework for testing extrapolation (needed for the upcoming comparisons to mlip)

