/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "lammps.h"

#include "exceptions.h"
#include "input.h"
#include "library.h"

#include "json.h"

#include <cstdlib>
#include <iostream>  // Added for std::cout
#include <mpi.h>
#include <new>

// import MolSSI Driver Interface library
#if defined(LMP_MDI)
#include <mdi.h>
#endif

using namespace LAMMPS_NS;

// for convenience
static void finalize()
{
  lammps_kokkos_finalize();
  lammps_python_finalize();
  lammps_plugin_finalize();
}

/* ----------------------------------------------------------------------
   main program to drive LAMMPS
------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
  // ✅ CRITICAL: Set JAX GPU environment BEFORE any LAMMPS/Python initialization
  std::cout << "🔧 LAMMPS main.cpp: Setting JAX GPU environment..." << std::endl;
  
  setenv("JAX_PLATFORMS", "cuda", 1);
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false", 1);
  setenv("XLA_FLAGS", "--xla_gpu_autotune_level=4", 1);
  
  // Debug: Verify environment was set
  const char* jax_platforms = getenv("JAX_PLATFORMS");
  const char* cuda_devices = getenv("CUDA_VISIBLE_DEVICES");
  
  std::cout << "✅ JAX_PLATFORMS: " << (jax_platforms ? jax_platforms : "NOT_SET") << std::endl;
  std::cout << "✅ CUDA_VISIBLE_DEVICES: " << (cuda_devices ? cuda_devices : "NOT_SET") << std::endl;
  std::cout << "🚀 Starting LAMMPS with GPU environment pre-configured..." << std::endl;

  MPI_Init(&argc, &argv);
  MPI_Comm lammps_comm = MPI_COMM_WORLD;

#if defined(LMP_MDI)
  // initialize MDI interface, if compiled in

  int mdi_flag;
  if (MDI_Init(&argc, &argv)) MPI_Abort(MPI_COMM_WORLD, 1);
  if (MDI_Initialized(&mdi_flag)) MPI_Abort(MPI_COMM_WORLD, 1);

  // get the MPI communicator that spans all ranks running LAMMPS
  // when using MDI, this may be a subset of MPI_COMM_WORLD

  if (mdi_flag)
    if (MDI_MPI_get_world_comm(&lammps_comm)) MPI_Abort(MPI_COMM_WORLD, 1);
#endif

  try {
    auto *lammps = new LAMMPS(argc, argv, lammps_comm);
    lammps->input->file();
    delete lammps;
  } catch (LAMMPSAbortException &ae) {
    finalize();
    MPI_Abort(ae.get_universe(), 1);
  } catch (LAMMPSException &) {
    finalize();
    MPI_Barrier(lammps_comm);
    MPI_Finalize();
    exit(1);
  } catch (fmt::format_error &fe) {
    fprintf(stderr, "\nfmt::format_error: %s%s\n", fe.what(), utils::errorurl(12).c_str());
    finalize();
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  } catch (json::exception &je) {
    fprintf(stderr, "\nJSON library error %d: %s\n", je.id, je.what());
    finalize();
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  } catch (std::bad_alloc &ae) {
    fprintf(stderr, "C++ memory allocation failed: %s\n", ae.what());
    finalize();
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  } catch (std::exception &e) {
    fprintf(stderr, "Exception: %s\n", e.what());
    finalize();
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
  finalize();
  MPI_Barrier(lammps_comm);
  MPI_Finalize();
  
  std::cout << "✅ LAMMPS completed successfully" << std::endl;
  return 0;
}