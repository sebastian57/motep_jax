/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Direct JAX Implementation - Dynamic Array Sizing
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(jax/mtp_direct,PairJaxMTPDirect);
// clang-format on
#else

#ifndef LMP_PAIR_JAX_MTP_DIRECT_H
#define LMP_PAIR_JAX_MTP_DIRECT_H

#include "pair.h"
#include <Python.h>

namespace LAMMPS_NS {

class PairJaxMTPDirect : public Pair {
 public:
  PairJaxMTPDirect(class LAMMPS *);
  ~PairJaxMTPDirect() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 private:
  char *jax_function_path;     // Path to JAX exported function (.bin file)
  double cutoff;               // Cutoff distance (extracted from JAX potential)
  int max_atoms;               // Maximum atoms per computation (runtime argument)
  int max_neighbors;           // Maximum neighbors per atom (runtime argument)
  
  // Direct JAX objects (NO Python module!)
  PyObject *jax_export_module;   // jax.export module
  PyObject *jax_function;        // Deserialized JAX function
  
  // Pre-allocated NumPy arrays (reused every timestep - KEY OPTIMIZATION)
  PyObject *itypes_array;
  PyObject *all_js_array;
  PyObject *all_rijs_array;
  PyObject *all_jtypes_array;
  PyObject *cell_rank_obj;
  PyObject *volume_obj;
  PyObject *natoms_actual_obj;
  PyObject *nneigh_actual_obj;
  
  // Pre-allocated result extraction
  bool python_initialized;
  
  // CORE CHANGES: Direct JAX initialization and execution
  void init_python_direct();           // NEW: Direct JAX loading
  void cleanup_python();
  void create_reusable_arrays();       // NEW: Pre-allocate arrays once
  void update_array_data(int natoms, int* itypes, int** all_js,
                        double*** all_rijs, int** all_jtypes,
                        int cell_rank, double volume,
                        int natoms_actual, int nneigh_actual);  // NEW: Update array contents
  
  // CORE CHANGES: Direct function call (no module overhead)
  int call_jax_direct(int natoms, int* itypes, int** all_js,
                     double*** all_rijs, int** all_jtypes,
                     int cell_rank, double volume,
                     int natoms_actual, int nneigh_actual,
                     double* energy, double** forces, double* stress);
  
  void allocate();
  
  // NumPy helper functions
  PyObject *create_numpy_array_int32(int *data, int dim1);
  PyObject *create_numpy_array_int32_2d(int **data, int dim1, int dim2);
  PyObject *create_numpy_array_float32_3d(double ***data, int dim1, int dim2, int dim3);
  void extract_numpy_array_float32_2d(PyObject *array, double **data, int dim1, int dim2);
  void extract_numpy_array_float32_1d(PyObject *array, double *data, int dim1);
  
  // Memory view helpers (avoid copying)
  void update_numpy_array_int32(PyObject *array, int *data, int size);
  void update_numpy_array_int32_2d(PyObject *array, int **data, int dim1, int dim2);
  void update_numpy_array_float32_3d(PyObject *array, double ***data, int dim1, int dim2, int dim3);
};

}

#endif
#endif