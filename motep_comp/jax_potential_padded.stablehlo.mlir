module @jit_jax_calc_wrapper attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64xi32>, %arg1: tensor<64x64x3xf32>, %arg2: tensor<64x64xi32>, %arg3: tensor<i32>, %arg4: tensor<f32>) -> (tensor<f32> {jax.result_info = "result[0]"}, tensor<64x3xf32> {jax.result_info = "result[1]"}, tensor<6xf32> {jax.result_info = "result[2]"}) {
    %cst = stablehlo.constant dense<[-2.13442039, -1.42303443]> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<[-5.50651693, -1.68415856, -3.08951521, 1.02423263, 3.47528768, 2.78716612, 2.03157115, 1.29981482, 2.65568686, 5.32603407, -5.462850e+00, -0.0875505656, 0.071106404, 0.23732692, -1.32929623, -1.72219217, -0.085194543, -0.521280527, -1.82596743, 0.310678154, 0.14799346, -3.70192671, -1.76723862, -0.0515363552, 0.137729079, 0.146934927, 0.119335808, -6.055290e-02, -0.191577688]> : tensor<29xf32>
    %cst_1 = stablehlo.constant dense<[0.158076227, 0.157874137, 0.0786165595, 0.121467963, 0.0386004895, 0.00693756947, -0.0238076057, -9.028900e-03]> : tensor<8xf32>
    %cst_2 = stablehlo.constant dense<[0.207484871, 0.373621613, 0.320314109, 0.223365441, 0.106207252, 0.0826742127, 0.0142681589, 0.0080147814]> : tensor<8xf32>
    %cst_3 = stablehlo.constant dense<[-0.335845172, -0.0979904755, -0.365581691, -0.0728622749, -0.120380893, -0.0533700809, -0.013209017, -0.00967365224]> : tensor<8xf32>
    %cst_4 = stablehlo.constant dense<[0.46890083, 0.00592084322, 0.406803608, 0.25960952, 0.176260978, 0.0420474373, -0.019938346, -0.00362472981]> : tensor<8xf32>
    %cst_5 = stablehlo.constant dense<[0.0366196521, 0.167430401, 0.125902072, 0.102618426, 0.0792881399, 0.0655014962, 0.0276192725, 0.00132603035]> : tensor<8xf32>
    %cst_6 = stablehlo.constant dense<[0.278138459, 0.193185762, -0.0735269636, 0.242082402, 0.387782395, 0.137699738, 0.105480149, -0.00986088905]> : tensor<8xf32>
    %cst_7 = stablehlo.constant dense<[0.235930696, 0.00750631187, 0.192909047, 0.0361860842, 0.037753731, -0.0210961122, -0.0184604116, -0.0196542852]> : tensor<8xf32>
    %cst_8 = stablehlo.constant dense<[-0.305040807, -0.41001004, -0.442647487, -0.202947915, -0.112970009, -0.110911243, -0.0633852109, -0.030109996]> : tensor<8xf32>
    %cst_9 = stablehlo.constant dense<[-0.292062312, -0.0371276215, 0.130894676, -0.134774223, -0.447155118, -0.124483153, -0.0968777909, 0.0276620518]> : tensor<8xf32>
    %cst_10 = stablehlo.constant dense<[0.379155278, 0.0627998784, 0.357896984, 0.185332119, 0.168721333, 0.0514729731, 0.0490280353, 0.0207109842]> : tensor<8xf32>
    %cst_11 = stablehlo.constant dense<[-0.0784524157, -0.123475477, -0.163461342, -0.02092731, 0.0264867935, 0.0409899503, 0.0126694916, 0.00346809067]> : tensor<8xf32>
    %cst_12 = stablehlo.constant dense<[-0.0671107322, -0.017523421, -0.0496234894, -0.0298735239, -7.095200e-02, -0.0271951947, -0.0101533374, -0.0131967496]> : tensor<8xf32>
    %c = stablehlo.constant dense<[0, 4, 8, 5, 2, 1]> : tensor<6xi32>
    %0 = stablehlo.broadcast_in_dim %cst_1, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %1 = stablehlo.broadcast_in_dim %cst_2, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %2 = stablehlo.broadcast_in_dim %cst_3, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %3 = stablehlo.concatenate %0, %1, %2, dim = 0 : (tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<3x8xf32>
    %4 = stablehlo.broadcast_in_dim %cst_4, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %5 = stablehlo.broadcast_in_dim %cst_5, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %6 = stablehlo.broadcast_in_dim %cst_6, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %7 = stablehlo.concatenate %4, %5, %6, dim = 0 : (tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<3x8xf32>
    %8 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<3x8xf32>) -> tensor<1x3x8xf32>
    %9 = stablehlo.broadcast_in_dim %7, dims = [1, 2] : (tensor<3x8xf32>) -> tensor<1x3x8xf32>
    %10 = stablehlo.concatenate %8, %9, dim = 0 : (tensor<1x3x8xf32>, tensor<1x3x8xf32>) -> tensor<2x3x8xf32>
    %11 = stablehlo.broadcast_in_dim %cst_7, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %12 = stablehlo.broadcast_in_dim %cst_8, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %13 = stablehlo.broadcast_in_dim %cst_9, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %14 = stablehlo.concatenate %11, %12, %13, dim = 0 : (tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<3x8xf32>
    %15 = stablehlo.broadcast_in_dim %cst_10, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %16 = stablehlo.broadcast_in_dim %cst_11, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %17 = stablehlo.broadcast_in_dim %cst_12, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %18 = stablehlo.concatenate %15, %16, %17, dim = 0 : (tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<3x8xf32>
    %19 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<3x8xf32>) -> tensor<1x3x8xf32>
    %20 = stablehlo.broadcast_in_dim %18, dims = [1, 2] : (tensor<3x8xf32>) -> tensor<1x3x8xf32>
    %21 = stablehlo.concatenate %19, %20, dim = 0 : (tensor<1x3x8xf32>, tensor<1x3x8xf32>) -> tensor<2x3x8xf32>
    %22 = stablehlo.broadcast_in_dim %10, dims = [1, 2, 3] : (tensor<2x3x8xf32>) -> tensor<1x2x3x8xf32>
    %23 = stablehlo.broadcast_in_dim %21, dims = [1, 2, 3] : (tensor<2x3x8xf32>) -> tensor<1x2x3x8xf32>
    %24 = stablehlo.concatenate %22, %23, dim = 0 : (tensor<1x2x3x8xf32>, tensor<1x2x3x8xf32>) -> tensor<2x2x3x8xf32>
    %cst_13 = stablehlo.constant dense<0.0856752247> : tensor<f32>
    %cst_14 = stablehlo.constant dense<1.81088519> : tensor<f32>
    %cst_15 = stablehlo.constant dense<5.000000e+00> : tensor<f32>
    %25 = call @_jax_calc_local_energy(%arg1, %arg0, %arg2, %cst, %cst_0, %24, %cst_13, %cst_14, %cst_15) : (tensor<64x64x3xf32>, tensor<64xi32>, tensor<64x64xi32>, tensor<2xf32>, tensor<29xf32>, tensor<2x2x3x8xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<64xf32>
    %26:12 = call @_jax_calc_local_energy_1(%arg1, %arg0, %arg2, %24, %cst_13, %cst_14, %cst_15) : (tensor<64x64x3xf32>, tensor<64xi32>, tensor<64x64xi32>, tensor<2x2x3x8xf32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<64x64xf32>, tensor<64x3x64xf32>, tensor<64x64x3x8xf32>, tensor<64x64xf32>, tensor<f32>, tensor<64x64xf32>, tensor<6x64x64xf32>, tensor<64x64xf32>, tensor<64x64x3xf32>, tensor<f32>, tensor<64x64xf32>, tensor<64x64xi1>)
    %27 = stablehlo.iota dim = 0 : tensor<1x1xi32>
    %28 = stablehlo.iota dim = 1 : tensor<1x1xi32>
    %c_16 = stablehlo.constant dense<0> : tensor<i32>
    %29 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
    %30 = stablehlo.add %27, %29 : tensor<1x1xi32>
    %31 = stablehlo.compare  EQ, %30, %28,  SIGNED : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi1>
    %32 = stablehlo.convert %31 : (tensor<1x1xi1>) -> tensor<1x1xf32>
    %33 = stablehlo.slice %32 [0:1, 0:1] : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %34 = stablehlo.reshape %33 : (tensor<1x1xf32>) -> tensor<1xf32>
    %35 = call @_jax_calc_local_energy_7(%cst_0, %arg1, %26#0, %26#1, %26#2, %26#3, %26#4, %26#5, %26#6, %26#7, %26#8, %26#9, %26#10, %26#11, %34) : (tensor<29xf32>, tensor<64x64x3xf32>, tensor<64x64xf32>, tensor<64x3x64xf32>, tensor<64x64x3x8xf32>, tensor<64x64xf32>, tensor<f32>, tensor<64x64xf32>, tensor<6x64x64xf32>, tensor<64x64xf32>, tensor<64x64x3xf32>, tensor<f32>, tensor<64x64xf32>, tensor<64x64xi1>, tensor<1xf32>) -> tensor<64x1x64x3xf32>
    %36 = stablehlo.slice %35 [0:64, 0:1, 0:64, 0:3] : (tensor<64x1x64x3xf32>) -> tensor<64x1x64x3xf32>
    %37 = stablehlo.reshape %36 : (tensor<64x1x64x3xf32>) -> tensor<64x64x3xf32>
    %38 = stablehlo.broadcast_in_dim %25, dims = [0] : (tensor<64xf32>) -> tensor<64x64xf32>
    %39 = stablehlo.transpose %arg1, dims = [0, 2, 1] : (tensor<64x64x3xf32>) -> tensor<64x3x64xf32>
    %40 = stablehlo.dot_general %39, %37, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3x64xf32>, tensor<64x64x3xf32>) -> tensor<64x3x3xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %41 = stablehlo.reduce(%40 init: %cst_17) applies stablehlo.add across dimensions = [0] : (tensor<64x3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
    %c_18 = stablehlo.constant dense<3> : tensor<i32>
    %42 = stablehlo.compare  EQ, %arg3, %c_18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %43 = stablehlo.convert %42 : (tensor<i1>) -> tensor<i32>
    %44 = "stablehlo.case"(%43) ({
      %cst_21 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %47 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<6xf32>
      stablehlo.return %47 : tensor<6xf32>
    }, {
      %47 = stablehlo.transpose %41, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
      %48 = stablehlo.add %41, %47 : tensor<3x3xf32>
      %cst_21 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
      %49 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
      %50 = stablehlo.multiply %48, %49 : tensor<3x3xf32>
      %51 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
      %52 = stablehlo.divide %50, %51 : tensor<3x3xf32>
      %53 = stablehlo.reshape %52 : (tensor<3x3xf32>) -> tensor<9xf32>
      %c_22 = stablehlo.constant dense<0> : tensor<i32>
      %54 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i32>) -> tensor<6xi32>
      %55 = stablehlo.compare  LT, %c, %54,  SIGNED : (tensor<6xi32>, tensor<6xi32>) -> tensor<6xi1>
      %c_23 = stablehlo.constant dense<9> : tensor<i32>
      %56 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i32>) -> tensor<6xi32>
      %57 = stablehlo.add %c, %56 : tensor<6xi32>
      %58 = stablehlo.select %55, %57, %c : tensor<6xi1>, tensor<6xi32>
      %59 = stablehlo.broadcast_in_dim %58, dims = [0] : (tensor<6xi32>) -> tensor<6x1xi32>
      %60 = "stablehlo.gather"(%53, %59) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf32>, tensor<6x1xi32>) -> tensor<6xf32>
      stablehlo.return %60 : tensor<6xf32>
    }) : (tensor<i32>) -> tensor<6xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.reduce(%37 init: %cst_19) applies stablehlo.add across dimensions = [1] : (tensor<64x64x3xf32>, tensor<f32>) -> tensor<64x3xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.reduce(%38 init: %cst_20) applies stablehlo.add across dimensions = [0, 1] : (tensor<64x64xf32>, tensor<f32>) -> tensor<f32>
    return %46, %45, %44 : tensor<f32>, tensor<64x3xf32>, tensor<6xf32>
  }
  func.func private @_jax_calc_local_energy(%arg0: tensor<64x64x3xf32>, %arg1: tensor<64xi32>, %arg2: tensor<64x64xi32>, %arg3: tensor<2xf32>, %arg4: tensor<29xf32>, %arg5: tensor<2x2x3x8xf32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<f32>) -> tensor<64xf32> {
    %0 = call @norm(%arg0) : (tensor<64x64x3xf32>) -> tensor<64x64xf32>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %2 = stablehlo.multiply %1, %0 : tensor<64x64xf32>
    %3 = stablehlo.add %arg7, %arg8 : tensor<f32>
    %4 = stablehlo.convert %3 : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %6 = stablehlo.subtract %2, %5 : tensor<64x64xf32>
    %7 = stablehlo.subtract %arg8, %arg7 : tensor<f32>
    %8 = stablehlo.convert %7 : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %10 = stablehlo.divide %6, %9 : tensor<64x64xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<64xf32>) -> tensor<64x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x64x64xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %14:5 = stablehlo.while(%iterArg = %10, %iterArg_6 = %c, %iterArg_7 = %12, %iterArg_8 = %10, %iterArg_9 = %13) : tensor<64x64xf32>, tensor<i32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<6x64x64xf32>
     cond {
      %c_10 = stablehlo.constant dense<6> : tensor<i32>
      %275 = stablehlo.compare  LT, %iterArg_6, %c_10,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %275 : tensor<i1>
    } do {
      %275:3 = func.call @None(%iterArg, %iterArg_7, %iterArg_8) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
      %276 = stablehlo.broadcast_in_dim %275#2, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
      %c_10 = stablehlo.constant dense<0> : tensor<i32>
      %277 = stablehlo.dynamic_update_slice %iterArg_9, %276, %iterArg_6, %c_10, %c_10 : (tensor<6x64x64xf32>, tensor<1x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x64x64xf32>
      %c_11 = stablehlo.constant dense<1> : tensor<i32>
      %278 = stablehlo.add %iterArg_6, %c_11 : tensor<i32>
      stablehlo.return %iterArg, %278, %275#0, %275#1, %277 : tensor<64x64xf32>, tensor<i32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<6x64x64xf32>
    }
    %15 = stablehlo.slice %14#4 [0:1, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %16 = stablehlo.transpose %15, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %17 = stablehlo.reshape %16 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %18 = stablehlo.slice %14#4 [1:2, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %19 = stablehlo.transpose %18, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %20 = stablehlo.reshape %19 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %21 = stablehlo.slice %14#4 [2:3, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %22 = stablehlo.transpose %21, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %23 = stablehlo.reshape %22 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %24 = stablehlo.slice %14#4 [3:4, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %25 = stablehlo.transpose %24, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %26 = stablehlo.reshape %25 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %27 = stablehlo.slice %14#4 [4:5, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %28 = stablehlo.transpose %27, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %29 = stablehlo.reshape %28 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %30 = stablehlo.slice %14#4 [5:6, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %31 = stablehlo.transpose %30, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %32 = stablehlo.reshape %31 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %33 = call @atleast_2d(%11) : (tensor<64xf32>) -> tensor<1x64xf32>
    %34 = stablehlo.transpose %33, dims = [1, 0] : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %35 = call @atleast_2d_0(%10) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %36 = stablehlo.transpose %35, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %37 = call @atleast_2d_0(%17) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %38 = stablehlo.transpose %37, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %39 = call @atleast_2d_0(%20) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %40 = stablehlo.transpose %39, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %41 = call @atleast_2d_0(%23) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %42 = stablehlo.transpose %41, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %43 = call @atleast_2d_0(%26) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %44 = stablehlo.transpose %43, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %45 = call @atleast_2d_0(%29) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %46 = stablehlo.transpose %45, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %47 = call @atleast_2d_0(%32) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %48 = stablehlo.transpose %47, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %49 = stablehlo.broadcast_in_dim %34, dims = [1, 2] : (tensor<64x1xf32>) -> tensor<64x64x1xf32>
    %50 = stablehlo.concatenate %49, %36, %38, %40, %42, %44, %46, %48, dim = 2 : (tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>) -> tensor<64x64x8xf32>
    %51 = stablehlo.convert %arg8 : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %53 = stablehlo.compare  LT, %0, %52,  FLOAT : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xi1>
    %54 = stablehlo.convert %arg8 : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %56 = stablehlo.subtract %55, %0 : tensor<64x64xf32>
    %57 = stablehlo.multiply %56, %56 : tensor<64x64xf32>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %58 = call @_where(%53, %57, %c_2) : (tensor<64x64xi1>, tensor<64x64xf32>, tensor<i32>) -> tensor<64x64xf32>
    %59 = stablehlo.convert %arg6 : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %61 = stablehlo.multiply %60, %58 : tensor<64x64xf32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %62 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %63 = stablehlo.compare  LT, %arg1, %62,  SIGNED : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi1>
    %c_4 = stablehlo.constant dense<2> : tensor<i32>
    %64 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %65 = stablehlo.add %arg1, %64 : tensor<64xi32>
    %66 = stablehlo.select %63, %65, %arg1 : tensor<64xi1>, tensor<64xi32>
    %67 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<64x64xi32>
    %68 = stablehlo.compare  LT, %arg2, %67,  SIGNED : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
    %69 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<64x64xi32>
    %70 = stablehlo.add %arg2, %69 : tensor<64x64xi32>
    %71 = stablehlo.select %68, %70, %arg2 : tensor<64x64xi1>, tensor<64x64xi32>
    %72 = stablehlo.broadcast_in_dim %66, dims = [0] : (tensor<64xi32>) -> tensor<64x64xi32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1] : (tensor<64x64xi32>) -> tensor<64x64x1xi32>
    %74 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<64x64xi32>) -> tensor<64x64x1xi32>
    %75 = stablehlo.concatenate %73, %74, dim = 2 : (tensor<64x64x1xi32>, tensor<64x64x1xi32>) -> tensor<64x64x2xi32>
    %76 = "stablehlo.gather"(%arg5, %75) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 3, 8>}> : (tensor<2x2x3x8xf32>, tensor<64x64x2xi32>) -> tensor<64x64x3x8xf32>
    %77 = stablehlo.dot_general %76, %50, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x8xf32>, tensor<64x64x8xf32>) -> tensor<64x64x3xf32>
    %78 = stablehlo.dot_general %77, %61, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x64xf32>) -> tensor<64x64x3xf32>
    %79 = stablehlo.transpose %78, dims = [0, 2, 1] : (tensor<64x64x3xf32>) -> tensor<64x3x64xf32>
    %80 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<64x64x3xf32>) -> tensor<64x3x64xf32>
    %81 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2] : (tensor<64x1x64xf32>) -> tensor<64x3x64xf32>
    %83 = stablehlo.divide %80, %82 : tensor<64x3x64xf32>
    %84 = stablehlo.transpose %83, dims = [0, 2, 1] : (tensor<64x3x64xf32>) -> tensor<64x64x3xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %85 = stablehlo.reduce(%79 init: %cst_5) applies stablehlo.add across dimensions = [2] : (tensor<64x3x64xf32>, tensor<f32>) -> tensor<64x3xf32>
    %86 = stablehlo.slice %85 [0:64, 0:1] : (tensor<64x3xf32>) -> tensor<64x1xf32>
    %87 = stablehlo.reshape %86 : (tensor<64x1xf32>) -> tensor<64xf32>
    %88 = stablehlo.slice %85 [0:64, 1:2] : (tensor<64x3xf32>) -> tensor<64x1xf32>
    %89 = stablehlo.reshape %88 : (tensor<64x1xf32>) -> tensor<64xf32>
    %90 = stablehlo.slice %85 [0:64, 2:3] : (tensor<64x3xf32>) -> tensor<64x1xf32>
    %91 = stablehlo.reshape %90 : (tensor<64x1xf32>) -> tensor<64xf32>
    %92 = stablehlo.dot_general %79, %84, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3x64xf32>, tensor<64x64x3xf32>) -> tensor<64x3x3xf32>
    %93 = stablehlo.slice %92 [0:64, 0:1, 0:3] : (tensor<64x3x3xf32>) -> tensor<64x1x3xf32>
    %94 = stablehlo.reshape %93 : (tensor<64x1x3xf32>) -> tensor<64x3xf32>
    %95 = stablehlo.slice %92 [0:64, 1:2, 0:3] : (tensor<64x3x3xf32>) -> tensor<64x1x3xf32>
    %96 = stablehlo.reshape %95 : (tensor<64x1x3xf32>) -> tensor<64x3xf32>
    %97 = stablehlo.dot_general %84, %79, batching_dims = [0, 1] x [0, 2], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x3x3xf32>
    %98 = stablehlo.dot_general %97, %84, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x3x3x3xf32>
    %99 = stablehlo.transpose %98, dims = [0, 2, 1, 3] : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xf32>
    %100 = stablehlo.slice %99 [0:64, 0:1, 0:3, 0:3] : (tensor<64x3x3x3xf32>) -> tensor<64x1x3x3xf32>
    %101 = stablehlo.reshape %100 : (tensor<64x1x3x3xf32>) -> tensor<64x3x3xf32>
    %102 = stablehlo.slice %99 [0:64, 1:2, 0:3, 0:3] : (tensor<64x3x3x3xf32>) -> tensor<64x1x3x3xf32>
    %103 = stablehlo.reshape %102 : (tensor<64x1x3x3xf32>) -> tensor<64x3x3xf32>
    %104 = stablehlo.dot_general %84, %79, batching_dims = [0, 1] x [0, 2], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x3x3xf32>
    %105 = stablehlo.dot_general %84, %84, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x3x3xf32>
    %106 = stablehlo.dot_general %105, %104, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x3x3x3x3xf32>
    %107 = stablehlo.transpose %106, dims = [0, 4, 3, 2, 1] : (tensor<64x3x3x3x3xf32>) -> tensor<64x3x3x3x3xf32>
    %108 = stablehlo.slice %107 [0:64, 0:1, 0:3, 0:3, 0:3] : (tensor<64x3x3x3x3xf32>) -> tensor<64x1x3x3x3xf32>
    %109 = stablehlo.reshape %108 : (tensor<64x1x3x3x3xf32>) -> tensor<64x3x3x3xf32>
    %110 = stablehlo.dot_general %84, %79, batching_dims = [0, 1] x [0, 2], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x3x3xf32>
    %111 = stablehlo.dot_general %84, %84, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x3x3xf32>
    %112 = stablehlo.dot_general %110, %84, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x3x3x3xf32>
    %113 = stablehlo.dot_general %112, %111, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x3x3x3x3x3xf32>
    %114 = stablehlo.transpose %113, dims = [0, 2, 1, 5, 4, 3] : (tensor<64x3x3x3x3x3xf32>) -> tensor<64x3x3x3x3x3xf32>
    %115 = stablehlo.slice %114 [0:64, 0:1, 0:3, 0:3, 0:3, 0:3] : (tensor<64x3x3x3x3x3xf32>) -> tensor<64x1x3x3x3x3xf32>
    %116 = stablehlo.reshape %115 : (tensor<64x1x3x3x3x3xf32>) -> tensor<64x3x3x3x3xf32>
    %117 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %118 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %119 = stablehlo.dot_general %117, %118, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %120 = stablehlo.convert %119 : (tensor<64xbf16>) -> tensor<64xf32>
    %121 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %122 = stablehlo.convert %89 : (tensor<64xf32>) -> tensor<64xbf16>
    %123 = stablehlo.dot_general %121, %122, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %124 = stablehlo.convert %123 : (tensor<64xbf16>) -> tensor<64xf32>
    %125 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %126 = stablehlo.convert %91 : (tensor<64xf32>) -> tensor<64xbf16>
    %127 = stablehlo.dot_general %125, %126, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %128 = stablehlo.convert %127 : (tensor<64xbf16>) -> tensor<64xf32>
    %129 = stablehlo.convert %94 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %130 = stablehlo.convert %94 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %131 = stablehlo.dot_general %129, %130, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3xbf16>) -> tensor<64xbf16>
    %132 = stablehlo.convert %131 : (tensor<64xbf16>) -> tensor<64xf32>
    %133 = stablehlo.convert %94 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %134 = stablehlo.convert %96 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %135 = stablehlo.dot_general %133, %134, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3xbf16>) -> tensor<64xbf16>
    %136 = stablehlo.convert %135 : (tensor<64xbf16>) -> tensor<64xf32>
    %137 = stablehlo.convert %101 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %138 = stablehlo.convert %101 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %139 = stablehlo.dot_general %137, %138, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64xbf16>
    %140 = stablehlo.convert %139 : (tensor<64xbf16>) -> tensor<64xf32>
    %141 = stablehlo.convert %101 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %142 = stablehlo.convert %103 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %143 = stablehlo.dot_general %141, %142, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64xbf16>
    %144 = stablehlo.convert %143 : (tensor<64xbf16>) -> tensor<64xf32>
    %145 = stablehlo.convert %109 : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xbf16>
    %146 = stablehlo.convert %109 : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xbf16>
    %147 = stablehlo.dot_general %145, %146, batching_dims = [0] x [0], contracting_dims = [1, 2, 3] x [1, 2, 3], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3x3xbf16>, tensor<64x3x3x3xbf16>) -> tensor<64xbf16>
    %148 = stablehlo.convert %147 : (tensor<64xbf16>) -> tensor<64xf32>
    %149 = stablehlo.convert %116 : (tensor<64x3x3x3x3xf32>) -> tensor<64x3x3x3x3xbf16>
    %150 = stablehlo.convert %116 : (tensor<64x3x3x3x3xf32>) -> tensor<64x3x3x3x3xbf16>
    %151 = stablehlo.dot_general %149, %150, batching_dims = [0] x [0], contracting_dims = [1, 2, 3, 4] x [1, 2, 3, 4], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3x3x3xbf16>, tensor<64x3x3x3x3xbf16>) -> tensor<64xbf16>
    %152 = stablehlo.convert %151 : (tensor<64xbf16>) -> tensor<64xf32>
    %153 = stablehlo.convert %89 : (tensor<64xf32>) -> tensor<64xbf16>
    %154 = stablehlo.convert %89 : (tensor<64xf32>) -> tensor<64xbf16>
    %155 = stablehlo.dot_general %153, %154, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %156 = stablehlo.convert %155 : (tensor<64xbf16>) -> tensor<64xf32>
    %157 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %158 = stablehlo.convert %120 : (tensor<64xf32>) -> tensor<64xbf16>
    %159 = stablehlo.dot_general %157, %158, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %160 = stablehlo.convert %159 : (tensor<64xbf16>) -> tensor<64xf32>
    %161 = stablehlo.convert %89 : (tensor<64xf32>) -> tensor<64xbf16>
    %162 = stablehlo.convert %120 : (tensor<64xf32>) -> tensor<64xbf16>
    %163 = stablehlo.dot_general %161, %162, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %164 = stablehlo.convert %163 : (tensor<64xbf16>) -> tensor<64xf32>
    %165 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %166 = stablehlo.convert %132 : (tensor<64xf32>) -> tensor<64xbf16>
    %167 = stablehlo.dot_general %165, %166, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %168 = stablehlo.convert %167 : (tensor<64xbf16>) -> tensor<64xf32>
    %169 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %170 = stablehlo.convert %136 : (tensor<64xf32>) -> tensor<64xbf16>
    %171 = stablehlo.dot_general %169, %170, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %172 = stablehlo.convert %171 : (tensor<64xbf16>) -> tensor<64xf32>
    %173 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %174 = stablehlo.convert %140 : (tensor<64xf32>) -> tensor<64xbf16>
    %175 = stablehlo.dot_general %173, %174, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %176 = stablehlo.convert %175 : (tensor<64xbf16>) -> tensor<64xf32>
    %177 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %178 = stablehlo.convert %148 : (tensor<64xf32>) -> tensor<64xbf16>
    %179 = stablehlo.dot_general %177, %178, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %180 = stablehlo.convert %179 : (tensor<64xbf16>) -> tensor<64xf32>
    %181 = stablehlo.convert %94 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %182 = stablehlo.convert %101 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %183 = stablehlo.dot_general %181, %182, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64x3xbf16>
    %184 = stablehlo.convert %183 : (tensor<64x3xbf16>) -> tensor<64x3xf32>
    %185 = stablehlo.convert %94 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %186 = stablehlo.convert %184 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %187 = stablehlo.dot_general %185, %186, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3xbf16>) -> tensor<64xbf16>
    %188 = stablehlo.convert %187 : (tensor<64xbf16>) -> tensor<64xf32>
    %189 = stablehlo.convert %89 : (tensor<64xf32>) -> tensor<64xbf16>
    %190 = stablehlo.convert %132 : (tensor<64xf32>) -> tensor<64xbf16>
    %191 = stablehlo.dot_general %189, %190, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %192 = stablehlo.convert %191 : (tensor<64xbf16>) -> tensor<64xf32>
    %193 = stablehlo.convert %101 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %194 = stablehlo.convert %109 : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xbf16>
    %195 = stablehlo.dot_general %193, %194, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3xbf16>, tensor<64x3x3x3xbf16>) -> tensor<64x3xbf16>
    %196 = stablehlo.convert %195 : (tensor<64x3xbf16>) -> tensor<64x3xf32>
    %197 = stablehlo.convert %94 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %198 = stablehlo.convert %196 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %199 = stablehlo.dot_general %197, %198, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3xbf16>) -> tensor<64xbf16>
    %200 = stablehlo.convert %199 : (tensor<64xbf16>) -> tensor<64xf32>
    %201 = stablehlo.convert %101 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %202 = stablehlo.convert %101 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %203 = stablehlo.dot_general %201, %202, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64x3x3xbf16>
    %204 = stablehlo.convert %203 : (tensor<64x3x3xbf16>) -> tensor<64x3x3xf32>
    %205 = stablehlo.convert %101 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %206 = stablehlo.convert %204 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %207 = stablehlo.dot_general %205, %206, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64xbf16>
    %208 = stablehlo.convert %207 : (tensor<64xbf16>) -> tensor<64xf32>
    %209 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %210 = stablehlo.convert %160 : (tensor<64xf32>) -> tensor<64xbf16>
    %211 = stablehlo.dot_general %209, %210, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %212 = stablehlo.convert %211 : (tensor<64xbf16>) -> tensor<64xf32>
    %213 = stablehlo.convert %89 : (tensor<64xf32>) -> tensor<64xbf16>
    %214 = stablehlo.convert %160 : (tensor<64xf32>) -> tensor<64xbf16>
    %215 = stablehlo.dot_general %213, %214, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %216 = stablehlo.convert %215 : (tensor<64xbf16>) -> tensor<64xf32>
    %217 = stablehlo.convert %132 : (tensor<64xf32>) -> tensor<64xbf16>
    %218 = stablehlo.convert %120 : (tensor<64xf32>) -> tensor<64xbf16>
    %219 = stablehlo.dot_general %217, %218, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %220 = stablehlo.convert %219 : (tensor<64xbf16>) -> tensor<64xf32>
    %221 = stablehlo.convert %140 : (tensor<64xf32>) -> tensor<64xbf16>
    %222 = stablehlo.convert %120 : (tensor<64xf32>) -> tensor<64xbf16>
    %223 = stablehlo.dot_general %221, %222, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %224 = stablehlo.convert %223 : (tensor<64xbf16>) -> tensor<64xf32>
    %225 = stablehlo.convert %87 : (tensor<64xf32>) -> tensor<64xbf16>
    %226 = stablehlo.convert %188 : (tensor<64xf32>) -> tensor<64xbf16>
    %227 = stablehlo.dot_general %225, %226, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %228 = stablehlo.convert %227 : (tensor<64xbf16>) -> tensor<64xf32>
    %229 = stablehlo.convert %132 : (tensor<64xf32>) -> tensor<64xbf16>
    %230 = stablehlo.convert %132 : (tensor<64xf32>) -> tensor<64xbf16>
    %231 = stablehlo.dot_general %229, %230, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %232 = stablehlo.convert %231 : (tensor<64xbf16>) -> tensor<64xf32>
    %233 = stablehlo.broadcast_in_dim %87, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %234 = stablehlo.broadcast_in_dim %89, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %235 = stablehlo.broadcast_in_dim %91, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %236 = stablehlo.broadcast_in_dim %120, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %237 = stablehlo.broadcast_in_dim %124, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %238 = stablehlo.broadcast_in_dim %128, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %239 = stablehlo.broadcast_in_dim %132, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %240 = stablehlo.broadcast_in_dim %136, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %241 = stablehlo.broadcast_in_dim %140, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %242 = stablehlo.broadcast_in_dim %144, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %243 = stablehlo.broadcast_in_dim %148, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %244 = stablehlo.broadcast_in_dim %152, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %245 = stablehlo.broadcast_in_dim %156, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %246 = stablehlo.broadcast_in_dim %160, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %247 = stablehlo.broadcast_in_dim %164, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %248 = stablehlo.broadcast_in_dim %168, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %249 = stablehlo.broadcast_in_dim %172, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %250 = stablehlo.broadcast_in_dim %176, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %251 = stablehlo.broadcast_in_dim %180, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %252 = stablehlo.broadcast_in_dim %188, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %253 = stablehlo.broadcast_in_dim %192, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %254 = stablehlo.broadcast_in_dim %200, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %255 = stablehlo.broadcast_in_dim %208, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %256 = stablehlo.broadcast_in_dim %212, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %257 = stablehlo.broadcast_in_dim %216, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %258 = stablehlo.broadcast_in_dim %220, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %259 = stablehlo.broadcast_in_dim %224, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %260 = stablehlo.broadcast_in_dim %228, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %261 = stablehlo.broadcast_in_dim %232, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %262 = stablehlo.concatenate %233, %234, %235, %236, %237, %238, %239, %240, %241, %242, %243, %244, %245, %246, %247, %248, dim = 1 : (tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>) -> tensor<64x16xf32>
    %263 = stablehlo.concatenate %249, %250, %251, %252, %253, %254, %255, %256, %257, %258, %259, %260, %261, dim = 1 : (tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x1xf32>) -> tensor<64x13xf32>
    %264 = stablehlo.concatenate %262, %263, dim = 1 : (tensor<64x16xf32>, tensor<64x13xf32>) -> tensor<64x29xf32>
    %265 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %266 = stablehlo.compare  LT, %arg1, %265,  SIGNED : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi1>
    %267 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %268 = stablehlo.add %arg1, %267 : tensor<64xi32>
    %269 = stablehlo.select %266, %268, %arg1 : tensor<64xi1>, tensor<64xi32>
    %270 = stablehlo.broadcast_in_dim %269, dims = [0] : (tensor<64xi32>) -> tensor<64x1xi32>
    %271 = "stablehlo.gather"(%arg3, %270) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<2xf32>, tensor<64x1xi32>) -> tensor<64x1xf32>
    %272 = stablehlo.reshape %271 : (tensor<64x1xf32>) -> tensor<64xf32>
    %273 = stablehlo.dot_general %arg4, %264, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<29xf32>, tensor<64x29xf32>) -> tensor<64xf32>
    %274 = stablehlo.add %272, %273 : tensor<64xf32>
    return %274 : tensor<64xf32>
  }
  func.func private @norm(%arg0: tensor<64x64x3xf32>) -> tensor<64x64xf32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<64x64x3xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x64x3xf32>, tensor<f32>) -> tensor<64x64xf32>
    %2 = stablehlo.sqrt %1 : tensor<64x64xf32>
    return %2 : tensor<64x64xf32>
  }
  func.func private @None(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %1 = stablehlo.multiply %0, %arg0 : tensor<64x64xf32>
    %2 = stablehlo.multiply %1, %arg2 : tensor<64x64xf32>
    %3 = stablehlo.subtract %2, %arg1 : tensor<64x64xf32>
    return %arg2, %3, %arg2 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }
  func.func private @atleast_2d(%arg0: tensor<64xf32>) -> tensor<1x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    return %0 : tensor<1x64xf32>
  }
  func.func private @atleast_2d_0(%arg0: tensor<64x64xf32>) -> tensor<64x1x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    return %0 : tensor<64x1x64xf32>
  }
  func.func private @_where(%arg0: tensor<64x64xi1>, %arg1: tensor<64x64xf32>, %arg2: tensor<i32>) -> tensor<64x64xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<64xf32>) -> tensor<64x64xf32>
    %3 = stablehlo.select %arg0, %arg1, %2 : tensor<64x64xi1>, tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
  func.func private @_jax_calc_local_energy_1(%arg0: tensor<64x64x3xf32>, %arg1: tensor<64xi32>, %arg2: tensor<64x64xi32>, %arg3: tensor<2x2x3x8xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>) -> (tensor<64x64xf32>, tensor<64x3x64xf32>, tensor<64x64x3x8xf32>, tensor<64x64xf32>, tensor<f32>, tensor<64x64xf32>, tensor<6x64x64xf32>, tensor<64x64xf32>, tensor<64x64x3xf32>, tensor<f32>, tensor<64x64xf32>, tensor<64x64xi1>) {
    %0:2 = call @norm_2(%arg0) : (tensor<64x64x3xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>)
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %2 = stablehlo.multiply %1, %0#0 : tensor<64x64xf32>
    %3 = stablehlo.add %arg5, %arg6 : tensor<f32>
    %4 = stablehlo.convert %3 : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %6 = stablehlo.subtract %2, %5 : tensor<64x64xf32>
    %7 = stablehlo.subtract %arg6, %arg5 : tensor<f32>
    %8 = stablehlo.convert %7 : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %10 = stablehlo.divide %6, %9 : tensor<64x64xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %13 = stablehlo.multiply %12, %10 : tensor<64x64xf32>
    %14 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<64xf32>) -> tensor<64x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<6x64x64xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<6x64x64xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %17:6 = stablehlo.while(%iterArg = %13, %iterArg_6 = %c, %iterArg_7 = %14, %iterArg_8 = %10, %iterArg_9 = %15, %iterArg_10 = %16) : tensor<64x64xf32>, tensor<i32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<6x64x64xf32>, tensor<6x64x64xf32>
     cond {
      %c_11 = stablehlo.constant dense<6> : tensor<i32>
      %85 = stablehlo.compare  LT, %iterArg_6, %c_11,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %85 : tensor<i1>
    } do {
      %85:4 = func.call @None_3(%iterArg, %iterArg_7, %iterArg_8) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
      %86 = stablehlo.broadcast_in_dim %85#2, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
      %c_11 = stablehlo.constant dense<0> : tensor<i32>
      %87 = stablehlo.dynamic_update_slice %iterArg_9, %86, %iterArg_6, %c_11, %c_11 : (tensor<6x64x64xf32>, tensor<1x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x64x64xf32>
      %88 = stablehlo.broadcast_in_dim %85#3, dims = [1, 2] : (tensor<64x64xf32>) -> tensor<1x64x64xf32>
      %89 = stablehlo.dynamic_update_slice %iterArg_10, %88, %iterArg_6, %c_11, %c_11 : (tensor<6x64x64xf32>, tensor<1x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x64x64xf32>
      %c_12 = stablehlo.constant dense<1> : tensor<i32>
      %90 = stablehlo.add %iterArg_6, %c_12 : tensor<i32>
      stablehlo.return %iterArg, %90, %85#0, %85#1, %87, %89 : tensor<64x64xf32>, tensor<i32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<6x64x64xf32>, tensor<6x64x64xf32>
    }
    %18 = stablehlo.slice %17#4 [0:1, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %19 = stablehlo.transpose %18, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %20 = stablehlo.reshape %19 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %21 = stablehlo.slice %17#4 [1:2, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %22 = stablehlo.transpose %21, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %23 = stablehlo.reshape %22 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %24 = stablehlo.slice %17#4 [2:3, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %25 = stablehlo.transpose %24, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %26 = stablehlo.reshape %25 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %27 = stablehlo.slice %17#4 [3:4, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %28 = stablehlo.transpose %27, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %29 = stablehlo.reshape %28 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %30 = stablehlo.slice %17#4 [4:5, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %31 = stablehlo.transpose %30, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %32 = stablehlo.reshape %31 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %33 = stablehlo.slice %17#4 [5:6, 0:64, 0:64] : (tensor<6x64x64xf32>) -> tensor<1x64x64xf32>
    %34 = stablehlo.transpose %33, dims = [1, 0, 2] : (tensor<1x64x64xf32>) -> tensor<64x1x64xf32>
    %35 = stablehlo.reshape %34 : (tensor<64x1x64xf32>) -> tensor<64x64xf32>
    %36 = call @atleast_2d_4(%11) : (tensor<64xf32>) -> tensor<1x64xf32>
    %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %38 = call @atleast_2d_5(%10) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %39 = stablehlo.transpose %38, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %40 = call @atleast_2d_5(%20) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %41 = stablehlo.transpose %40, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %42 = call @atleast_2d_5(%23) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %43 = stablehlo.transpose %42, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %44 = call @atleast_2d_5(%26) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %45 = stablehlo.transpose %44, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %46 = call @atleast_2d_5(%29) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %47 = stablehlo.transpose %46, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %48 = call @atleast_2d_5(%32) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %49 = stablehlo.transpose %48, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %50 = call @atleast_2d_5(%35) : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %51 = stablehlo.transpose %50, dims = [0, 2, 1] : (tensor<64x1x64xf32>) -> tensor<64x64x1xf32>
    %52 = stablehlo.broadcast_in_dim %37, dims = [1, 2] : (tensor<64x1xf32>) -> tensor<64x64x1xf32>
    %53 = stablehlo.concatenate %52, %39, %41, %43, %45, %47, %49, %51, dim = 2 : (tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>, tensor<64x64x1xf32>) -> tensor<64x64x8xf32>
    %54 = stablehlo.convert %arg6 : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %56 = stablehlo.compare  LT, %0#0, %55,  FLOAT : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xi1>
    %57 = stablehlo.convert %arg6 : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %59 = stablehlo.subtract %58, %0#0 : tensor<64x64xf32>
    %60 = stablehlo.multiply %59, %59 : tensor<64x64xf32>
    %61 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %62 = stablehlo.multiply %61, %59 : tensor<64x64xf32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %63 = call @_where_6(%56, %60, %c_3) : (tensor<64x64xi1>, tensor<64x64xf32>, tensor<i32>) -> tensor<64x64xf32>
    %64 = stablehlo.convert %arg4 : tensor<f32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %66 = stablehlo.multiply %65, %63 : tensor<64x64xf32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %67 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %68 = stablehlo.compare  LT, %arg1, %67,  SIGNED : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi1>
    %c_5 = stablehlo.constant dense<2> : tensor<i32>
    %69 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %70 = stablehlo.add %arg1, %69 : tensor<64xi32>
    %71 = stablehlo.select %68, %70, %arg1 : tensor<64xi1>, tensor<64xi32>
    %72 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<64x64xi32>
    %73 = stablehlo.compare  LT, %arg2, %72,  SIGNED : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi1>
    %74 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<64x64xi32>
    %75 = stablehlo.add %arg2, %74 : tensor<64x64xi32>
    %76 = stablehlo.select %73, %75, %arg2 : tensor<64x64xi1>, tensor<64x64xi32>
    %77 = stablehlo.broadcast_in_dim %71, dims = [0] : (tensor<64xi32>) -> tensor<64x64xi32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [0, 1] : (tensor<64x64xi32>) -> tensor<64x64x1xi32>
    %79 = stablehlo.broadcast_in_dim %76, dims = [0, 1] : (tensor<64x64xi32>) -> tensor<64x64x1xi32>
    %80 = stablehlo.concatenate %78, %79, dim = 2 : (tensor<64x64x1xi32>, tensor<64x64x1xi32>) -> tensor<64x64x2xi32>
    %81 = "stablehlo.gather"(%arg3, %80) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 3, 8>}> : (tensor<2x2x3x8xf32>, tensor<64x64x2xi32>) -> tensor<64x64x3x8xf32>
    %82 = stablehlo.dot_general %81, %53, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x8xf32>, tensor<64x64x8xf32>) -> tensor<64x64x3xf32>
    %83 = stablehlo.dot_general %82, %66, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x64xf32>) -> tensor<64x64x3xf32>
    %84 = stablehlo.transpose %83, dims = [0, 2, 1] : (tensor<64x64x3xf32>) -> tensor<64x3x64xf32>
    return %0#0, %84, %81, %0#1, %8, %13, %17#5, %66, %82, %64, %62, %56 : tensor<64x64xf32>, tensor<64x3x64xf32>, tensor<64x64x3x8xf32>, tensor<64x64xf32>, tensor<f32>, tensor<64x64xf32>, tensor<6x64x64xf32>, tensor<64x64xf32>, tensor<64x64x3xf32>, tensor<f32>, tensor<64x64xf32>, tensor<64x64xi1>
  }
  func.func private @norm_2(%arg0: tensor<64x64x3xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<64x64x3xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x64x3xf32>, tensor<f32>) -> tensor<64x64xf32>
    %2 = stablehlo.sqrt %1 : tensor<64x64xf32>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x64xf32>
    %4 = stablehlo.divide %3, %2 : tensor<64x64xf32>
    return %2, %4 : tensor<64x64xf32>, tensor<64x64xf32>
  }
  func.func private @None_3(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    %0 = stablehlo.multiply %arg0, %arg2 : tensor<64x64xf32>
    %1 = stablehlo.subtract %0, %arg1 : tensor<64x64xf32>
    return %arg2, %1, %arg2, %arg2 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }
  func.func private @atleast_2d_4(%arg0: tensor<64xf32>) -> tensor<1x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    return %0 : tensor<1x64xf32>
  }
  func.func private @atleast_2d_5(%arg0: tensor<64x64xf32>) -> tensor<64x1x64xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    return %0 : tensor<64x1x64xf32>
  }
  func.func private @_where_6(%arg0: tensor<64x64xi1>, %arg1: tensor<64x64xf32>, %arg2: tensor<i32>) -> tensor<64x64xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<64xf32>) -> tensor<64x64xf32>
    %3 = stablehlo.select %arg0, %arg1, %2 : tensor<64x64xi1>, tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
  func.func private @_jax_calc_local_energy_7(%arg0: tensor<29xf32>, %arg1: tensor<64x64x3xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x3x64xf32>, %arg4: tensor<64x64x3x8xf32>, %arg5: tensor<64x64xf32>, %arg6: tensor<f32>, %arg7: tensor<64x64xf32>, %arg8: tensor<6x64x64xf32>, %arg9: tensor<64x64xf32>, %arg10: tensor<64x64x3xf32>, %arg11: tensor<f32>, %arg12: tensor<64x64xf32>, %arg13: tensor<64x64xi1>, %arg14: tensor<1xf32>) -> tensor<64x1x64x3xf32> {
    %0 = stablehlo.dot_general %arg14, %arg0, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xf32>, tensor<29xf32>) -> tensor<1x29xf32>
    %1:4 = stablehlo.optimization_barrier %arg1, %arg2, %arg3, %0 : tensor<64x64x3xf32>, tensor<64x64xf32>, tensor<64x3x64xf32>, tensor<1x29xf32>
    %2 = stablehlo.transpose %1#0, dims = [0, 2, 1] : (tensor<64x64x3xf32>) -> tensor<64x3x64xf32>
    %3 = stablehlo.broadcast_in_dim %1#1, dims = [0, 2] : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<64x1x64xf32>) -> tensor<64x3x64xf32>
    %5 = stablehlo.divide %2, %4 : tensor<64x3x64xf32>
    %6 = stablehlo.multiply %3, %3 : tensor<64x1x64xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1x64xf32>
    %8 = stablehlo.divide %7, %6 : tensor<64x1x64xf32>
    %9 = stablehlo.transpose %5, dims = [0, 2, 1] : (tensor<64x3x64xf32>) -> tensor<64x64x3xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.reduce(%1#2 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<64x3x64xf32>, tensor<f32>) -> tensor<64x3xf32>
    %11 = stablehlo.slice %10 [0:64, 0:1] : (tensor<64x3xf32>) -> tensor<64x1xf32>
    %12 = stablehlo.reshape %11 : (tensor<64x1xf32>) -> tensor<64xf32>
    %13 = stablehlo.slice %10 [0:64, 1:2] : (tensor<64x3xf32>) -> tensor<64x1xf32>
    %14 = stablehlo.reshape %13 : (tensor<64x1xf32>) -> tensor<64xf32>
    %15 = stablehlo.slice %10 [0:64, 2:3] : (tensor<64x3xf32>) -> tensor<64x1xf32>
    %16 = stablehlo.reshape %15 : (tensor<64x1xf32>) -> tensor<64xf32>
    %17 = stablehlo.dot_general %1#2, %9, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3x64xf32>, tensor<64x64x3xf32>) -> tensor<64x3x3xf32>
    %18 = stablehlo.slice %17 [0:64, 0:1, 0:3] : (tensor<64x3x3xf32>) -> tensor<64x1x3xf32>
    %19 = stablehlo.reshape %18 : (tensor<64x1x3xf32>) -> tensor<64x3xf32>
    %20 = stablehlo.slice %17 [0:64, 1:2, 0:3] : (tensor<64x3x3xf32>) -> tensor<64x1x3xf32>
    %21 = stablehlo.reshape %20 : (tensor<64x1x3xf32>) -> tensor<64x3xf32>
    %22 = stablehlo.dot_general %9, %1#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x3x3xf32>
    %23 = stablehlo.dot_general %22, %9, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x3x3x3xf32>
    %24 = stablehlo.transpose %23, dims = [0, 2, 1, 3] : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xf32>
    %25 = stablehlo.slice %24 [0:64, 0:1, 0:3, 0:3] : (tensor<64x3x3x3xf32>) -> tensor<64x1x3x3xf32>
    %26 = stablehlo.reshape %25 : (tensor<64x1x3x3xf32>) -> tensor<64x3x3xf32>
    %27 = stablehlo.slice %24 [0:64, 1:2, 0:3, 0:3] : (tensor<64x3x3x3xf32>) -> tensor<64x1x3x3xf32>
    %28 = stablehlo.reshape %27 : (tensor<64x1x3x3xf32>) -> tensor<64x3x3xf32>
    %29 = stablehlo.dot_general %9, %1#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x3x3xf32>
    %30 = stablehlo.dot_general %9, %9, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x3x3xf32>
    %31 = stablehlo.dot_general %30, %29, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x3x3x3x3xf32>
    %32 = stablehlo.transpose %31, dims = [0, 4, 3, 2, 1] : (tensor<64x3x3x3x3xf32>) -> tensor<64x3x3x3x3xf32>
    %33 = stablehlo.slice %32 [0:64, 0:1, 0:3, 0:3, 0:3] : (tensor<64x3x3x3x3xf32>) -> tensor<64x1x3x3x3xf32>
    %34 = stablehlo.reshape %33 : (tensor<64x1x3x3x3xf32>) -> tensor<64x3x3x3xf32>
    %35 = stablehlo.dot_general %9, %1#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x3x3xf32>
    %36 = stablehlo.dot_general %9, %9, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x3x3xf32>
    %37 = stablehlo.dot_general %35, %9, batching_dims = [0, 1] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x3x3x3xf32>
    %38 = stablehlo.dot_general %37, %36, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x64x3x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x3x3x3x3x3xf32>
    %39 = stablehlo.transpose %38, dims = [0, 2, 1, 5, 4, 3] : (tensor<64x3x3x3x3x3xf32>) -> tensor<64x3x3x3x3x3xf32>
    %40 = stablehlo.slice %39 [0:64, 0:1, 0:3, 0:3, 0:3, 0:3] : (tensor<64x3x3x3x3x3xf32>) -> tensor<64x1x3x3x3x3xf32>
    %41 = stablehlo.reshape %40 : (tensor<64x1x3x3x3x3xf32>) -> tensor<64x3x3x3x3xf32>
    %42 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %43 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %44 = stablehlo.dot_general %42, %43, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %45 = stablehlo.convert %44 : (tensor<64xbf16>) -> tensor<64xf32>
    %46 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %47 = stablehlo.convert %14 : (tensor<64xf32>) -> tensor<64xbf16>
    %48 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %49 = stablehlo.convert %16 : (tensor<64xf32>) -> tensor<64xbf16>
    %50 = stablehlo.convert %19 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %51 = stablehlo.convert %19 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %52 = stablehlo.dot_general %50, %51, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3xbf16>) -> tensor<64xbf16>
    %53 = stablehlo.convert %52 : (tensor<64xbf16>) -> tensor<64xf32>
    %54 = stablehlo.convert %19 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %55 = stablehlo.convert %21 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %56 = stablehlo.dot_general %54, %55, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3xbf16>) -> tensor<64xbf16>
    %57 = stablehlo.convert %56 : (tensor<64xbf16>) -> tensor<64xf32>
    %58 = stablehlo.convert %26 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %59 = stablehlo.convert %26 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %60 = stablehlo.dot_general %58, %59, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64xbf16>
    %61 = stablehlo.convert %60 : (tensor<64xbf16>) -> tensor<64xf32>
    %62 = stablehlo.convert %26 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %63 = stablehlo.convert %28 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %64 = stablehlo.convert %34 : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xbf16>
    %65 = stablehlo.convert %34 : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xbf16>
    %66 = stablehlo.dot_general %64, %65, batching_dims = [0] x [0], contracting_dims = [1, 2, 3] x [1, 2, 3], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3x3xbf16>, tensor<64x3x3x3xbf16>) -> tensor<64xbf16>
    %67 = stablehlo.convert %66 : (tensor<64xbf16>) -> tensor<64xf32>
    %68 = stablehlo.convert %41 : (tensor<64x3x3x3x3xf32>) -> tensor<64x3x3x3x3xbf16>
    %69 = stablehlo.convert %41 : (tensor<64x3x3x3x3xf32>) -> tensor<64x3x3x3x3xbf16>
    %70 = stablehlo.convert %14 : (tensor<64xf32>) -> tensor<64xbf16>
    %71 = stablehlo.convert %14 : (tensor<64xf32>) -> tensor<64xbf16>
    %72 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %73 = stablehlo.convert %45 : (tensor<64xf32>) -> tensor<64xbf16>
    %74 = stablehlo.dot_general %72, %73, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<64xbf16>
    %75 = stablehlo.convert %74 : (tensor<64xbf16>) -> tensor<64xf32>
    %76 = stablehlo.convert %14 : (tensor<64xf32>) -> tensor<64xbf16>
    %77 = stablehlo.convert %45 : (tensor<64xf32>) -> tensor<64xbf16>
    %78 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %79 = stablehlo.convert %53 : (tensor<64xf32>) -> tensor<64xbf16>
    %80 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %81 = stablehlo.convert %57 : (tensor<64xf32>) -> tensor<64xbf16>
    %82 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %83 = stablehlo.convert %61 : (tensor<64xf32>) -> tensor<64xbf16>
    %84 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %85 = stablehlo.convert %67 : (tensor<64xf32>) -> tensor<64xbf16>
    %86 = stablehlo.convert %19 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %87 = stablehlo.convert %26 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %88 = stablehlo.dot_general %86, %87, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64x3xbf16>
    %89 = stablehlo.convert %88 : (tensor<64x3xbf16>) -> tensor<64x3xf32>
    %90 = stablehlo.convert %19 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %91 = stablehlo.convert %89 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %92 = stablehlo.dot_general %90, %91, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3xbf16>, tensor<64x3xbf16>) -> tensor<64xbf16>
    %93 = stablehlo.convert %92 : (tensor<64xbf16>) -> tensor<64xf32>
    %94 = stablehlo.convert %14 : (tensor<64xf32>) -> tensor<64xbf16>
    %95 = stablehlo.convert %53 : (tensor<64xf32>) -> tensor<64xbf16>
    %96 = stablehlo.convert %26 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %97 = stablehlo.convert %34 : (tensor<64x3x3x3xf32>) -> tensor<64x3x3x3xbf16>
    %98 = stablehlo.dot_general %96, %97, batching_dims = [0] x [0], contracting_dims = [1, 2] x [1, 2], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3xbf16>, tensor<64x3x3x3xbf16>) -> tensor<64x3xbf16>
    %99 = stablehlo.convert %98 : (tensor<64x3xbf16>) -> tensor<64x3xf32>
    %100 = stablehlo.convert %19 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %101 = stablehlo.convert %99 : (tensor<64x3xf32>) -> tensor<64x3xbf16>
    %102 = stablehlo.convert %26 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %103 = stablehlo.convert %26 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %104 = stablehlo.dot_general %102, %103, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x3x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64x3x3xbf16>
    %105 = stablehlo.convert %104 : (tensor<64x3x3xbf16>) -> tensor<64x3x3xf32>
    %106 = stablehlo.convert %26 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %107 = stablehlo.convert %105 : (tensor<64x3x3xf32>) -> tensor<64x3x3xbf16>
    %108 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %109 = stablehlo.convert %75 : (tensor<64xf32>) -> tensor<64xbf16>
    %110 = stablehlo.convert %14 : (tensor<64xf32>) -> tensor<64xbf16>
    %111 = stablehlo.convert %75 : (tensor<64xf32>) -> tensor<64xbf16>
    %112 = stablehlo.convert %53 : (tensor<64xf32>) -> tensor<64xbf16>
    %113 = stablehlo.convert %45 : (tensor<64xf32>) -> tensor<64xbf16>
    %114 = stablehlo.convert %61 : (tensor<64xf32>) -> tensor<64xbf16>
    %115 = stablehlo.convert %45 : (tensor<64xf32>) -> tensor<64xbf16>
    %116 = stablehlo.convert %12 : (tensor<64xf32>) -> tensor<64xbf16>
    %117 = stablehlo.convert %93 : (tensor<64xf32>) -> tensor<64xbf16>
    %118 = stablehlo.convert %53 : (tensor<64xf32>) -> tensor<64xbf16>
    %119 = stablehlo.convert %53 : (tensor<64xf32>) -> tensor<64xbf16>
    %120 = stablehlo.slice %1#3 [0:1, 0:16] : (tensor<1x29xf32>) -> tensor<1x16xf32>
    %121 = stablehlo.slice %1#3 [0:1, 16:29] : (tensor<1x29xf32>) -> tensor<1x13xf32>
    %122 = stablehlo.slice %121 [0:1, 0:1] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %123 = stablehlo.slice %121 [0:1, 1:2] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %124 = stablehlo.slice %121 [0:1, 2:3] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %125 = stablehlo.slice %121 [0:1, 3:4] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %126 = stablehlo.slice %121 [0:1, 4:5] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %127 = stablehlo.slice %121 [0:1, 5:6] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %128 = stablehlo.slice %121 [0:1, 6:7] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %129 = stablehlo.slice %121 [0:1, 7:8] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %130 = stablehlo.slice %121 [0:1, 8:9] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %131 = stablehlo.slice %121 [0:1, 9:10] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %132 = stablehlo.slice %121 [0:1, 10:11] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %133 = stablehlo.slice %121 [0:1, 11:12] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %134 = stablehlo.slice %121 [0:1, 12:13] : (tensor<1x13xf32>) -> tensor<1x1xf32>
    %135 = stablehlo.slice %120 [0:1, 0:1] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %136 = stablehlo.slice %120 [0:1, 1:2] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %137 = stablehlo.slice %120 [0:1, 2:3] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %138 = stablehlo.slice %120 [0:1, 3:4] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %139 = stablehlo.slice %120 [0:1, 4:5] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %140 = stablehlo.slice %120 [0:1, 5:6] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %141 = stablehlo.slice %120 [0:1, 6:7] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %142 = stablehlo.slice %120 [0:1, 7:8] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %143 = stablehlo.slice %120 [0:1, 8:9] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %144 = stablehlo.slice %120 [0:1, 9:10] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %145 = stablehlo.slice %120 [0:1, 10:11] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %146 = stablehlo.slice %120 [0:1, 11:12] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %147 = stablehlo.slice %120 [0:1, 12:13] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %148 = stablehlo.slice %120 [0:1, 13:14] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %149 = stablehlo.slice %120 [0:1, 14:15] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %150 = stablehlo.slice %120 [0:1, 15:16] : (tensor<1x16xf32>) -> tensor<1x1xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %151 = stablehlo.reduce(%134 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %152 = stablehlo.reduce(%133 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %153 = stablehlo.reduce(%132 init: %cst_3) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %154 = stablehlo.reduce(%131 init: %cst_4) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %155 = stablehlo.reduce(%130 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %156 = stablehlo.reduce(%129 init: %cst_6) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %157 = stablehlo.reduce(%128 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %158 = stablehlo.reduce(%127 init: %cst_8) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %159 = stablehlo.reduce(%126 init: %cst_9) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %160 = stablehlo.reduce(%125 init: %cst_10) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %161 = stablehlo.reduce(%124 init: %cst_11) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %162 = stablehlo.reduce(%123 init: %cst_12) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %163 = stablehlo.reduce(%122 init: %cst_13) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %164 = stablehlo.reduce(%150 init: %cst_14) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %165 = stablehlo.reduce(%149 init: %cst_15) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %166 = stablehlo.reduce(%148 init: %cst_16) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %167 = stablehlo.reduce(%147 init: %cst_17) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %168 = stablehlo.reduce(%146 init: %cst_18) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %169 = stablehlo.reduce(%145 init: %cst_19) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %170 = stablehlo.reduce(%144 init: %cst_20) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %171 = stablehlo.reduce(%143 init: %cst_21) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %172 = stablehlo.reduce(%142 init: %cst_22) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %173 = stablehlo.reduce(%141 init: %cst_23) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %174 = stablehlo.reduce(%140 init: %cst_24) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %175 = stablehlo.reduce(%139 init: %cst_25) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %176 = stablehlo.reduce(%138 init: %cst_26) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %177 = stablehlo.reduce(%137 init: %cst_27) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %178 = stablehlo.reduce(%136 init: %cst_28) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %179 = stablehlo.reduce(%135 init: %cst_29) applies stablehlo.add across dimensions = [1] : (tensor<1x1xf32>, tensor<f32>) -> tensor<1xf32>
    %180 = stablehlo.convert %151 : (tensor<1xf32>) -> tensor<1xbf16>
    %181 = stablehlo.dot_general %180, %118, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %182 = stablehlo.dot_general %180, %119, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %183 = stablehlo.convert %181 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %184 = stablehlo.broadcast_in_dim %173, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %185 = stablehlo.add %184, %183 : tensor<1x64xf32>
    %186 = stablehlo.convert %182 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %187 = stablehlo.add %185, %186 : tensor<1x64xf32>
    %188 = stablehlo.convert %152 : (tensor<1xf32>) -> tensor<1xbf16>
    %189 = stablehlo.dot_general %188, %116, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %190 = stablehlo.dot_general %188, %117, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %191 = stablehlo.convert %189 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %192 = stablehlo.broadcast_in_dim %160, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %193 = stablehlo.add %192, %191 : tensor<1x64xf32>
    %194 = stablehlo.convert %190 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %195 = stablehlo.broadcast_in_dim %179, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %196 = stablehlo.add %195, %194 : tensor<1x64xf32>
    %197 = stablehlo.convert %153 : (tensor<1xf32>) -> tensor<1xbf16>
    %198 = stablehlo.dot_general %197, %114, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %199 = stablehlo.dot_general %197, %115, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %200 = stablehlo.convert %198 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %201 = stablehlo.broadcast_in_dim %176, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %202 = stablehlo.add %201, %200 : tensor<1x64xf32>
    %203 = stablehlo.convert %199 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %204 = stablehlo.broadcast_in_dim %171, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %205 = stablehlo.add %204, %203 : tensor<1x64xf32>
    %206 = stablehlo.convert %154 : (tensor<1xf32>) -> tensor<1xbf16>
    %207 = stablehlo.dot_general %206, %112, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %208 = stablehlo.dot_general %206, %113, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %209 = stablehlo.convert %207 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %210 = stablehlo.add %202, %209 : tensor<1x64xf32>
    %211 = stablehlo.convert %208 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %212 = stablehlo.add %187, %211 : tensor<1x64xf32>
    %213 = stablehlo.convert %155 : (tensor<1xf32>) -> tensor<1xbf16>
    %214 = stablehlo.dot_general %213, %110, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %215 = stablehlo.dot_general %213, %111, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %216 = stablehlo.convert %214 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %217 = stablehlo.broadcast_in_dim %166, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %218 = stablehlo.add %217, %216 : tensor<1x64xf32>
    %219 = stablehlo.convert %215 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %220 = stablehlo.broadcast_in_dim %178, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %221 = stablehlo.add %220, %219 : tensor<1x64xf32>
    %222 = stablehlo.convert %156 : (tensor<1xf32>) -> tensor<1xbf16>
    %223 = stablehlo.dot_general %222, %108, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %224 = stablehlo.dot_general %222, %109, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %225 = stablehlo.convert %223 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %226 = stablehlo.add %218, %225 : tensor<1x64xf32>
    %227 = stablehlo.convert %224 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %228 = stablehlo.add %196, %227 : tensor<1x64xf32>
    %229 = stablehlo.convert %157 : (tensor<1xf32>) -> tensor<1xbf16>
    %230 = stablehlo.dot_general %229, %106, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64x3x3xbf16>) -> tensor<1x64x3x3xbf16>
    %231 = stablehlo.dot_general %229, %107, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64x3x3xbf16>) -> tensor<1x64x3x3xbf16>
    %232 = stablehlo.convert %230 : (tensor<1x64x3x3xbf16>) -> tensor<1x64x3x3xf32>
    %233 = stablehlo.convert %231 : (tensor<1x64x3x3xbf16>) -> tensor<1x64x3x3xf32>
    %234 = stablehlo.convert %232 : (tensor<1x64x3x3xf32>) -> tensor<1x64x3x3xbf16>
    %235 = stablehlo.dot_general %234, %102, batching_dims = [1] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x64x3x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64x1x3x3xbf16>
    %236 = stablehlo.transpose %235, dims = [0, 1, 3, 2] : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xbf16>
    %237 = stablehlo.dot_general %234, %103, batching_dims = [1] x [0], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x64x3x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64x1x3x3xbf16>
    %238 = stablehlo.transpose %237, dims = [0, 1, 3, 2] : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xbf16>
    %239 = stablehlo.convert %236 : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xf32>
    %240 = stablehlo.transpose %233, dims = [1, 0, 2, 3] : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %241 = stablehlo.add %240, %239 : tensor<64x1x3x3xf32>
    %242 = stablehlo.convert %238 : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xf32>
    %243 = stablehlo.add %241, %242 : tensor<64x1x3x3xf32>
    %244 = stablehlo.convert %158 : (tensor<1xf32>) -> tensor<1xbf16>
    %245 = stablehlo.dot_general %244, %100, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64x3xbf16>) -> tensor<1x64x3xbf16>
    %246 = stablehlo.dot_general %244, %101, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64x3xbf16>) -> tensor<1x64x3xbf16>
    %247 = stablehlo.convert %245 : (tensor<1x64x3xbf16>) -> tensor<1x64x3xf32>
    %248 = stablehlo.convert %246 : (tensor<1x64x3xbf16>) -> tensor<1x64x3xf32>
    %249 = stablehlo.convert %247 : (tensor<1x64x3xf32>) -> tensor<1x64x3xbf16>
    %250 = stablehlo.dot_general %249, %96, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64x1x3x3x3xbf16>
    %251 = stablehlo.transpose %250, dims = [0, 1, 3, 4, 2] : (tensor<64x1x3x3x3xbf16>) -> tensor<64x1x3x3x3xbf16>
    %252 = stablehlo.dot_general %249, %97, batching_dims = [1] x [0], contracting_dims = [2] x [3], precision = [DEFAULT, DEFAULT] : (tensor<1x64x3xbf16>, tensor<64x3x3x3xbf16>) -> tensor<64x1x3x3xbf16>
    %253 = stablehlo.convert %251 : (tensor<64x1x3x3x3xbf16>) -> tensor<64x1x3x3x3xf32>
    %254 = stablehlo.convert %252 : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xf32>
    %255 = stablehlo.add %243, %254 : tensor<64x1x3x3xf32>
    %256 = stablehlo.convert %159 : (tensor<1xf32>) -> tensor<1xbf16>
    %257 = stablehlo.dot_general %256, %94, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %258 = stablehlo.dot_general %256, %95, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %259 = stablehlo.convert %257 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %260 = stablehlo.add %212, %259 : tensor<1x64xf32>
    %261 = stablehlo.convert %258 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %262 = stablehlo.add %221, %261 : tensor<1x64xf32>
    %263 = stablehlo.convert %193 : (tensor<1x64xf32>) -> tensor<1x64xbf16>
    %264 = stablehlo.dot_general %263, %90, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3xbf16>) -> tensor<64x1x3xbf16>
    %265 = stablehlo.dot_general %263, %91, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3xbf16>) -> tensor<64x1x3xbf16>
    %266 = stablehlo.convert %264 : (tensor<64x1x3xbf16>) -> tensor<64x1x3xf32>
    %267 = stablehlo.convert %265 : (tensor<64x1x3xbf16>) -> tensor<64x1x3xf32>
    %268 = stablehlo.transpose %248, dims = [1, 0, 2] : (tensor<1x64x3xf32>) -> tensor<64x1x3xf32>
    %269 = stablehlo.add %268, %267 : tensor<64x1x3xf32>
    %270 = stablehlo.convert %266 : (tensor<64x1x3xf32>) -> tensor<64x1x3xbf16>
    %271 = stablehlo.dot_general %270, %86, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3xbf16>, tensor<64x3xbf16>) -> tensor<64x1x3x3xbf16>
    %272 = stablehlo.transpose %271, dims = [0, 1, 3, 2] : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xbf16>
    %273 = stablehlo.dot_general %270, %87, batching_dims = [0] x [0], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3xbf16>, tensor<64x3x3xbf16>) -> tensor<64x1x3xbf16>
    %274 = stablehlo.convert %272 : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xf32>
    %275 = stablehlo.add %255, %274 : tensor<64x1x3x3xf32>
    %276 = stablehlo.convert %273 : (tensor<64x1x3xbf16>) -> tensor<64x1x3xf32>
    %277 = stablehlo.add %269, %276 : tensor<64x1x3xf32>
    %278 = stablehlo.convert %161 : (tensor<1xf32>) -> tensor<1xbf16>
    %279 = stablehlo.dot_general %278, %84, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %280 = stablehlo.dot_general %278, %85, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %281 = stablehlo.convert %279 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %282 = stablehlo.broadcast_in_dim %169, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %283 = stablehlo.add %282, %281 : tensor<1x64xf32>
    %284 = stablehlo.convert %280 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %285 = stablehlo.add %228, %284 : tensor<1x64xf32>
    %286 = stablehlo.convert %162 : (tensor<1xf32>) -> tensor<1xbf16>
    %287 = stablehlo.dot_general %286, %82, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %288 = stablehlo.dot_general %286, %83, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %289 = stablehlo.convert %287 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %290 = stablehlo.add %205, %289 : tensor<1x64xf32>
    %291 = stablehlo.convert %288 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %292 = stablehlo.add %285, %291 : tensor<1x64xf32>
    %293 = stablehlo.convert %163 : (tensor<1xf32>) -> tensor<1xbf16>
    %294 = stablehlo.dot_general %293, %80, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %295 = stablehlo.dot_general %293, %81, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %296 = stablehlo.convert %294 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %297 = stablehlo.broadcast_in_dim %172, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %298 = stablehlo.add %297, %296 : tensor<1x64xf32>
    %299 = stablehlo.convert %295 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %300 = stablehlo.add %292, %299 : tensor<1x64xf32>
    %301 = stablehlo.convert %164 : (tensor<1xf32>) -> tensor<1xbf16>
    %302 = stablehlo.dot_general %301, %78, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %303 = stablehlo.dot_general %301, %79, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %304 = stablehlo.convert %302 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %305 = stablehlo.add %260, %304 : tensor<1x64xf32>
    %306 = stablehlo.convert %303 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %307 = stablehlo.add %300, %306 : tensor<1x64xf32>
    %308 = stablehlo.convert %165 : (tensor<1xf32>) -> tensor<1xbf16>
    %309 = stablehlo.dot_general %308, %76, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %310 = stablehlo.dot_general %308, %77, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %311 = stablehlo.convert %309 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %312 = stablehlo.add %210, %311 : tensor<1x64xf32>
    %313 = stablehlo.convert %310 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %314 = stablehlo.add %262, %313 : tensor<1x64xf32>
    %315 = stablehlo.convert %226 : (tensor<1x64xf32>) -> tensor<1x64xbf16>
    %316 = stablehlo.dot_general %315, %72, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64xbf16>) -> tensor<64x1xbf16>
    %317 = stablehlo.dot_general %315, %73, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64xbf16>) -> tensor<64x1xbf16>
    %318 = stablehlo.convert %316 : (tensor<64x1xbf16>) -> tensor<64x1xf32>
    %319 = stablehlo.transpose %312, dims = [1, 0] : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %320 = stablehlo.add %319, %318 : tensor<64x1xf32>
    %321 = stablehlo.convert %317 : (tensor<64x1xbf16>) -> tensor<64x1xf32>
    %322 = stablehlo.transpose %307, dims = [1, 0] : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %323 = stablehlo.add %322, %321 : tensor<64x1xf32>
    %324 = stablehlo.convert %167 : (tensor<1xf32>) -> tensor<1xbf16>
    %325 = stablehlo.dot_general %324, %70, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %326 = stablehlo.dot_general %324, %71, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %327 = stablehlo.convert %325 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %328 = stablehlo.add %314, %327 : tensor<1x64xf32>
    %329 = stablehlo.convert %326 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %330 = stablehlo.add %328, %329 : tensor<1x64xf32>
    %331 = stablehlo.convert %168 : (tensor<1xf32>) -> tensor<1xbf16>
    %332 = stablehlo.dot_general %331, %68, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64x3x3x3x3xbf16>) -> tensor<1x64x3x3x3x3xbf16>
    %333 = stablehlo.dot_general %331, %69, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64x3x3x3x3xbf16>) -> tensor<1x64x3x3x3x3xbf16>
    %334 = stablehlo.convert %332 : (tensor<1x64x3x3x3x3xbf16>) -> tensor<1x64x3x3x3x3xf32>
    %335 = stablehlo.convert %333 : (tensor<1x64x3x3x3x3xbf16>) -> tensor<1x64x3x3x3x3xf32>
    %336 = stablehlo.add %334, %335 : tensor<1x64x3x3x3x3xf32>
    %337 = stablehlo.convert %283 : (tensor<1x64xf32>) -> tensor<1x64xbf16>
    %338 = stablehlo.dot_general %337, %64, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3x3x3xbf16>) -> tensor<64x1x3x3x3xbf16>
    %339 = stablehlo.dot_general %337, %65, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3x3x3xbf16>) -> tensor<64x1x3x3x3xbf16>
    %340 = stablehlo.convert %338 : (tensor<64x1x3x3x3xbf16>) -> tensor<64x1x3x3x3xf32>
    %341 = stablehlo.add %253, %340 : tensor<64x1x3x3x3xf32>
    %342 = stablehlo.convert %339 : (tensor<64x1x3x3x3xbf16>) -> tensor<64x1x3x3x3xf32>
    %343 = stablehlo.add %341, %342 : tensor<64x1x3x3x3xf32>
    %344 = stablehlo.convert %170 : (tensor<1xf32>) -> tensor<1xbf16>
    %345 = stablehlo.dot_general %344, %62, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64x3x3xbf16>) -> tensor<1x64x3x3xbf16>
    %346 = stablehlo.dot_general %344, %63, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64x3x3xbf16>) -> tensor<1x64x3x3xbf16>
    %347 = stablehlo.convert %345 : (tensor<1x64x3x3xbf16>) -> tensor<1x64x3x3xf32>
    %348 = stablehlo.convert %346 : (tensor<1x64x3x3xbf16>) -> tensor<1x64x3x3xf32>
    %349 = stablehlo.transpose %275, dims = [1, 0, 2, 3] : (tensor<64x1x3x3xf32>) -> tensor<1x64x3x3xf32>
    %350 = stablehlo.add %349, %348 : tensor<1x64x3x3xf32>
    %351 = stablehlo.convert %290 : (tensor<1x64xf32>) -> tensor<1x64xbf16>
    %352 = stablehlo.dot_general %351, %58, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3x3xbf16>) -> tensor<64x1x3x3xbf16>
    %353 = stablehlo.dot_general %351, %59, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3x3xbf16>) -> tensor<64x1x3x3xbf16>
    %354 = stablehlo.convert %352 : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xf32>
    %355 = stablehlo.transpose %350, dims = [1, 0, 2, 3] : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %356 = stablehlo.add %355, %354 : tensor<64x1x3x3xf32>
    %357 = stablehlo.convert %353 : (tensor<64x1x3x3xbf16>) -> tensor<64x1x3x3xf32>
    %358 = stablehlo.add %356, %357 : tensor<64x1x3x3xf32>
    %359 = stablehlo.convert %298 : (tensor<1x64xf32>) -> tensor<1x64xbf16>
    %360 = stablehlo.dot_general %359, %54, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3xbf16>) -> tensor<64x1x3xbf16>
    %361 = stablehlo.dot_general %359, %55, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3xbf16>) -> tensor<64x1x3xbf16>
    %362 = stablehlo.convert %360 : (tensor<64x1x3xbf16>) -> tensor<64x1x3xf32>
    %363 = stablehlo.convert %361 : (tensor<64x1x3xbf16>) -> tensor<64x1x3xf32>
    %364 = stablehlo.add %277, %363 : tensor<64x1x3xf32>
    %365 = stablehlo.convert %305 : (tensor<1x64xf32>) -> tensor<1x64xbf16>
    %366 = stablehlo.dot_general %365, %50, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3xbf16>) -> tensor<64x1x3xbf16>
    %367 = stablehlo.dot_general %365, %51, batching_dims = [1] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1x64xbf16>, tensor<64x3xbf16>) -> tensor<64x1x3xbf16>
    %368 = stablehlo.convert %366 : (tensor<64x1x3xbf16>) -> tensor<64x1x3xf32>
    %369 = stablehlo.add %364, %368 : tensor<64x1x3xf32>
    %370 = stablehlo.convert %367 : (tensor<64x1x3xbf16>) -> tensor<64x1x3xf32>
    %371 = stablehlo.add %369, %370 : tensor<64x1x3xf32>
    %372 = stablehlo.convert %174 : (tensor<1xf32>) -> tensor<1xbf16>
    %373 = stablehlo.dot_general %372, %48, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %374 = stablehlo.dot_general %372, %49, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %375 = stablehlo.convert %373 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %376 = stablehlo.broadcast_in_dim %177, dims = [0] : (tensor<1xf32>) -> tensor<1x64xf32>
    %377 = stablehlo.add %376, %375 : tensor<1x64xf32>
    %378 = stablehlo.convert %374 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %379 = stablehlo.transpose %323, dims = [1, 0] : (tensor<64x1xf32>) -> tensor<1x64xf32>
    %380 = stablehlo.add %379, %378 : tensor<1x64xf32>
    %381 = stablehlo.convert %175 : (tensor<1xf32>) -> tensor<1xbf16>
    %382 = stablehlo.dot_general %381, %46, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %383 = stablehlo.dot_general %381, %47, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<1xbf16>, tensor<64xbf16>) -> tensor<1x64xbf16>
    %384 = stablehlo.convert %382 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %385 = stablehlo.add %330, %384 : tensor<1x64xf32>
    %386 = stablehlo.convert %383 : (tensor<1x64xbf16>) -> tensor<1x64xf32>
    %387 = stablehlo.add %380, %386 : tensor<1x64xf32>
    %388 = stablehlo.convert %320 : (tensor<64x1xf32>) -> tensor<64x1xbf16>
    %389 = stablehlo.dot_general %388, %42, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x1xbf16>, tensor<64xbf16>) -> tensor<64x1xbf16>
    %390 = stablehlo.dot_general %388, %43, batching_dims = [0] x [0], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x1xbf16>, tensor<64xbf16>) -> tensor<64x1xbf16>
    %391 = stablehlo.convert %389 : (tensor<64x1xbf16>) -> tensor<64x1xf32>
    %392 = stablehlo.transpose %387, dims = [1, 0] : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %393 = stablehlo.add %392, %391 : tensor<64x1xf32>
    %394 = stablehlo.convert %390 : (tensor<64x1xbf16>) -> tensor<64x1xf32>
    %395 = stablehlo.add %393, %394 : tensor<64x1xf32>
    %396 = stablehlo.transpose %336, dims = [1, 0, 2, 3, 4, 5] : (tensor<1x64x3x3x3x3xf32>) -> tensor<64x1x3x3x3x3xf32>
    %397 = stablehlo.broadcast_in_dim %396, dims = [0, 1, 3, 4, 5, 6] : (tensor<64x1x3x3x3x3xf32>) -> tensor<64x1x1x3x3x3x3xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %398 = stablehlo.pad %397, %cst_30, low = [0, 0, 0, 0, 0, 0, 0], high = [0, 0, 2, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0, 0, 0] : (tensor<64x1x1x3x3x3x3xf32>, tensor<f32>) -> tensor<64x1x3x3x3x3x3xf32>
    %399 = stablehlo.transpose %398, dims = [0, 1, 3, 2, 6, 5, 4] : (tensor<64x1x3x3x3x3x3xf32>) -> tensor<64x1x3x3x3x3x3xf32>
    %400 = stablehlo.dot_general %399, %37, batching_dims = [0] x [0], contracting_dims = [2, 3, 4] x [2, 3, 4], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3x3x3x3x3xf32>, tensor<64x64x3x3x3xf32>) -> tensor<64x1x3x3x64xf32>
    %401 = stablehlo.transpose %400, dims = [0, 1, 4, 2, 3] : (tensor<64x1x3x3x64xf32>) -> tensor<64x1x64x3x3xf32>
    %402 = stablehlo.dot_general %399, %36, batching_dims = [0] x [0], contracting_dims = [5, 6] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3x3x3x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x1x3x3x3x64xf32>
    %403 = stablehlo.transpose %402, dims = [0, 1, 5, 2, 3, 4] : (tensor<64x1x3x3x3x64xf32>) -> tensor<64x1x64x3x3x3xf32>
    %404 = stablehlo.dot_general %403, %35, batching_dims = [0, 2] x [0, 1], contracting_dims = [3, 4] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x1x3xf32>
    %405 = stablehlo.dot_general %403, %9, batching_dims = [0, 2] x [0, 1], contracting_dims = [5] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1x3x3xf32>
    %406 = stablehlo.dot_general %401, %9, batching_dims = [0, 2] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1x3xf32>
    %407 = stablehlo.add %404, %406 : tensor<64x64x1x3xf32>
    %408 = stablehlo.dot_general %401, %9, batching_dims = [0, 2] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1x3xf32>
    %409 = stablehlo.add %407, %408 : tensor<64x64x1x3xf32>
    %410 = stablehlo.dot_general %405, %9, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x64x1x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1x3xf32>
    %411 = stablehlo.transpose %410, dims = [0, 2, 3, 1] : (tensor<64x64x1x3xf32>) -> tensor<64x1x3x64xf32>
    %412 = stablehlo.dot_general %405, %1#2, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x64x1x3x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x1x3xf32>
    %413 = stablehlo.add %409, %412 : tensor<64x64x1x3xf32>
    %414 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 3, 4, 5] : (tensor<64x1x3x3x3xf32>) -> tensor<64x1x1x3x3x3xf32>
    %415 = stablehlo.pad %414, %cst_30, low = [0, 0, 0, 0, 0, 0], high = [0, 0, 2, 0, 0, 0], interior = [0, 0, 0, 0, 0, 0] : (tensor<64x1x1x3x3x3xf32>, tensor<f32>) -> tensor<64x1x3x3x3x3xf32>
    %416 = stablehlo.transpose %415, dims = [0, 1, 5, 4, 3, 2] : (tensor<64x1x3x3x3x3xf32>) -> tensor<64x1x3x3x3x3xf32>
    %417 = stablehlo.dot_general %416, %30, batching_dims = [0] x [0], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3x3x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x1x3x3x64xf32>
    %418 = stablehlo.transpose %417, dims = [0, 1, 4, 2, 3] : (tensor<64x1x3x3x64xf32>) -> tensor<64x1x64x3x3xf32>
    %419 = stablehlo.dot_general %416, %29, batching_dims = [0] x [0], contracting_dims = [4, 5] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3x3x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x1x3x3x64xf32>
    %420 = stablehlo.transpose %419, dims = [0, 1, 4, 2, 3] : (tensor<64x1x3x3x64xf32>) -> tensor<64x1x64x3x3xf32>
    %421 = stablehlo.dot_general %420, %9, batching_dims = [0, 2] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1x3xf32>
    %422 = stablehlo.add %413, %421 : tensor<64x64x1x3xf32>
    %423 = stablehlo.dot_general %420, %9, batching_dims = [0, 2] x [0, 1], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1x3xf32>
    %424 = stablehlo.add %422, %423 : tensor<64x64x1x3xf32>
    %425 = stablehlo.dot_general %418, %9, batching_dims = [0, 2] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1x3xf32>
    %426 = stablehlo.transpose %425, dims = [0, 2, 3, 1] : (tensor<64x64x1x3xf32>) -> tensor<64x1x3x64xf32>
    %427 = stablehlo.add %411, %426 : tensor<64x1x3x64xf32>
    %428 = stablehlo.dot_general %418, %1#2, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x1x3xf32>
    %429 = stablehlo.add %424, %428 : tensor<64x64x1x3xf32>
    %430 = stablehlo.transpose %347, dims = [1, 0, 2, 3] : (tensor<1x64x3x3xf32>) -> tensor<64x1x3x3xf32>
    %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 3, 4] : (tensor<64x1x3x3xf32>) -> tensor<64x1x1x3x3xf32>
    %432 = stablehlo.pad %431, %cst_30, low = [0, 0, 1, 0, 0], high = [0, 0, 1, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<64x1x1x3x3xf32>, tensor<f32>) -> tensor<64x1x3x3x3xf32>
    %433 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 3, 4] : (tensor<64x1x3x3xf32>) -> tensor<64x1x1x3x3xf32>
    %434 = stablehlo.pad %433, %cst_30, low = [0, 0, 0, 0, 0], high = [0, 0, 2, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<64x1x1x3x3xf32>, tensor<f32>) -> tensor<64x1x3x3x3xf32>
    %435 = stablehlo.add %432, %434 : tensor<64x1x3x3x3xf32>
    %436 = stablehlo.transpose %435, dims = [0, 1, 3, 2, 4] : (tensor<64x1x3x3x3xf32>) -> tensor<64x1x3x3x3xf32>
    %437 = stablehlo.dot_general %436, %22, batching_dims = [0] x [0], contracting_dims = [2, 3] x [2, 3], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3x3x3xf32>, tensor<64x64x3x3xf32>) -> tensor<64x1x3x64xf32>
    %438 = stablehlo.transpose %437, dims = [0, 1, 3, 2] : (tensor<64x1x3x64xf32>) -> tensor<64x1x64x3xf32>
    %439 = stablehlo.transpose %429, dims = [0, 2, 1, 3] : (tensor<64x64x1x3xf32>) -> tensor<64x1x64x3xf32>
    %440 = stablehlo.add %439, %438 : tensor<64x1x64x3xf32>
    %441 = stablehlo.dot_general %436, %9, batching_dims = [0] x [0], contracting_dims = [4] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x1x3x3x64xf32>
    %442 = stablehlo.transpose %441, dims = [0, 1, 4, 2, 3] : (tensor<64x1x3x3x64xf32>) -> tensor<64x1x64x3x3xf32>
    %443 = stablehlo.dot_general %442, %9, batching_dims = [0, 2] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1x3xf32>
    %444 = stablehlo.transpose %443, dims = [0, 2, 3, 1] : (tensor<64x64x1x3xf32>) -> tensor<64x1x3x64xf32>
    %445 = stablehlo.add %427, %444 : tensor<64x1x3x64xf32>
    %446 = stablehlo.dot_general %442, %1#2, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3x3xf32>, tensor<64x3x64xf32>) -> tensor<64x64x1x3xf32>
    %447 = stablehlo.transpose %440, dims = [0, 2, 1, 3] : (tensor<64x1x64x3xf32>) -> tensor<64x64x1x3xf32>
    %448 = stablehlo.add %447, %446 : tensor<64x64x1x3xf32>
    %449 = stablehlo.broadcast_in_dim %362, dims = [0, 1, 3] : (tensor<64x1x3xf32>) -> tensor<64x1x1x3xf32>
    %450 = stablehlo.pad %449, %cst_30, low = [0, 0, 1, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<64x1x1x3xf32>, tensor<f32>) -> tensor<64x1x3x3xf32>
    %451 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 3] : (tensor<64x1x3xf32>) -> tensor<64x1x1x3xf32>
    %452 = stablehlo.pad %451, %cst_30, low = [0, 0, 0, 0], high = [0, 0, 2, 0], interior = [0, 0, 0, 0] : (tensor<64x1x1x3xf32>, tensor<f32>) -> tensor<64x1x3x3xf32>
    %453 = stablehlo.add %450, %452 : tensor<64x1x3x3xf32>
    %454 = stablehlo.dot_general %453, %1#2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3x3xf32>, tensor<64x3x64xf32>) -> tensor<64x1x3x64xf32>
    %455 = stablehlo.transpose %454, dims = [0, 1, 3, 2] : (tensor<64x1x3x64xf32>) -> tensor<64x1x64x3xf32>
    %456 = stablehlo.transpose %448, dims = [0, 2, 1, 3] : (tensor<64x64x1x3xf32>) -> tensor<64x1x64x3xf32>
    %457 = stablehlo.add %456, %455 : tensor<64x1x64x3xf32>
    %458 = stablehlo.dot_general %453, %9, batching_dims = [0] x [0], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x3x3xf32>, tensor<64x64x3xf32>) -> tensor<64x1x3x64xf32>
    %459 = stablehlo.add %445, %458 : tensor<64x1x3x64xf32>
    %460 = stablehlo.transpose %377, dims = [1, 0] : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %461 = stablehlo.broadcast_in_dim %460, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %462 = stablehlo.pad %461, %cst_30, low = [0, 0, 2], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<64x1x1xf32>, tensor<f32>) -> tensor<64x1x3xf32>
    %463 = stablehlo.transpose %385, dims = [1, 0] : (tensor<1x64xf32>) -> tensor<64x1xf32>
    %464 = stablehlo.broadcast_in_dim %463, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %465 = stablehlo.pad %464, %cst_30, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0] : (tensor<64x1x1xf32>, tensor<f32>) -> tensor<64x1x3xf32>
    %466 = stablehlo.add %462, %465 : tensor<64x1x3xf32>
    %467 = stablehlo.broadcast_in_dim %395, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1xf32>
    %468 = stablehlo.pad %467, %cst_30, low = [0, 0, 0], high = [0, 0, 2], interior = [0, 0, 0] : (tensor<64x1x1xf32>, tensor<f32>) -> tensor<64x1x3xf32>
    %469 = stablehlo.add %466, %468 : tensor<64x1x3xf32>
    %470 = stablehlo.broadcast_in_dim %469, dims = [0, 1, 2] : (tensor<64x1x3xf32>) -> tensor<64x1x3x64xf32>
    %471 = stablehlo.add %459, %470 : tensor<64x1x3x64xf32>
    %472 = stablehlo.transpose %457, dims = [0, 1, 3, 2] : (tensor<64x1x64x3xf32>) -> tensor<64x1x3x64xf32>
    %473 = stablehlo.broadcast_in_dim %8, dims = [0, 2, 3] : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %474 = stablehlo.broadcast_in_dim %473, dims = [0, 1, 2, 3] : (tensor<64x1x1x64xf32>) -> tensor<64x1x3x64xf32>
    %475 = stablehlo.multiply %472, %474 : tensor<64x1x3x64xf32>
    %476 = stablehlo.broadcast_in_dim %2, dims = [0, 2, 3] : (tensor<64x3x64xf32>) -> tensor<64x1x3x64xf32>
    %477 = stablehlo.multiply %475, %476 : tensor<64x1x3x64xf32>
    %cst_31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %478 = stablehlo.reduce(%477 init: %cst_31) applies stablehlo.add across dimensions = [2] : (tensor<64x1x3x64xf32>, tensor<f32>) -> tensor<64x1x64xf32>
    %479 = stablehlo.reshape %478 : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %480 = stablehlo.negate %479 : tensor<64x1x1x64xf32>
    %481 = stablehlo.broadcast_in_dim %3, dims = [0, 2, 3] : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %482 = stablehlo.broadcast_in_dim %481, dims = [0, 1, 2, 3] : (tensor<64x1x1x64xf32>) -> tensor<64x1x3x64xf32>
    %483 = stablehlo.divide %472, %482 : tensor<64x1x3x64xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %484 = stablehlo.reduce(%480 init: %cst_32) applies stablehlo.add across dimensions = [2] : (tensor<64x1x1x64xf32>, tensor<f32>) -> tensor<64x1x64xf32>
    %485 = stablehlo.transpose %483, dims = [0, 1, 3, 2] : (tensor<64x1x3x64xf32>) -> tensor<64x1x64x3xf32>
    %486 = stablehlo.transpose %471, dims = [0, 1, 3, 2] : (tensor<64x1x3x64xf32>) -> tensor<64x1x64x3xf32>
    %487 = stablehlo.dot_general %486, %arg10, batching_dims = [0, 2] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3xf32>, tensor<64x64x3xf32>) -> tensor<64x64x1xf32>
    %488 = stablehlo.dot_general %486, %arg9, batching_dims = [0, 2] x [0, 1], contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<64x1x64x3xf32>, tensor<64x64xf32>) -> tensor<64x64x1x3xf32>
    %489 = stablehlo.dot_general %488, %arg4, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<64x64x1x3xf32>, tensor<64x64x3x8xf32>) -> tensor<64x64x1x8xf32>
    %490 = stablehlo.broadcast_in_dim %arg11, dims = [] : (tensor<f32>) -> tensor<64x64x1xf32>
    %491 = stablehlo.multiply %490, %487 : tensor<64x64x1xf32>
    %492 = call @_where_8(%arg13, %491) : (tensor<64x64xi1>, tensor<64x64x1xf32>) -> tensor<64x1x64xf32>
    %493 = stablehlo.broadcast_in_dim %arg12, dims = [0, 2] : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %494 = stablehlo.multiply %492, %493 : tensor<64x1x64xf32>
    %495 = stablehlo.negate %494 : tensor<64x1x64xf32>
    %496 = stablehlo.add %484, %495 : tensor<64x1x64xf32>
    %497 = stablehlo.slice %489 [0:64, 0:64, 0:1, 0:1] : (tensor<64x64x1x8xf32>) -> tensor<64x64x1x1xf32>
    %498 = stablehlo.slice %489 [0:64, 0:64, 0:1, 1:2] : (tensor<64x64x1x8xf32>) -> tensor<64x64x1x1xf32>
    %499 = stablehlo.slice %489 [0:64, 0:64, 0:1, 2:3] : (tensor<64x64x1x8xf32>) -> tensor<64x64x1x1xf32>
    %500 = stablehlo.slice %489 [0:64, 0:64, 0:1, 3:4] : (tensor<64x64x1x8xf32>) -> tensor<64x64x1x1xf32>
    %501 = stablehlo.slice %489 [0:64, 0:64, 0:1, 4:5] : (tensor<64x64x1x8xf32>) -> tensor<64x64x1x1xf32>
    %502 = stablehlo.slice %489 [0:64, 0:64, 0:1, 5:6] : (tensor<64x64x1x8xf32>) -> tensor<64x64x1x1xf32>
    %503 = stablehlo.slice %489 [0:64, 0:64, 0:1, 6:7] : (tensor<64x64x1x8xf32>) -> tensor<64x64x1x1xf32>
    %504 = stablehlo.slice %489 [0:64, 0:64, 0:1, 7:8] : (tensor<64x64x1x8xf32>) -> tensor<64x64x1x1xf32>
    %505 = stablehlo.transpose %504, dims = [0, 2, 3, 1] : (tensor<64x64x1x1xf32>) -> tensor<64x1x1x64xf32>
    %506 = call @atleast_2d_9(%505) : (tensor<64x1x1x64xf32>) -> tensor<64x1x64xf32>
    %507 = stablehlo.transpose %503, dims = [0, 2, 3, 1] : (tensor<64x64x1x1xf32>) -> tensor<64x1x1x64xf32>
    %508 = call @atleast_2d_9(%507) : (tensor<64x1x1x64xf32>) -> tensor<64x1x64xf32>
    %509 = stablehlo.transpose %502, dims = [0, 2, 3, 1] : (tensor<64x64x1x1xf32>) -> tensor<64x1x1x64xf32>
    %510 = call @atleast_2d_9(%509) : (tensor<64x1x1x64xf32>) -> tensor<64x1x64xf32>
    %511 = stablehlo.transpose %501, dims = [0, 2, 3, 1] : (tensor<64x64x1x1xf32>) -> tensor<64x1x1x64xf32>
    %512 = call @atleast_2d_9(%511) : (tensor<64x1x1x64xf32>) -> tensor<64x1x64xf32>
    %513 = stablehlo.transpose %500, dims = [0, 2, 3, 1] : (tensor<64x64x1x1xf32>) -> tensor<64x1x1x64xf32>
    %514 = call @atleast_2d_9(%513) : (tensor<64x1x1x64xf32>) -> tensor<64x1x64xf32>
    %515 = stablehlo.transpose %499, dims = [0, 2, 3, 1] : (tensor<64x64x1x1xf32>) -> tensor<64x1x1x64xf32>
    %516 = call @atleast_2d_9(%515) : (tensor<64x1x1x64xf32>) -> tensor<64x1x64xf32>
    %517 = stablehlo.transpose %498, dims = [0, 2, 3, 1] : (tensor<64x64x1x1xf32>) -> tensor<64x1x1x64xf32>
    %518 = call @atleast_2d_9(%517) : (tensor<64x1x1x64xf32>) -> tensor<64x1x64xf32>
    %519 = stablehlo.broadcast_in_dim %506, dims = [0, 1, 3] : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %cst_33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %520 = stablehlo.pad %519, %cst_33, low = [0, 0, 5, 0], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<64x1x1x64xf32>, tensor<f32>) -> tensor<64x1x6x64xf32>
    %521 = stablehlo.broadcast_in_dim %508, dims = [0, 1, 3] : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %522 = stablehlo.pad %521, %cst_33, low = [0, 0, 4, 0], high = [0, 0, 1, 0], interior = [0, 0, 0, 0] : (tensor<64x1x1x64xf32>, tensor<f32>) -> tensor<64x1x6x64xf32>
    %523 = stablehlo.add %520, %522 : tensor<64x1x6x64xf32>
    %524 = stablehlo.broadcast_in_dim %510, dims = [0, 1, 3] : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %525 = stablehlo.pad %524, %cst_33, low = [0, 0, 3, 0], high = [0, 0, 2, 0], interior = [0, 0, 0, 0] : (tensor<64x1x1x64xf32>, tensor<f32>) -> tensor<64x1x6x64xf32>
    %526 = stablehlo.add %523, %525 : tensor<64x1x6x64xf32>
    %527 = stablehlo.broadcast_in_dim %512, dims = [0, 1, 3] : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %528 = stablehlo.pad %527, %cst_33, low = [0, 0, 2, 0], high = [0, 0, 3, 0], interior = [0, 0, 0, 0] : (tensor<64x1x1x64xf32>, tensor<f32>) -> tensor<64x1x6x64xf32>
    %529 = stablehlo.add %526, %528 : tensor<64x1x6x64xf32>
    %530 = stablehlo.broadcast_in_dim %514, dims = [0, 1, 3] : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %531 = stablehlo.pad %530, %cst_33, low = [0, 0, 1, 0], high = [0, 0, 4, 0], interior = [0, 0, 0, 0] : (tensor<64x1x1x64xf32>, tensor<f32>) -> tensor<64x1x6x64xf32>
    %532 = stablehlo.add %529, %531 : tensor<64x1x6x64xf32>
    %533 = stablehlo.broadcast_in_dim %516, dims = [0, 1, 3] : (tensor<64x1x64xf32>) -> tensor<64x1x1x64xf32>
    %534 = stablehlo.pad %533, %cst_33, low = [0, 0, 0, 0], high = [0, 0, 5, 0], interior = [0, 0, 0, 0] : (tensor<64x1x1x64xf32>, tensor<f32>) -> tensor<64x1x6x64xf32>
    %535 = stablehlo.add %532, %534 : tensor<64x1x6x64xf32>
    %536 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %537 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %538 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %539 = stablehlo.broadcast_in_dim %538, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %540 = stablehlo.broadcast_in_dim %536, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %541 = stablehlo.broadcast_in_dim %537, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %542 = stablehlo.transpose %535, dims = [0, 2, 1, 3] : (tensor<64x1x6x64xf32>) -> tensor<64x6x1x64xf32>
    %543 = stablehlo.broadcast_in_dim %539, dims = [1, 2] : (tensor<1x64xf32>) -> tensor<64x1x64xf32>
    %544 = stablehlo.broadcast_in_dim %540, dims = [1, 2] : (tensor<1x64xf32>) -> tensor<64x1x64xf32>
    %545 = stablehlo.broadcast_in_dim %541, dims = [1, 2] : (tensor<1x64xf32>) -> tensor<64x1x64xf32>
    %546 = stablehlo.transpose %542, dims = [1, 0, 2, 3] : (tensor<64x6x1x64xf32>) -> tensor<6x64x1x64xf32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %547:7 = stablehlo.while(%iterArg = %546, %iterArg_35 = %arg8, %iterArg_36 = %arg7, %iterArg_37 = %c, %iterArg_38 = %543, %iterArg_39 = %544, %iterArg_40 = %545) : tensor<6x64x1x64xf32>, tensor<6x64x64xf32>, tensor<64x64xf32>, tensor<i32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>
     cond {
      %c_41 = stablehlo.constant dense<6> : tensor<i32>
      %557 = stablehlo.compare  LT, %iterArg_37, %c_41,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %557 : tensor<i1>
    } do {
      %c_41 = stablehlo.constant dense<6> : tensor<i32>
      %557 = stablehlo.subtract %c_41, %iterArg_37 : tensor<i32>
      %c_42 = stablehlo.constant dense<1> : tensor<i32>
      %558 = stablehlo.subtract %557, %c_42 : tensor<i32>
      %c_43 = stablehlo.constant dense<0> : tensor<i32>
      %559 = stablehlo.dynamic_slice %iterArg, %558, %c_43, %c_43, %c_43, sizes = [1, 64, 1, 64] : (tensor<6x64x1x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x64x1x64xf32>
      %560 = stablehlo.reshape %559 : (tensor<1x64x1x64xf32>) -> tensor<64x1x64xf32>
      %561 = stablehlo.dynamic_slice %iterArg_35, %558, %c_43, %c_43, sizes = [1, 64, 64] : (tensor<6x64x64xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x64x64xf32>
      %562 = stablehlo.reshape %561 : (tensor<1x64x64xf32>) -> tensor<64x64xf32>
      %563:3 = func.call @None_10(%iterArg_36, %iterArg_38, %iterArg_39, %iterArg_40, %560, %562) : (tensor<64x64xf32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>, tensor<64x64xf32>) -> (tensor<64x1x64xf32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>)
      %564 = stablehlo.add %iterArg_37, %c_42 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_35, %iterArg_36, %564, %563#0, %563#1, %563#2 : tensor<6x64x1x64xf32>, tensor<6x64x64xf32>, tensor<64x64xf32>, tensor<i32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>
    }
    %548 = stablehlo.add %518, %547#4 : tensor<64x1x64xf32>
    %549 = stablehlo.add %548, %547#6 : tensor<64x1x64xf32>
    %550 = stablehlo.broadcast_in_dim %arg6, dims = [] : (tensor<f32>) -> tensor<64x1x64xf32>
    %551 = stablehlo.divide %549, %550 : tensor<64x1x64xf32>
    %cst_34 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %552 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<64x1x64xf32>
    %553 = stablehlo.multiply %552, %551 : tensor<64x1x64xf32>
    %554 = stablehlo.add %496, %553 : tensor<64x1x64xf32>
    %555 = call @norm_11(%arg1, %arg5, %554) : (tensor<64x64x3xf32>, tensor<64x64xf32>, tensor<64x1x64xf32>) -> tensor<64x1x64x3xf32>
    %556 = stablehlo.add %485, %555 : tensor<64x1x64x3xf32>
    return %556 : tensor<64x1x64x3xf32>
  }
  func.func private @_where_8(%arg0: tensor<64x64xi1>, %arg1: tensor<64x64x1xf32>) -> tensor<64x1x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<64x64xi1>) -> tensor<64x1x64xi1>
    %2 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %3 = stablehlo.transpose %arg1, dims = [0, 2, 1] : (tensor<64x64x1xf32>) -> tensor<64x1x64xf32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<1x64xf32>) -> tensor<64x1x64xf32>
    %5 = stablehlo.select %1, %3, %4 : tensor<64x1x64xi1>, tensor<64x1x64xf32>
    return %5 : tensor<64x1x64xf32>
  }
  func.func private @atleast_2d_9(%arg0: tensor<64x1x1x64xf32>) -> tensor<64x1x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<64x1x1x64xf32>, tensor<f32>) -> tensor<64x1x64xf32>
    return %0 : tensor<64x1x64xf32>
  }
  func.func private @None_10(%arg0: tensor<64x64xf32>, %arg1: tensor<64x1x64xf32>, %arg2: tensor<64x1x64xf32>, %arg3: tensor<64x1x64xf32>, %arg4: tensor<64x1x64xf32>, %arg5: tensor<64x64xf32>) -> (tensor<64x1x64xf32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>) {
    %0 = stablehlo.add %arg2, %arg4 : tensor<64x1x64xf32>
    %1 = stablehlo.negate %arg3 : tensor<64x1x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2] : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %3 = stablehlo.multiply %2, %arg3 : tensor<64x1x64xf32>
    %4 = stablehlo.add %0, %3 : tensor<64x1x64xf32>
    %5 = stablehlo.broadcast_in_dim %arg5, dims = [0, 2] : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %6 = stablehlo.multiply %arg3, %5 : tensor<64x1x64xf32>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x1x64xf32>
    %8 = stablehlo.multiply %7, %6 : tensor<64x1x64xf32>
    %9 = stablehlo.add %arg1, %8 : tensor<64x1x64xf32>
    return %9, %1, %4 : tensor<64x1x64xf32>, tensor<64x1x64xf32>, tensor<64x1x64xf32>
  }
  func.func private @norm_11(%arg0: tensor<64x64x3xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x1x64xf32>) -> tensor<64x1x64x3xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2] : (tensor<64x64xf32>) -> tensor<64x1x64xf32>
    %1 = stablehlo.multiply %arg2, %0 : tensor<64x1x64xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2] : (tensor<64x1x64xf32>) -> tensor<64x1x64x3xf32>
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<64x64x3xf32>) -> tensor<64x1x64x3xf32>
    %4 = stablehlo.multiply %3, %2 : tensor<64x1x64x3xf32>
    %5 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<64x64x3xf32>) -> tensor<64x1x64x3xf32>
    %6 = stablehlo.multiply %2, %5 : tensor<64x1x64x3xf32>
    %7 = stablehlo.add %4, %6 : tensor<64x1x64x3xf32>
    return %7 : tensor<64x1x64x3xf32>
  }
}
