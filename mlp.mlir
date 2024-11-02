module @jit_mlp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128x128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x128xf32>, %arg11: tensor<128xf32>, %arg12: tensor<128x128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128x128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<1x128xf32>) -> (tensor<1x128xf32> {jax.result_info = ""}) {
    %0 = stablehlo.dot_general %arg16, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
    %3 = call @relu(%2) : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %4 = stablehlo.dot_general %3, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %5 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %6 = stablehlo.add %4, %5 : tensor<1x128xf32>
    %7 = call @relu(%6) : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %8 = stablehlo.dot_general %7, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %9 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %10 = stablehlo.add %8, %9 : tensor<1x128xf32>
    %11 = call @relu(%10) : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %12 = stablehlo.dot_general %11, %arg6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %13 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %14 = stablehlo.add %12, %13 : tensor<1x128xf32>
    %15 = call @relu(%14) : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %16 = stablehlo.dot_general %15, %arg8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %17 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %18 = stablehlo.add %16, %17 : tensor<1x128xf32>
    %19 = call @relu(%18) : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %20 = stablehlo.dot_general %19, %arg10, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %21 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x128xf32>
    %23 = call @relu(%22) : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %24 = stablehlo.dot_general %23, %arg12, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %25 = stablehlo.broadcast_in_dim %arg13, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %26 = stablehlo.add %24, %25 : tensor<1x128xf32>
    %27 = call @relu(%26) : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %28 = stablehlo.dot_general %27, %arg14, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %29 = stablehlo.broadcast_in_dim %arg15, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %30 = stablehlo.add %28, %29 : tensor<1x128xf32>
    return %30 : tensor<1x128xf32>
  }
  func.func private @relu(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x128xf32>
    return %1 : tensor<1x128xf32>
  }
}
