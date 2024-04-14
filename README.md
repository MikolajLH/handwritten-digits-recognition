# Overview

**Multilayer perceptron** written from scratch in **C++23**.

There are 2 header-only libraries: `matvec.h` and `neural_network.h`.

**`matvec.h`** is header-only library implementing matrices with compile time sizes.
```c++
#include "matvec.h"

mv::Matrix<double, 3, 4> M{1., 2. , 3., 4., 5.};
/*
M = 1 2 3 4
    5 0 0 0
    0 0 0 0
*/

M(-1, -1) = 7;

auto N = mv::mat_mul(M, M.transpose());
static_assert(std::is_same<mv::Matrix<double, 3, 3>, decltype(N));

auto A = mv::Matrix<int, 1, 6>([](size_t r, size_t c){ return r + c;});

// mv::mat_mul(M, A); // gives compile time error since M.cols() != A.rows()

auto rowM = A.get_row(0);
auto colM = B.get_col(-1);
/*
rowM = 1 2 3 4
colM = 4
       0
       7
*/
auto Z = mv::mat_mul(colM, rowM).reshape<1, 12>().apply([](int x){ return x*x; });
// multiplies colM anc rowM ( 3x1 * 1x4 = 3x4 ),
// reshapes resulting matrix into 1x12 row vector,
// applies f(x) = x*x element-wise.
```

**`neural_network.h`** is header-only library implementing the multi-layer perceptron build upon the `matvec.h` library.
```c++
#include "neural_network.h"

int main(){
  nn::MLP<double, 28 * 28, 15, 10> mlp(mv::FillTag(1.));
  mlp.get_function<1>() = mv::MatrixFunction::relu();
  mlp.get_function<2>() = mv::MatrixFunction::softmax();
  mlp.randomize();

  const double eta = 0.001;
  auto X = mv::Matrix<double,28 * 28, 1>(1.);
  auto Y = mv::Matrix<double, 10, 1>(0.);

  mlp.reset_errors();
  mlp.get_input() = X;
  mlp.feed_forward();
  const auto Y_hat = mlp.get_output();
  const auto dCx = A - Y;
  mlp.backpropagate(dCx);
  mlp.multiply_weights_deltas(eta);
  mlp.apply_weights_deltas();
}
```
