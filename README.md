# BPNetWork

A simple **Back-Propagation Neural Network** implemented in C++11.

## Network Architecture

```
Input (2) --> Hidden (4) --> Output (1)
```

- Activation: Sigmoid
- Training: Online SGD (stochastic gradient descent)
- Loss: Mean Squared Error

## Configuration

Edit `lib/Config.h` to adjust hyperparameters:

| Parameter    | Default | Description          |
|--------------|---------|----------------------|
| `INNODE`     | 2       | Input layer size     |
| `HIDENODE`   | 4       | Hidden layer size    |
| `OUTNODE`    | 1       | Output layer size    |
| `lr`         | 0.1     | Learning rate        |
| `threshold`  | 1e-4    | Convergence threshold|
| `max_epoch`  | 1e6     | Max training epochs  |

## Data Format

**Training data** (`data/traindata.txt`): space-separated values, one sample per line.
```
feature1 feature2 label
0 0 0
0 1 1
1 0 1
1 1 0
```

**Test data** (`data/testdata.txt`): features only, no labels.
```
feature1 feature2
0.111 0.112
0.001 0.999
```

## Build & Run

```bash
mkdir build && cd build
cmake ..
make
./BPNN
```

Requires CMake 2.8.12+ and a C++11-compatible compiler.

## Author

Despacato — dlmu_zxg@163.com

## Acknowledgement

This project is fully based on [GavinTechStudio/Back-Propagation-Neural-Network](https://github.com/GavinTechStudio/Back-Propagation-Neural-Network) for learning purposes. Many thanks to GavinTechStudio for the excellent reference implementation.
