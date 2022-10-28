# Scoped Buffered Persistency Model for GPUs

We provide the source code and setup for our GPU persistency model, Scoped Buffered Release Persistency (SBRP). SBRP is a scope-aware, buffered persistency model that provides high performance to GPU applications that wish to persist data on Non-Volatile Memory (NVM). SBRP modifies the GPU hardware and has been implemented using GPGPU-Sim, a GPU simulator. For more details on the simulator requirements, check the **[README](simulator/README.md)** in the simulator folder.

This repository consists of the source code of the simulator, benchmarks used for evaluation and all scripts needed to replicate the figures in the paper. 

We shall first explain how to replicate our results, then highlight the important files and folders contained in this repository.

## Setting up GPGPU-sim and running
To install cuda-11.4, follow:

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_470.57.02_linux.run
sudo sh cuda_11.4.1_470.57.02_linux.run
```

(Find detailed instructions at: https://developer.nvidia.com/cuda-11-4-1-download-archive?target_version=20.04) 
If your system does not have a GPU, do not install any CUDA drivers.
To run GPGPU-sim only the CUDA toolkit with the nvidia compiler (nvcc) is enough.  

For GPGPU-sim dependencies: 
```bash
sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev 
```
For cuda-sdk dependencies: 
```bash
sudo apt-get install libxi-dev libxmu-dev libglut3-dev 
```

Ensure that nvcc and cuda are in the PATH variable 
```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$CUDA_INSTALL_PATH/bin
```


## Running all figures and generating outputs 
```bash
make run_output_all 
```
Alternatively, one can first execute all the figures as: 
```bash
make run_all
```
And then generate all outputs using: 
```bash
make output_all
```

## Running figures individually

To execute figure 6, use the following command: 
```bash
make run_figure6
```

To execute figure 8, use the following command: 
```bash
make run_figure8
```

To execute figure 9, use the following command: 
```bash
make run_figure9
```

To execute figure 10a, use the following command: 
```bash
make run_figure10_a
```

To execute figure 10b, use the following command: 
```bash
make run_figure10_b
```

To execute figure 10c, use the following command: 
```bash
make run_figure10_c
```

## Generating output CSV files and graphs individually
To generate figure 6, use the following command: 
```bash
make output_figure6
```
To generate figure 8, use the following command: 
```bash
make output_figure8
```
To generate figure 9, use the following command: 
```bash
make output_figure9
```
To generate figure 10(a), use the following command: 
```bash
make output_figure10_a
```
To generate figure 10(b), use the following command: 
```bash
make output_figure10_b
```
To generate figure 10(c), use the following command: 
```bash
make output_figure10_c
```

## Source code
There are four main folders in this repository:
- **[benchmarks](benchmarks/)**: This folder contains the CUDA source code for the benchmarks evaluated. 
- **[models](models/)**: This folder contains the GPGPU-Sim configurations for all the different models evaluated.
- **[scripts](scripts/)**: This folder contains the python scripts for plotting the different graphs.
- **[simulator](simulator/)**: This is the main folder containing GPGPU-Sim and the source code for SBRP. The majority of the functionality for SBRP is encapsulated in [src/gpgpu-sim/persist.cc](simulator/src/gpgpu-sim/persist.cc), which contains the main implementation of the buffer. Other modifications were performed in the cache and memory hierarchy to support this new buffer.
