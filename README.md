# Record Linkage Process

This project exposes the process of linking records, known as Record Linkage, between large databases using heterogeneous computing systems that offer high computional power, such as:

- [CUDA C]
- [MIC]

### To execute

- CUDA C code:

To run cuda c code, two prerequisites are necessary: your computer must have some **nvidia device** and must have installed **NVIDIA CUDA Compiler** ([NVCC](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4Rnk5ZlXr)) on the machine.

```sh
$ cd cuda-c
$ make
$ ./linkage <num_threads_per_block>
```

[MIC]: <http://www.intel.com/content/www/us/en/architecture-and-technology/many-integrated-core/intel-many-integrated-core-architecture.html>
[CUDA C]: <http://www.nvidia.com/object/cuda_home_new.html>
