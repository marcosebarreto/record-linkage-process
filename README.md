# Record Linkage Process

Examine large databases from different domains in the search for records that represent a same entity in the real world is a problem known as Record Linkage process. Because it has a high computational cost, this project exposes different solutions to this problem using **heterogeneous computing systems** that offer high computational power, such as:

- [OpenMP]
- [CUDA C]
- [MIC]

### To execute

* OpenMP code:

    To compile openmp code, it's necessary to put **-fopenmp** directive in the compilation time. This mean that openmp code will be use as many threads as available cores on the computer.

    ```sh
    $ cd openmp
    $ gcc -fopenmp linkage.c -o linkage
    $ ./linkage <problem_size> <num_threads>
    ```

* CUDA C code:

    To run cuda c code, two prerequisites are necessary: your computer must have some **nvidia device** and must have installed **NVIDIA CUDA Compiler** ([NVCC](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4Rnk5ZlXr)) on the computer.

    * one-GPU:

        ```sh
        $ cd cuda-c/one-GPU
        $ make clean
        $ make
        $ ./linkage <num_threads_per_block> <larger_file>
        ```

    * multi-GPU:

        ```sh
        $ cd cuda-c/multi-GPU
        $ make clean
        $ make
        $ ./linkage <num_threads_per_block> <file1> <threads_openmp> <percentage_each_gpu> <qtd_gpu>
        ```


#### Update logs

> **12/17/2016** : cuda c code for one gpu added.
> **02/01/2017** : openmp code added.
> **02/03/2017** : cuda c code for two or manu gpus added.

[OpenMP]: <http://www.openmp.org/>
[MIC]: <http://www.intel.com/content/www/us/en/architecture-and-technology/many-integrated-core/intel-many-integrated-core-architecture.html>
[CUDA C]: <http://www.nvidia.com/object/cuda_home_new.html>
