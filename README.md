# jax-demo

A small test demo for using jax, nnx, and optax to train a simple relu-activated MLP on MNIST for 10 epochs and visualize outputs.
Largely adapted from the [demo](https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers) for the Titanic dataset from Google.


## Setup
Using conda:

```
$ conda create -n jax-demo python=3.9
$ conda activate jax-demo
(jax-demo) $ pip install -r requirements.txt
```

Then, install the corresponding Jax library:

For NVIDIA GPUs running CUDA, use

```
(jax-demo) $ pip install -U "jax[X]"
```

With X = (your CUDA version), e.g. a GPU w/ CUDA 12 needs `jax[cuda12]` whilst a setup planning to use Google's cloud TPUs would want `jax[tpu]`.
CPU-only setups would want `jax`.

This environment is now set up to serve as your notebook kernel.