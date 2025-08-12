# jax-demo
Learning jax, a few reference implementations here for later as well as some notes.

## Setup
With python @3.10:
```
$ conda create -n jax-demo python=3.9
$ conda activate jax-demo
(jax-demo) $ pip install -r requirements.txt
(jax-demo) $ pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then, install the corresponding Jax library:

For NVIDIA GPUs running CUDA, use

```
(jax-demo) $ pip install -U "jax[X]"
```

With X = (your CUDA version), e.g. a GPU w/ CUDA 12 needs `jax[cuda12]` whilst a setup planning to use Google's cloud TPUs would want `jax[tpu]`.
CPU-only setups would want `jax`.

This environment is now set up to serve as your notebook kernel.