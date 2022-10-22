# Taichi NGP Renderer

[![License](https://img.shields.io/badge/license-Apache-green.svg)](LICENSE)

This is an [Instant-NGP](https://github.com/NVlabs/instant-ngp) renderer implemented using [Taichi](https://github.com/taichi-dev/taichi), which is written completely in Python. **No CUDA!** This repository only implemented the rendering part of the NGP but is more simple and has a lesser amount of code compared to the original (Instant-NGP and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)).

<!-- It is fun to write this code that implements even just the rendering part of the NGP, especially when you compare it original repo, which is a hundred lines verse thousands (Instant-NGP and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)). -->

<p align="center">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/example.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/lego_depth.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/interaction.gif", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/samples.gif", width="24%">
  <br>
</p>

## Installation

Install Taichi to get started:

```bash
pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
```
> **Note**
> There is a bug in Taichi codegen that a `uint32` value can not properly work on modulo operation. You have to install nightly version in which they fix this bug. See detail in issue [#6118](https://github.com/taichi-dev/taichi/issues/6118)

Clone this repository and install the required package:

```bash
git clone https://github.com/Linyou/taichi-ngp-renderer.git
python -m pip install -r requirement.txt
```

## Description

This repository only implemented the forward part of the **Instant-NGP**, which include:

- Rays intersection with bounding box: `ray_intersect()`
- Ray marching strategic: `raymarching_test_kernel()`
- Spherical harmonics encoding for ray direction: `dir_encode()`
- Hash table encoding for 3d coordinate: `hash_encode()`
- Fully Fused MLP using shared memory: `sigma_layer()`, `rgb_layer()`
- Volume rendering: `composite_test()`

<!-- Since this repository only implemented the rendering code, so you have to load trained parameters from a pre-trained scene (This repository provides a lego scene). -->

However, there are some differences compared to the original:

###### Missing function

- Taichi currently missing `frexp()` method, so I have to use a hard-coded scale which is 0.5. I will update the code once **Taichi** supports this function.

###### Fully Fused MLP

- Instead of having a single kernel like **tiny-cuda-nn**, this repo use separated kernel `sigma_layer()` and `rgb_layer()` because the shared memory size that **Taichi** currency allow is `48KB` as issue [#6385](https://github.com/taichi-dev/taichi/issues/6385) points out. This could be improved in the future.
- In the **tiny-cuda-nn**, they use TensorCore for `float16` multiplication, which is not an accessible feature for Taichi, so I directly convert all the data to `ti.float16` to speed up the computation.

## GUI

This code supports real-time rendering GUI interactions with less than 1GB VRAM:

- Control the camera with the mouse and keyboard
- Changing the number of samples for each ray while rendering
- Rendering with different resolution
- Support video recording and export video and snapshot (need [ffmpeg](https://docs.taichi-lang.org/docs/export_results#install-ffmpeg-on-windows))
- Up to 17 fps on a 3090 GPU at 800 $\times$ 800 resolution (using default pose)

Simply run `python taichi_ngp.py --gui --scene lego` to start the GUI. This repository provided 8 pre-trained NeRF synthesis scenes: _Lego, Ship, Mic, Materials, Hotdog, Ficus, Drums, Chair_

<p align="center">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/lego.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/ship.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/mic.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/materials.png", width="24%">
  <br>
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/hotdog.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/ficus.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/drums.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/chair.png", width="24%">
  <br>
</p>

Running `python taichi_ngp.py --gui --scene <name>` will automatically download pre-trained model `<name>` in the `./npy_file` folder. For more options, please check out the argument parameters in `taichi_ngp.py`.

## Custom scene

You can train a new scene with [ngp_pl](https://github.com/kwea123/ngp_pl), and save the pytorch model to numpy using `np.save()`. After that, use `--model_path` argument to specify the model file.

## Todo

- [ ] Refactor to separate modules
- [ ] Support Vulkan backend
- [ ] Support real scene
...
