---
title: NeRF
draft: false
tags:
  - 3D_representation
description: Nerf is an implicit 3D repr
date: 2024-02-04
---
## TLDR

[Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) introduced a concept Neural Radiance Field(NeRF) to implicitly represent a static 3D scene. In a nutshell, NeRF is a simple MLP network which takes $(x,y,z,\theta,\phi)$ and computes $(RGB, \sigma)$.

[Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf) introduces a concept known as the Neural Radiance Field (NeRF) to represent static 3D scenes. Simply put, NeRF uses a type of neural network (Multi-Layer Perceptron; MLP) to translate a set of coordinates and viewing angles, $(x,y,z,\theta,\phi)$, into color and opacity values, $(RGB, \sigma)$.

## Introduction to Neural Radiance Fields

NeRF employs an MLP network to output the radiance (color information; RGB) at any given point in 3D space. This setup effectively creates a "(vector) field" where each point in space is assigned a color values, hence the name "Neural Radiance Field."

- **Direction-Dependent Color**: The color of an object changes based on where you're viewing it from. To capture this, NeRF considers the viewing direction. It can be identified by two angles $(\theta, \phi)$, but NeRF representes it by a normalized three-dimensional vector $\mathbf{d}$.
- **Capturing Opacity**: NeRF also outputs the opacity (denoted as $\sigma$) of a point, allowing it to represent occlusions or how objects block each other in 3D space.

## Digging into the Details

### Positional Encoding

In practice, deep networks like NeRF struggle to capture high-frequency details.[^1] To counter this, we introduce a higher-order mapping function, $\gamma$, similar to the positional encoding used in transformer models. This encoding amplifies the model's sensitivity to finer details.

$$
\begin{array}{l}
\text{Frequency Encoding}\\
\gamma: \mathbb{R} \to \mathbb{R}^{2L} \\
\gamma(x)=\sin(2^0\pi x)+\cos(2^0\pi x) + \cdots + \sin(2^{L-1}\pi x)+\cos(2^{L-1}\pi x)
\end{array}
$$
- We first normalize spatial coordinates $(x,y,z)$ to fall between -1 and +1. Then, each coordinate is encoded using $\gamma(\cdot)$, with $L=10$.
- Each component of 3D view direction unit vector $\mathbf{d}$ is also encoded using $\gamma(\cdot)$, with $L=4$.

### Hierarchical Sampling

NeRF models a 3D scene implicitly, meaning it doesn't use explicit geometric shapes. Instead, it uses density values ($\sigma$) to infer the presence of objects. But mapping an entire space directly is intractable, so NeRF uses a technique called _Ray Marching_ along with _hierarchical sampling_.

- **Efficient Sampling**: Initially, the model coarsely samples 64 points ($N_c$) across the space to estimate density distribution. It then refines this by focusing on areas with higher densities, selecting an additional 128 points ($N_f$).
- **Dual Network Training**: NeRF trains two parallel networks, one coarse and one fine, using the sampled points. The coarse network handles the initial 64 points, while the fine network processes both sets of points (64+128), leading to a more detailed final image.

### Challenges in Neural Radiance Fields

While Neural Radiance Fields (NeRF) present a groundbreaking approach to 3D scene representation, there are notable challenges that impact its performance and practicality:

1. **Slow Rendering Process**:
    - *Computational Intensity*: Rendering an image using NeRF is a computation-heavy process. For instance, rendering a medium-sized 400×400 image necessitates approximately 41 million MLP (Multi-Layer Perceptron) computations. This is due to the ray marching technique, which requires 256 MLP forward computations for each pixel, leading to prolonged rendering times.
2. **Slow Training**:
    - *Correlation with Rendering Speed*: The slow rendering speed directly impacts the training efficiency. Slow rendering equates to slower training iterations.
    - _Overfitting Requirement_: Additionally, the NeRF model needs to be overfitted to a specific scene, which inherently requires a substantial number of iterations. Typically, training a NeRF model for a single scene take around 100– 300k iterations to converge on a single NVIDIA V100 GPU (about 1–2 days).
3. **Cloudy Artifacts (Visual Floaters)**:
    - _Reconstruction Challenges_: In image-based reconstruction tasks, a simplistic approach might involve placing artifacts near each camera. While these artifacts might appear accurate in training views, slight camera movements can lead to significant discrepancies in the reconstructed image, manifesting as cloudy artifacts or visual floaters.
4. **Flickering Textures**:
    - _Sampling Limitations_: NeRF involves sampling a finite number of points in a continuous space, which can lead to missing finer details encoded in the MLP. As the camera moves, the sampling points in the scene change, causing some details to be intermittently included or excluded. This results in flickering textures on the rendered objects, affecting the visual consistency.

## Volume Rendering

These are simple notes on volume rendering equations, not a comprehensive explanation.

### Differential Form

- ${\bf r}(t) = {\bf o} + t{\bf d}$ : camera ray
	- ${\bf o}$: origin; position of a camera
	- ${\bf d}$: viewing direction
	- $t \in [t_n, t_f]$
- ${\bf c}({\bf r}(t),\mathrm{d})$: view dependent radiance
- $\sigma({\bf r}(t))$: volume density; the differential probability of a ray terminating at an infinitesimal particle at location ${\bf x}$
- $T(t)=\exp\left(-\int_{t_{n}}^{t}\sigma({\bf r}(s))d s\right)$: accumulated transmittance along the ray from $t_n$ to $t$.

$$
\begin{aligned}
&\text{Expected Color}\\

&C({\bf r})=\int_{t_{n}}^{t_{f}} T(t)\sigma({\bf r}(t)){\bf c}({\bf r}(t),{\bf d})dt
\end{aligned}
$$

### Discrete Form

$\hat{C}({\bf r})$ estimates $C({\bf r})$ with the quadrature rule.

- $\delta_i = t_{i+1} - t_i$: distance between adjacent samples
- $T_i = \exp\left(- \sum^{i-1}_{j=1} \sigma_j \delta_j \right)$

$$
\begin{aligned}
&\text{Estimated Expected Color}\\

&\hat{C}({\bf r})=\sum^{N}_{i=1}T_i \left(1-\exp(-\sigma_i \delta_i)\right) {\bf c}_i
\end{aligned}
$$
This equation reduces to traditional alpha compositing with $\alpha_i = 1-\exp(-\sigma_i \delta_i)$.

[^1]: Rahaman, N., Baratin, A., Arpit, D., Dr¨axler, F., Lin, M., Hamprecht, F.A., Bengio, Y., Courville, A.C.: On the spectral bias of neural networks. In: ICML (2018)