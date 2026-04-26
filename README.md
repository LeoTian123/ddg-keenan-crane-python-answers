# DDG (Keenan Crane) Python Exercise Solutions

## Project Overview

This repository contains Python implementations and solutions for the book
*Discrete Differential Geometry: An Applied Introduction* by Keenan Crane.

The project implements all core algorithms mentioned in the book and provides 3D visualization based on pyvista. The code is modular and can be directly used as a lightweight toolkit for learning and developing discrete differential geometry.

If you can read Chinese, [this article](https://zhuanlan.zhihu.com/p/2028207376482714165) introduces the core content of the book together with the code in this repository.

## Implemented Content

* `closed_surface.py`: Encapsulates closed, connected discrete surfaces represented by triangle meshes, including topological and geometric information such as vertices, edges, and faces. Implements core algorithms such as Gaussian curvature / angle defect, Laplace operators and smoothing, Hodge star and exterior derivatives, generators, harmonic bases, connection and vector field design.

* `pyvista_wrapped.py`: A unified 3D visualization interface based on pyvista, supporting mesh coloring, path visualization, visualization of 1-forms on edges and dual edges, switching between multiple 1-forms, and tangent vector field visualization on faces.

## Main Features of Example Scripts

* `01-gaussian-and-laplacian.py`: Computes and visualizes Gaussian curvature / angle defect, verifies the Gauss–Bonnet theorem; visualizes Laplace operators; compares forward/backward Euler smoothing and shows geometric changes before and after smoothing.

* `02-conformal-mapping.py`: Performs a parameterization using eigenfunctions of the Laplacian matrix, mapping a closed surface to the plane and visualizing the result with a 2D scatter plot. Note that this is not a conformal mapping, but a simplified alternative.

* `03-hodge-decomposition.py`: Performs Hodge decomposition on discrete 1-forms, verifying decomposition into gradient, curl, and harmonic components, and checks the properties of each component.

* `04-vector-field-design.py`: Constructs generators of the $H_1$ homology group and harmonic bases of 1-forms, computes the holonomy of the Levi-Civita connection, designs vector fields with prescribed singularities, and generates tangent vector fields on faces.

* `11-volume-preserving-smoothing.py`: Added in the first additional update. Performs iterative volume-preserving Laplacian smoothing on surfaces.

* `12-conformal-mapping.py`: Added in the first additional update. Fully implemented by AI, this script performs a proper conformal mapping for closed surfaces. See the header comments in the file for details. Outputs are saved under `output_lscm/`.

## Usage

Run the example scripts directly to test the algorithms and visualization.

The modules can also be imported as a toolkit for your own geometry processing projects.

## About the `prototype/` Folder

This folder contains an outdated version of the code, using pyOpenGL for visualization. The code is relatively complex and poorly encapsulated, and is provided for reference only.

Notably:

* `prototype/61-拉普拉斯平滑-2.py`: This script caches all intermediate results of the Laplacian and allows controlling the number of iterations displayed via the mouse wheel, a feature not implemented in later versions.

* The visualizations in this folder support adjusting the light position via right-click.

## Additional Notes

This is a partial solution for personal learning purposes.

The implementation of holonomy and the code for generating vector fields from a connection were implemented with the help of AI.

For the full DDG course and official materials, please visit:
[https://www.cs.cmu.edu/~kmcrane/Projects/DDG/](https://www.cs.cmu.edu/~kmcrane/Projects/DDG/)

----------------------

# DDG（Keenan Crane）的 Python 习题答案

## 项目简介

本仓库包含书籍《DISCRETE DIFFERENTIAL GEOMETRY: AN APPLIED INTRODUCTION》（Keenan Crane）的 Python 实现与答案。

项目实现了本书提到的所有核心算法，并提供基于 pyvista 的 3D 可视化渲染。代码结构模块化，可直接作为轻量级工具包用于离散微分几何的学习与开发。

如果你能看懂中文，[这篇文章](https://zhuanlan.zhihu.com/p/2028207376482714165)介绍了关于这本书的核心内容，结合本仓库的代码。

## 实现内容
- ```closed_surface.py```: 封装了三角 mesh 的闭合连通离散曲面，包含顶点、边、面的拓扑与几何信息，实现了高斯曲率/角亏、拉普拉斯算子与平滑、Hodge 星与外微分算子、生成元、调和基、connection与向量场设计等核心算法。

- ```pyvista_wrapped.py```: 基于 pyvista 的统一三维可视化接口，支持网格着色、路径可视化、边与对偶边上的 1‑form 可视化、多组 1‑form 的切换查看，以及面上的切向量场显示。

## 示例脚本主要功能

- ```01-gaussian-and-laplacian.py```: 计算并可视化高斯曲率/角亏，验证高斯–博内定理；可视化拉普拉斯算子；对比前向/后向欧拉平滑，展示平滑前后的几何变化。

- ```02-a-simple-mapping.py```: 通过拉普拉斯矩阵的特征函数进行参数化，将封闭曲面映射到平面，并用二维散点图展示。请注意这不是一种共形映射，而是另一种简化的映射。

- ```03-hodge-decomposition.py```: 对离散 1‑form 进行 Hodge 分解，验证分解为梯度部分、旋度部分和调和部分，并检查各分量的性质。

- ```04-vector-field-design.py```: 构造曲面的 $H_1$ 同调群生成元与 1‑form 调和基底，计算 Levi‑Civita connection 的 holonomy，设计带指定奇点的向量场并在面上生成切向量场。

- ```11-volume-preserving-smoothing.py```: 第一次额外更新加入。对曲面迭代执行保持的体积拉普拉斯平滑。

- ```12-conformal-mapping.py```: 第一次额外更新加入。完全由AI实现，实现了正确的对于闭曲面的共形映射，详细情况见文件开篇的注释。其输出在```output_lscm/```下。

## 使用方法

直接运行以上示例脚本即可测试算法与可视化效果，打开脚本可做细致编辑。

可将模块作为工具包导入，用于你自己的几何处理项目。

## 关于 ``prototype/``` 文件夹

里面是一套过时的老版本代码，使用 pyOpenGL 做可视化，较为繁复，且代码封装程度很低。仅作参考使用。

其中值得一提的是：

- ```prototype/61-拉普拉斯平滑-2.py``` 文件，它缓存了拉普拉斯的所有中间计算结果，并使用滚轮控制显示的迭代次数，是后续版本没有实现的功能。

- 这些代码的可视化都支持右键调节光源位置。

## 其他说明

这是个人学习用的部分答案。

关于 holonomy 的代码，以及如何从一个 connection 生成向量场的代码是 AI 实现的。

完整 DDG 课程与官方资料请访问：https://www.cs.cmu.edu/~kmcrane/Projects/DDG/
