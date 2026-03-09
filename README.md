# opencv-header-only
A lightweight header-only wrapper for OpenCV, enabling easy integration without building or linking against OpenCV binaries.


本库旨在提供一个最轻量的opencv库，能够通过头文件引入，解决用户80%的需求。
主要目标：用户不需要修改原始opencv代码，而只需要将命名空间从cv改成cvh就能兼容目前80%的opencv代码需求，并且能够达到80%的性能。

包含模块：
- core
    - 实现基本的Mat内存和计算。
- imgproc
    - 实现主要的图像处理库
- imgcodecs
    - 仅实现部分图像格式的读写操作

实现计划在doc中，主要api对齐opencv，一步步实现中，详细实施计划见：doc/header-only-opencv-plan.md
