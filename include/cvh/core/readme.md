Core module contains basic of  folder contains all basic function.

主要包括：
Mat基础定义

TODO：
- 
- 移除所有MINFER风格的代码，那是我另一个项目，和这个无关。
- 数据类型的对齐：目前的数据类型命名和OpenCV不一样，需要针对性的修改，从CV_8U到CV_8U等。
- 错误处理函数：对齐OpenCV，M_Error 改成OpenCV形式的接口
- 通道类型的移植：目前Mat是没有通道的概念，只有数据类型的概念，这部分需要完全对齐到OpenCV。
- 迁移C++到header file，目前一部分实现在src/core中，需要完全迁移到 include/cvh/core中去。
- 为了防止和原始的命名空间冲突，全局使用cvh作为外层命名空间。