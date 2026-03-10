//
// Created by mzh on 2023/11/2.
//

#ifndef CVH_MEMORY_UTILS_H
#define CVH_MEMORY_UTILS_H

#include <stdio.h>
#include "cvh/core/define.h"
#include "define.impl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CV_MEMORY_ALIGN_DEFAULT 64

/**
 * @brief alloc memory with given size & alignment.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 */
CV_EXPORTS void* MMemoryAllocAlign(size_t size, size_t align = CV_MEMORY_ALIGN_DEFAULT);

/**
 * @brief alloc memory with given size & alignment, and fill memory space with 0.
 * @param size  given size. size should > 0.
 * @param align given alignment.
 * @return memory pointer.
 * @warning use `MNNMemoryFreeAlign` to free returned pointer.
 * @sa MNNMemoryFreeAlign
 */
CV_EXPORTS void* MMemoryCallocAlign(size_t size, size_t align = CV_MEMORY_ALIGN_DEFAULT);

/**
 * @brief free aligned memory pointer.
 * @param mem   aligned memory pointer.
 * @warning do NOT pass any pointer NOT returned by `MNNMemoryAllocAlign` or `MNNMemoryCallocAlign`.
 * @sa MNNMemoryAllocAlign
 * @sa MNNMemoryCallocAlign
 */
CV_EXPORTS void MMemoryFreeAlign(void* mem);


#ifdef __cplusplus
}
#endif
#endif //CVH_MEMORY_UTILS_H
