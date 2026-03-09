#include "memory_utils.h"
#include "minfer/system.h"

static inline void **alignPointer(void **ptr, size_t alignment) {
    return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
}

extern "C" void *MMemoryAllocAlign(size_t size, size_t alignment) {
    M_Assert(size > 0);

#ifdef MU_DEBUG_MEMORY
    return malloc(size);
#else
    void **origin = (void **)malloc(size + sizeof(void *) + alignment); // 这个size是以byte为单位。
    M_Assert(origin != NULL);
    if (!origin) {
        return NULL;
    }

    void **aligned = alignPointer(origin + 1, alignment);
    aligned[-1]    = origin;
    return aligned;
#endif
}

extern "C" void *MMemoryCallocAlign(size_t size, size_t alignment) {
    M_Assert(size > 0);

#ifdef MU_DEBUG_MEMORY
    return calloc(size, 1);
#else
    void **origin = (void **)calloc(size + sizeof(void *) + alignment, 1);
    M_Assert(origin != NULL);
    if (!origin) {
        return NULL;
    }
    void **aligned = alignPointer(origin + 1, alignment);
    aligned[-1]    = origin;
    return aligned;
#endif
}

extern "C" void MMemoryFreeAlign(void *aligned) {
#ifdef MU_DEBUG_MEMORY
    free(aligned);
#else
    if (aligned) {
        void *origin = ((void **)aligned)[-1];
        free(origin);
    }
#endif
}
