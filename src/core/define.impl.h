//
// Created by mzh on 2023/11/2.
//

#ifndef MINFER_DEFINE_IMPL_H
#define MINFER_DEFINE_IMPL_H

// usefull macro

#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)

#include <mutex>
namespace minfer
{
    typedef std::recursive_mutex Mutex;
    typedef std::lock_guard<Mutex> AutoLock;
}

#endif //MINFER_DEFINE_IMPL_H
