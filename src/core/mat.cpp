//
// Created by mzh on 2024/1/31.
//

#include "cvh/core/mat.h"
#include "cvh/core/system.h"
#include "cvh/core/utils.h"
#include "cvh/core/saturate.h"
#include "cvh/core/define.h"
#include <stdatomic.h>
#include <cstring>
#include <vector>

// If is not Mac or IOS, then include the following code.
#ifndef __APPLE__
static inline
void memset_pattern4(void *data, const void *pattern4, size_t len)
{
    std::fill_n(reinterpret_cast<uint8_t*>(data), len, *reinterpret_cast<uint8_t*>(&pattern4));
}
#endif

namespace cvh
{

template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    assert((n & (n - 1)) == 0); // n is a power of 2
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

#define  MAT_MALLOC_ALIGN    64
void* fastMalloc(size_t size)
{
    // size + one more pointer + alignment_size
    uchar* udata = (uchar*) malloc(size + sizeof(void*) + MAT_MALLOC_ALIGN);
    if (!udata)
    {
        std::cerr<<"Out of memory in fastMalloc"<<std::endl;
        return nullptr;
    }

    uchar** adata = alignPtr((uchar**)udata + 1, MAT_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void fastFree(void* ptr)
{
    if (ptr)
    {
        uchar* udata = ((uchar**)ptr)[-1];
        assert(udata < (uchar*) ptr && ((uchar*)ptr - udata) <= (sizeof(void *) + MAT_MALLOC_ALIGN));
        free(udata);
    }
}

// <<<<<<<<<<<<<<<<<<<<<   MatData   >>>>>>>>>>>>
MatData::MatData(const MatAllocator *_allocator)
{
    allocator = _allocator;
    refcount = 0;
    data = 0;
    size = 0;
    flags = static_cast<MatData::MemoryFlag>(0);
}

MatData::~MatData()
{
    allocator = 0;
    refcount = 0;
    data = 0;
    flags = static_cast<MatData::MemoryFlag>(0);
}


// <<<<<<<<<<<<<<<<<<<<<   StdMatAllocator   >>>>>>>>>>>>
class StdMatAllocator : public MatAllocator
{
public:

    MatData* allocate(int dims, const int* sizes, int type, void* data0) const override
    {
        size_t total = CV_ELEM_SIZE(type);

        for (int i = dims -1; i >= 0; i--)
        {
            total *= sizes[i];
        }

        uchar* data = data0 ? (uchar*)data0 : (uchar*) fastMalloc(total);
        MatData* u = new MatData(this);
        u->data = data;
        u->size = total;
        if (data0)
            u->flags = MatData::USER_MEMORY;

        return u;
    }

    // check if the MatData is allocated?
    bool allocate(MatData* u, MatDataUsageFlags) const override
    {
        if (!u) return false;
        return true;
    }

    void deallocate(MatData* u) const override
    {
        if(!u)
            return;

        assert(u->refcount == 0);
        if (u->flags != MatData::USER_MEMORY)
        {
            fastFree(u->data);
            u->data = 0;
        }

        delete u;
    }
};

void MatAllocator::map(MatData*) const
{
    // do nothing
}

void MatAllocator::unmap(MatData* u) const
{
    if(u->refcount == 0)
    {
        deallocate(u);
    }
}

// <<<<<<<<<<<<<<<<<<<<<   MatSize   >>>>>>>>>>>>
MatSize::MatSize(int *_p):p0(_p)
{
    if (_p)
        p = _p + 1;
    else
        p = nullptr;
}

int MatSize::dims() const
{
    return p0[0];
}

const int& MatSize::operator[](int i) const
{
    return p[i];
}

int& MatSize::operator[](int i)
{
    return p[i];
}

bool MatSize::operator!=(const MatSize &sz) const
{
    return !(*this == sz);
}

bool MatSize::operator==(const MatSize &sz) const
{
    int d = dims();
    int dsz = sz.dims();

    if (d != dsz)
        return false;

    for (int i = 0; i < d; i++)
    {
        if (p[i] != sz[i])
            return false;
    }

    return true;
}

// <<<<<<<<<<<<<<<<<<<<<   Mat   >>>>>>>>>>>>
// Setting the dim for mat.
void _setSize(Mat& m, int _dim, const int* _sz)
{
    M_Assert(_dim <= MAT_MAX_DIM && _dim >= 0);

    if (_dim != m.dims)
    {
        fastFree(m.size.p0); // free and reset the dim
        m.size.p0 = (int *) fastMalloc((_dim + 1) * sizeof( int)); // one more for dim
        m.size.p = m.size.p0 + 1;
        m.size.p0[0] = _dim;
    }

    m.dims = _dim;

    if (!_sz)
        return;

    for (int i = _dim - 1; i >= 0; i--)
    {
        int s = _sz[i];
        if (s <= 0)
        {
            m.release();
            break;
        }
        M_Assert(s > 0);
        m.size.p[i] = s;
    }
}

static
MatAllocator*& getDefaultAllocatorMatRef()
{
    static MatAllocator* g_matAllocator = Mat::getStdAllocator();
    return g_matAllocator;
}

MatAllocator *Mat::getDefaultAllocator()
{
    return getDefaultAllocatorMatRef();
}

MatAllocator *Mat::getStdAllocator()
{
    static MatAllocator* const allocator = new StdMatAllocator();
    return allocator;
}

void Mat::copySize(const Mat &m)
{
    // This code will free and alloc a new buffer for m.size.
    _setSize(*this, m.dims, 0);

    for (int i = 0; i < dims; i++)
    {
        size[i] = m.size[i];
    }
}

// Create empty Mat
Mat::Mat()
:dims(0), data(0), allocator(0), u(0), size(0), matType(CV_32F)
{
}

Mat::Mat(int _dims, const int* _sizes, int _type)
:dims(0), data(0), allocator(0), u(0), size(0), matType(CV_32F)
{
    create(_dims, _sizes, _type);
}

Mat::Mat(const std::vector<int> _sizes, int _type)
:dims(0), data(0), allocator(0), u(0), size(0), matType(CV_32F)
{
    create(_sizes, _type);
}

Mat::Mat(int _dims, const int* _sizes, int _type, int v)
:dims(0), data(0), allocator(0), u(0), size(0), matType(_type)
{
    M_Assert(_type == CV_32S || _type == CV_32U || _type == CV_32F);
    int type = _type;
    create(_dims, _sizes, type);

    // set all value to v
    size_t allSize = this->total();
    int* d = (int *)this->data;
    for (size_t i = 0; i < allSize; i++)
    {
        d[i] = v;
    }
}

Mat::Mat(const std::vector<int> _sizes, int _type, int v)
:dims(0), data(0), allocator(0), u(0), size(0), matType(_type)
{
    M_Assert(_type == CV_32S || _type == CV_32U || _type == CV_32F);
    int type = _type;
    create(_sizes, type);

    // set all value to v
    size_t allSize = this->total();
    int* d = (int *)this->data;
    for (size_t i = 0; i < allSize; i++)
    {
        d[i] = v;
    }
}

Mat::Mat(const Mat& m)
:dims(m.dims), data(m.data), allocator(m.allocator), u(m.u), size(0), matType(m.matType)
{
    if (u)
    {
        //    std::atomic_fetch_add((_Atomic(int)*)(&u->refcount), 1);
        CV_XADD(&u->refcount, 1);
    }

    dims = 0; // reset dims
    copySize(m);
}

Mat::Mat(int _dims, const int* _sizes, int _type, void* _data)
:dims(0), data(0), allocator(0), u(0), size(nullptr), matType(_type)
{
    data = (uchar*)_data;
    _setSize(*this, _dims, _sizes);
}

Mat::Mat(const std::vector<int> _sizes, int _type, void* _data)
:dims(0), data(0), allocator(0), u(0), size(nullptr), matType(_type)
{
    data = (uchar*)_data;
    _setSize(*this, _sizes.size(), _sizes.data());
}

Mat::~Mat()
{
    release();
    fastFree(size.p0);
}

// Not copy mat data, only add reference counter.
Mat& Mat::operator=(const Mat& m)
{
    if (this != &m) // check if there are same Mat.
    {
        if (m.u)
            CV_XADD(&m.u->refcount, 1);

        release(); // release this resource first.
        copySize(m);

        data = m.data;
        allocator = m.allocator;
        u = m.u;
        matType = m.matType;
    }

    return *this;
}

Mat& Mat::operator=(const float v)
{
    if (this->empty())
        return *this;

    size_t total_v = total();

    if (type() == CV_32F)
    {
        float* p = (float*)data;
        for (int i = 0; i < total_v; i++)
        {
            p[i] = (float)v;
        }
    }
    else if (type() == CV_32S)
    {
        int* p = (int*)data;
        for (int i = 0; i < total_v; i++)
        {
            p[i] = (int)v;
        }
    }
    else
        M_Error_(NULL, ("Unsupported format at function \"Mat::operator= \" type = %d!", type()));

    return *this;
}

Mat& Mat::operator=(const int v)
{
    if (this->empty())
        return *this;

    size_t total_v = total();

    if (type() == CV_32F)
    {
        float* p = (float*)data;
        for (int i = 0; i < total_v; i++)
        {
            p[i] = (float)v;
        }
    }
    else if (type() == CV_32S)
    {
        int* p = (int*)data;
        for (int i = 0; i < total_v; i++)
        {
            p[i] = (int)v;
        }
    }
    else
        M_Error_(NULL, ("Unsupported format at function \"Mat::operator= \" type = %d!", type()));

    return *this;
}

// Mat& Mat::operator+=(const Mat& m)
// {
//     (*this) = (*this) + m;
//     return *this;
// }
//
// Mat& Mat::operator-=(const Mat& m)
// {
//     (*this) = (*this) - m;
//     return *this;
// }
//
// Mat& Mat::operator/=(const Mat& m)
// {
//     (*this) = (*this) / m;
//     return *this;
// }
//
// Mat& Mat::operator*=(const Mat& m)
// {
//     (*this) = (*this) * m;
//     return *this;
// }

Mat Mat::reshape(const std::vector<int> newSizes) const
{
    return this->reshape(newSizes.size(), newSizes.data());
}

Mat Mat::reshape(int newDims, const int* newSizes) const
{
    size_t new_total = 1;

    for (int i = 0; i < newDims; i++)
    {
        new_total *= newSizes[i];
    }

    if (new_total != total())
    {
        M_Error(NULL, "The total size of the new Mat is not equal to the original Mat!");
        return Mat();
    }

    Mat m;
    m = *this;
    m.setSize(newDims, newSizes);
    return m;
}

// copy a full Mat, it will re-allocate memory.
void Mat::copyTo(Mat &dst) const
{
    int t = dst.type();

    // TODO 增加常用数据转换下的copyTo函数
    if (t != matType)
    {
        M_Error(Error::Code::StsBadType, "Unsupported data transform right now!");
        return;
    }

    if (empty())
    {
        dst.release();
        return;
    }

    dst.create(dims, size.p, type());
    if (data == dst.data) // the two mat are the same.
        return;

    // if the data ptr is not the same,
    if (total() != 0)
    {
        // TODO handle the sub mat.
        int esz = CV_ELEM_SIZE(type());
        size_t total_size = total() * esz;
        memcpy(dst.data, data, total_size);
    }
}

Mat Mat::clone() const
{
    Mat m;
    m.matType = this->type();
    copyTo(m);
    return m;
}

void Mat::create(int _dims, const int* _sizes, int _type)
{
    int i;
    M_Assert(0 < _dims && _dims <= MAT_MAX_DIM && _sizes);

    // same Mat
    if (data && (_dims == dims) && type() == _type)
    {
        for (i = 0; i < _dims; i++)
        {
            if (size.p[i] != _sizes[i]) break;
        }

        if (i == _dims)
            return; // same Mat, do not free and re-allocate
    }

    int size_back[MAT_MAX_DIM];
    if (_sizes == size.p)
    {
        for (int i = 0; i < _dims; i++)
        {
            size_back[i] = size.p[i];
        }
        _sizes = size_back;
    }

    release();

    if (_dims == 0)
        return;

    matType = _type;
    _setSize(*this, _dims, _sizes);

    // check if need to allocate memory
    if (total() > 0)
    {
        MatAllocator *a = allocator;

        if (!a)
            a = getDefaultAllocator();

        try
        {
            u = a->allocate(dims, size.p, type(), 0);
            assert(u != 0);
        }
        catch (...)
        {
            throw;
        }
    }

    addref();

    if (u)
        data = u->data;
}

void Mat::create(const std::vector<int> _sizes, int _type)
{
    create(_sizes.size(), _sizes.data(), _type);
}

void Mat::release()
{
    if (u && CV_XADD(&u->refcount, -1) == 1)
    {
        deallocate();
    }

    u = NULL;
    data = 0;

    for (int i = 0; i < dims; ++i)
    {
        size.p[i] = 0;
    }
}

void Mat::deallocate()
{
    if (u)
    {
        MatData* u_ = u;
        u = NULL;
        (u_->allocator ? u_->allocator : allocator ? allocator : getDefaultAllocator())->unmap(u_);
    }
}

int Mat::type() const
{
    return (int)matType;
}

void Mat::setSize(std::vector<int> size)
{
    _setSize(*this, size.size(), size.data());
}

void Mat::setSize(int dim, const int *size)
{
    M_Assert(dim > 0);
    _setSize(*this, dim, size);
}

void Mat::setSize(Mat &m)
{
    _setSize(*this, m.size.dims(), m.size.p);
}

void Mat::setTo(float v)
{
    int vi;
    switch (matType)
    {
        case CV_8U:
        {
            uint8_t u = saturate_cast<uint8_t>(v);
            uchar* p = (uchar *)&vi;
            p[0] = u; p[1] = u; p[2] = u; p[3] = u;
            memset(data, vi, total() * sizeof(uint8_t));
//#ifdef __APPLE__
//            memset_pattern4(data, &vi, total() * sizeof(uint8_t));
//#else
//            int* data_f = (int*)data;
//            std::fill(data_f, data_f + total(), vi);
//#endif
            break;
        }
        case CV_8S:
        {
            int8_t u = saturate_cast<int8_t>(v);
            char* p = (char *)&vi;
            p[0] = u; p[1] = u; p[2] = u; p[3] = u;
            memset(data, vi, total() * sizeof(int8_t));
//            memset_pattern4(data, &vi, total() * sizeof(uint8_t));
            break;
        }
        case CV_16U:
        {
            ushort u = saturate_cast<ushort>(v);
            ushort * p = (ushort *)&vi;
            p[0] = u; p[1] = u;
//            memset_pattern4(data, &vi, total() * sizeof(ushort));
            int* data_f = (int*)data;
            size_t t = + total()/(sizeof(int)/sizeof(ushort));
            std::fill(data_f, data_f + t, vi);
            break;
        }
        case CV_16S:
        {
            short u = saturate_cast<short>(v);
            short * p = (short *)&vi;
            p[0] = u; p[1] = u;
            int* data_f = (int*)data;
            size_t t = + total()/(sizeof(int)/sizeof(short));
            std::fill(data_f, data_f + t, vi);
//            memset_pattern4(data, &vi, total() * sizeof(ushort));
            break;
        }
        case CV_32F:
        {
            memcpy(&vi, &v, sizeof(float ));
#ifdef __APPLE__
            memset_pattern4(data, &vi, total() * sizeof(int ));
#else
            int* data_f = (int*)data;
            std::fill(data_f, data_f + total(), vi);
#endif
            break;
        }
        case CV_32S:
        {
            int* data_f = (int*)data;
            std::fill(data_f, data_f + total(), v);
//            memset_pattern4(data, &v, total() * sizeof(int ));
            break;
        }
        case CV_32U:
        {
            uint uv = saturate_cast<uint>(v);
            memcpy(&vi, &uv, sizeof(int ));

            int* data_f = (int*)data;
            std::fill(data_f, data_f + total(), vi);
//            memset_pattern4(data, &vi, total() * sizeof(int ));
            break;
        }
        case CV_16F:
        {
            // TODO check if the half-float is correct?
            ushort u = *(ushort *)hfloat(v).get_ptr();
            ushort * p = (ushort *)&vi;
            p[0] = u; p[1] = u;
            size_t t = + total()/(sizeof(int)/sizeof(ushort));
            int* data_f = (int*)data;
            std::fill(data_f, data_f + t, vi);
//            memset_pattern4(data, &vi, total() * sizeof(ushort));
            break;
        }
        default:
            M_Error_(Error::StsNotImplemented, ("Unsupported type:%d in setTo function!", matType));
    }
}

size_t Mat::total() const
{
    if (size.p == 0 || dims == 0)
        return 0;
    size_t p = 1;

    for (int i = dims-1; i >= 0; i--)
    {
        p *= size.p[i];
    }
    return p;
}

size_t Mat::total(int startDim, int endDim) const
{
    assert(startDim >= 0 && startDim <= endDim);
    size_t p = 1;

    int endDim_ = endDim <= dims ? endDim : dims;
    for (int i = startDim; i < endDim_; i++)
    {
        p *= size.p[i];
    }
    return p;
}

void Mat::printShape() const
{
    if (empty())
    {
        std::cout<<"shape = []"<<std::endl;
        return;
    }

    std::cout<<"shape = ["<<size.p[0]<<"x";
    for (int i = 1; i < dims; i++)
    {
        if (i == dims - 1)
            std::cout<<size.p[i];
        else
            std::cout<<size.p[i]<<"x";
    }
    std::cout<<"]"<<std::endl;
}

void Mat::print(int len) const
{
    if (empty())
    {
        std::cout<<"mat is empty"<<std::endl;
        return;
    }

    // print shape
    std::cout<<"shape = ["<<size.p[0]<<"x";
    for (int i = 1; i < dims; i++)
    {
        if (i == dims - 1)
            std::cout<<size.p[i];
        else
            std::cout<<size.p[i]<<"x";
    }
    std::cout<<"]"<<std::endl;

    // print value
    size_t printLen = len == -1 ? total() :
            len > total() ? total() : len;

    std::cout<<"value = ";
    if (type() == CV_32F)
    {
        const float* p = (const float*)data;
        for (int i = 0; i < printLen; i++)
        {
            std::cout<<p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (type() == CV_32S)
    {
        const int* p = (const int*)data;
        for (int i = 0; i < printLen; i++)
        {
            std::cout<<p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (type() == CV_32U)
    {
        const uint* p = (const uint*)data;
        for (int i = 0; i < printLen; i++)
        {
            std::cout<<p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (type() == CV_8S)
    {
        const char* p = (const char*)data;
        for (int i = 0; i < printLen; i++)
        {
            std::cout<<(int)p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (type() == CV_8U)
    {
        const uchar* p = (const uchar*)data;
        for (int i = 0; i < printLen; i++)
        {
            std::cout<<(int)p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (type() == CV_16U)
    {
        const ushort* p = (const ushort *)data;
        for (int i = 0; i < printLen; i++)
        {
            std::cout<<(int)p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (type() == CV_16S)
    {
        const short* p = (const short *)data;
        for (int i = 0; i < printLen; i++)
        {
            std::cout<<(int)p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else
        M_Error_(Error::Code::StsBadType, ("Unsupported format at function \" Mat::print \" type = %d!", type()));
}

MatShape Mat::shape() const
{
    if (this->empty())
        return {};

    MatShape shape(dims, 0);
    memcpy(shape.data(), this->size.p, dims * sizeof (int));

    return shape;
}

bool Mat::empty() const
{
    return data == 0 || total() == 0 || dims == 0;
}

uchar *Mat::ptr()
{
    return data;
}

template<typename _Tp>
_Tp &Mat::at(int i0)
{
    return (_Tp*)data[i0];
}

void Mat::addref()
{
    if (u)
        CV_XADD(&u->refcount, 1);
}

size_t total(const Mat& m)
{
    return m.total();
}

size_t total(const Mat& m, int startDim, int endDim)
{
    if (endDim == -1)
    {
        endDim = m.dims;
    }
    return m.total(startDim, endDim);
}

size_t total(const MatShape shape)
{
    return total(shape, 0, shape.size());
}

size_t total(const MatShape shape, int startDim, int endDim)
{
    if (endDim == -1)
    {
        endDim = shape.size();
    }

    assert(startDim >= 0 && startDim <= endDim);
    size_t p = 1;
    int dims = shape.size();

    int endDim_ = endDim <= dims ? endDim : dims;
    for (int i = startDim; i < endDim_; i++)
    {
        p *= shape[i];
    }
    return p;
}

}
