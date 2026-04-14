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
#include <climits>
#include <limits>
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

namespace {

inline bool is_supported_depth(int depth)
{
    return depth >= CV_8U && depth <= CV_16F;
}

inline bool is_supported_type(int type)
{
    const int depth = CV_MAT_DEPTH(type);
    const int channels = CV_MAT_CN(type);
    return is_supported_depth(depth) && channels > 0 && channels <= CV_CN_MAX;
}

inline void clear_steps(Mat& m)
{
    for (int i = 0; i < MAT_MAX_DIM; ++i)
    {
        m.stepBuf[i] = 0;
    }
}

inline void copy_steps(Mat& dst, const Mat& src)
{
    for (int i = 0; i < MAT_MAX_DIM; ++i)
    {
        dst.stepBuf[i] = src.stepBuf[i];
    }
}

inline void update_continuous_steps(Mat& m)
{
    clear_steps(m);

    if (m.dims <= 0 || !m.size.p)
        return;

    size_t stride = m.elemSize();
    for (int i = m.dims - 1; i >= 0; --i)
    {
        m.stepBuf[i] = stride;
        stride *= static_cast<size_t>(m.size.p[i]);
    }
}

inline std::string format_shape_string(const std::vector<int>& dims)
{
    if (dims.empty())
        return "[]";

    std::string out = "[" + std::to_string(dims[0]);
    for (size_t i = 1; i < dims.size(); ++i)
    {
        out += "x" + std::to_string(dims[i]);
    }
    out += "]";
    return out;
}

template<typename _Tp>
inline void fill_scalar_pattern(_Tp* out, size_t pixel_count, int cn, const Scalar& s)
{
    _Tp lane[4];
    for (int ch = 0; ch < cn; ++ch)
    {
        lane[ch] = saturate_cast<_Tp>(s[ch]);
    }

    for (size_t i = 0; i < pixel_count; ++i)
    {
        _Tp* px = out + i * static_cast<size_t>(cn);
        for (int ch = 0; ch < cn; ++ch)
        {
            px[ch] = lane[ch];
        }
    }
}

} // namespace

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

Range::Range() : start(0), end(0)
{
}

Range::Range(int _start, int _end) : start(_start), end(_end)
{
}

Range Range::all()
{
    return Range(0, INT_MAX);
}

// <<<<<<<<<<<<<<<<<<<<<   Mat   >>>>>>>>>>>>
// Setting the dim for mat.
void _setSize(Mat& m, int _dim, const int* _sz)
{
    CV_Assert(_dim <= MAT_MAX_DIM && _dim >= 0);

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
        CV_Assert(s > 0);
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
    const int depth_code = CV_MAT_DEPTH(_type);
    CV_Assert(depth_code == CV_32S || depth_code == CV_32U || depth_code == CV_32F);
    int type = _type;
    create(_dims, _sizes, type);

    // set all value to v
    size_t allSize = this->total() * static_cast<size_t>(channels());
    int* data_i = (int *)this->data;
    for (size_t i = 0; i < allSize; i++)
    {
        data_i[i] = v;
    }
}

Mat::Mat(const std::vector<int> _sizes, int _type, int v)
:dims(0), data(0), allocator(0), u(0), size(0), matType(_type)
{
    const int depth_code = CV_MAT_DEPTH(_type);
    CV_Assert(depth_code == CV_32S || depth_code == CV_32U || depth_code == CV_32F);
    int type = _type;
    create(_sizes, type);

    // set all value to v
    size_t allSize = this->total() * static_cast<size_t>(channels());
    int* data_i = (int *)this->data;
    for (size_t i = 0; i < allSize; i++)
    {
        data_i[i] = v;
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
    copy_steps(*this, m);
}

Mat::Mat(int _dims, const int* _sizes, int _type, void* _data)
:dims(0), data(0), allocator(0), u(0), size(nullptr), matType(_type)
{
    if (!is_supported_type(_type))
    {
        CV_Error_(Error::StsNotImplemented,
                 ("Unsupported Mat type=%d in external-memory constructor", _type));
    }
    CV_Assert(_data != nullptr);
    data = (uchar*)_data;
    _setSize(*this, _dims, _sizes);
    update_continuous_steps(*this);
}

Mat::Mat(const std::vector<int> _sizes, int _type, void* _data)
:dims(0), data(0), allocator(0), u(0), size(nullptr), matType(_type)
{
    if (!is_supported_type(_type))
    {
        CV_Error_(Error::StsNotImplemented,
                 ("Unsupported Mat type=%d in external-memory constructor", _type));
    }
    CV_Assert(_data != nullptr);
    data = (uchar*)_data;
    _setSize(*this, _sizes.size(), _sizes.data());
    update_continuous_steps(*this);
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
        copy_steps(*this, m);
    }

    return *this;
}

Mat& Mat::operator=(const float v)
{
    setTo(v);
    return *this;
}

Mat& Mat::operator=(const int v)
{
    setTo(static_cast<float>(v));
    return *this;
}

Mat& Mat::operator=(const Scalar& s)
{
    setTo(s);
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

Mat Mat::reinterpret(int rtype) const
{
    if (empty())
    {
        CV_Error(Error::Code::StsBadArg, "reinterpret expects a non-empty Mat");
    }

    if (rtype < 0)
    {
        CV_Error_(Error::StsBadArg, ("reinterpret expects non-negative type, got=%d", rtype));
    }

    const int dtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), CV_MAT_CN(rtype));
    if (!is_supported_type(dtype))
    {
        CV_Error_(Error::StsNotImplemented, ("reinterpret target type=%d is unsupported in v1", dtype));
    }

    if (CV_ELEM_SIZE(dtype) != elemSize())
    {
        CV_Error_(Error::StsBadArg,
                  ("reinterpret requires identical element byte-size, src=%zu dst=%d",
                   elemSize(),
                   CV_ELEM_SIZE(dtype)));
    }

    Mat out = *this;
    out.matType = dtype;
    return out;
}

Mat Mat::rowRange(int startrow, int endrow) const
{
    if (dims < 1)
    {
        CV_Error(Error::Code::StsBadArg, "rowRange expects dims >= 1");
    }
    if (startrow < 0 || endrow < startrow || endrow > size.p[0])
    {
        CV_Error_(Error::Code::StsBadArg,
                 ("rowRange out of range, start=%d end=%d rows=%d", startrow, endrow, size.p[0]));
    }

    if (startrow == 0 && endrow == size.p[0])
        return *this;

    Mat out = *this;
    MatShape new_shape = shape();
    new_shape[0] = endrow - startrow;
    out.setSize(new_shape);
    copy_steps(out, *this);
    out.data = data + static_cast<size_t>(startrow) * step(0);
    return out;
}

Mat Mat::colRange(int startcol, int endcol) const
{
    if (dims < 2)
    {
        CV_Error(Error::Code::StsBadArg, "colRange expects dims >= 2");
    }
    if (startcol < 0 || endcol < startcol || endcol > size.p[1])
    {
        CV_Error_(Error::Code::StsBadArg,
                 ("colRange out of range, start=%d end=%d cols=%d", startcol, endcol, size.p[1]));
    }

    if (startcol == 0 && endcol == size.p[1])
        return *this;

    Mat out = *this;
    MatShape new_shape = shape();
    new_shape[1] = endcol - startcol;
    out.setSize(new_shape);
    copy_steps(out, *this);
    out.data = data + static_cast<size_t>(startcol) * step(1);
    return out;
}

Mat Mat::operator()(const Range& row_range, const Range& col_range) const
{
    CV_Assert(dims >= 2);

    const int row_start = row_range.start;
    const int row_end = row_range.end == INT_MAX ? size.p[0] : row_range.end;
    const int col_start = col_range.start;
    const int col_end = col_range.end == INT_MAX ? size.p[1] : col_range.end;

    return rowRange(row_start, row_end).colRange(col_start, col_end);
}

Mat Mat::reshape(int newDims, const int* newSizes) const
{
    CV_Assert(newDims > 0 && newDims <= MAT_MAX_DIM);
    CV_Assert(newSizes != nullptr);
    if (!isContinuous())
    {
        CV_Error(Error::Code::StsNotImplemented, "reshape on non-contiguous Mat is not supported in v1");
    }

    size_t new_total = 1;

    for (int i = 0; i < newDims; i++)
    {
        if (newSizes[i] <= 0)
        {
            CV_Error_(Error::Code::StsBadSize,
                     ("Invalid reshape size at dim=%d, value=%d", i, newSizes[i]));
        }
        if (new_total > std::numeric_limits<size_t>::max() / static_cast<size_t>(newSizes[i]))
        {
            CV_Error(Error::Code::StsOutOfMem, "reshape size overflow");
        }
        new_total *= newSizes[i];
    }

    if (new_total != total())
    {
        CV_Error(Error::Code::StsBadSize, "The total size of the new Mat is not equal to the original Mat!");
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
    if (empty())
    {
        dst.release();
        return;
    }

    if (!dst.empty() && dst.type() != matType)
    {
        CV_Error_(Error::Code::StsBadType,
                 ("Mat::copyTo type mismatch, src=%d dst=%d", matType, dst.type()));
    }

    dst.create(dims, size.p, type());
    if (data == dst.data) // the two mat are the same.
        return;

    if (total() == 0)
        return;

    if (isContinuous() && dst.isContinuous())
    {
        const size_t total_size = total() * elemSize();
        memcpy(dst.data, data, total_size);
        return;
    }

    const int outer = dims > 1 ? size.p[0] : 1;
    const size_t inner_bytes =
            (dims > 1 ? total(1, dims) : total()) * elemSize();
    const size_t src_step0 = dims > 1 ? step(0) : inner_bytes;
    const size_t dst_step0 = dims > 1 ? dst.step(0) : inner_bytes;

    for (int i = 0; i < outer; ++i)
    {
        const uchar* src_row = data + static_cast<size_t>(i) * src_step0;
        uchar* dst_row = dst.data + static_cast<size_t>(i) * dst_step0;
        memcpy(dst_row, src_row, inner_bytes);
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
    CV_Assert(0 < _dims && _dims <= MAT_MAX_DIM && _sizes);
    if (!is_supported_type(_type))
    {
        CV_Error_(Error::StsNotImplemented, ("Unsupported Mat type=%d in Mat::create", _type));
    }

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
    update_continuous_steps(*this);

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
        stepBuf[i] = 0;
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

int Mat::depth() const
{
    return CV_MAT_DEPTH(matType);
}

int Mat::channels() const
{
    return CV_MAT_CN(matType);
}

size_t Mat::elemSize1() const
{
    int d = depth();
    if (!is_supported_depth(d))
    {
        CV_Error_(Error::StsNotImplemented,
                 ("Mat::elemSize1 supports scalar depth in [CV_8U..CV_16F], depth=%d", d));
    }

    return static_cast<size_t>(CV_ELEM_SIZE1(d));
}

size_t Mat::elemSize() const
{
    return elemSize1() * static_cast<size_t>(channels());
}

size_t Mat::step(int dim) const
{
    if (empty())
        return 0;

    CV_Assert(dim >= 0 && dim < dims);
    return stepBuf[dim];
}

size_t Mat::step1(int dim) const
{
    if (empty())
        return 0;

    CV_Assert(dim >= 0 && dim < dims);
    return stepBuf[dim] / elemSize1();
}

bool Mat::isContinuous() const
{
    if (empty())
        return false;

    size_t expected = elemSize();
    for (int i = dims - 1; i >= 0; --i)
    {
        if (stepBuf[i] != expected)
            return false;
        expected *= static_cast<size_t>(size.p[i]);
    }
    return true;
}

void Mat::setSize(std::vector<int> size)
{
    _setSize(*this, size.size(), size.data());
    update_continuous_steps(*this);
}

void Mat::setSize(int dim, const int *size)
{
    CV_Assert(dim > 0);
    _setSize(*this, dim, size);
    update_continuous_steps(*this);
}

void Mat::setSize(Mat &m)
{
    _setSize(*this, m.size.dims(), m.size.p);
    update_continuous_steps(*this);
}

void Mat::setTo(float v)
{
    if (empty())
        return;

    const int d = depth();
    const size_t outer = dims > 1 ? static_cast<size_t>(size.p[0]) : 1;
    const size_t scalar_per_outer =
            (dims > 1 ? total(1, dims) : total()) * static_cast<size_t>(channels());
    const size_t outer_step = dims > 1 ? step(0) : scalar_per_outer * elemSize1();

    auto fill_block = [d, v](uchar* ptr, size_t scalar_count) {
        switch (d)
        {
            case CV_8U:
            {
                const uchar u = saturate_cast<uchar>(v);
                uchar* out = reinterpret_cast<uchar*>(ptr);
                std::fill(out, out + scalar_count, u);
                break;
            }
            case CV_8S:
            {
                const schar s = saturate_cast<schar>(v);
                schar* out = reinterpret_cast<schar*>(ptr);
                std::fill(out, out + scalar_count, s);
                break;
            }
            case CV_16U:
            {
                const ushort u = saturate_cast<ushort>(v);
                ushort* out = reinterpret_cast<ushort*>(ptr);
                std::fill(out, out + scalar_count, u);
                break;
            }
            case CV_16S:
            {
                const short s = saturate_cast<short>(v);
                short* out = reinterpret_cast<short*>(ptr);
                std::fill(out, out + scalar_count, s);
                break;
            }
            case CV_32F:
            {
                float* out = reinterpret_cast<float*>(ptr);
                std::fill(out, out + scalar_count, v);
                break;
            }
            case CV_32S:
            {
                const int iv = saturate_cast<int>(v);
                int* out = reinterpret_cast<int*>(ptr);
                std::fill(out, out + scalar_count, iv);
                break;
            }
            case CV_32U:
            {
                const uint uv = saturate_cast<uint>(v);
                uint* out = reinterpret_cast<uint*>(ptr);
                std::fill(out, out + scalar_count, uv);
                break;
            }
            case CV_16F:
            {
                const hfloat hf(v);
                hfloat* out = reinterpret_cast<hfloat*>(ptr);
                std::fill(out, out + scalar_count, hf);
                break;
            }
            default:
                CV_Error_(Error::StsNotImplemented, ("Unsupported type:%d in setTo function!", d));
        }
    };

    for (size_t i = 0; i < outer; ++i)
    {
        uchar* row_ptr = data + i * outer_step;
        fill_block(row_ptr, scalar_per_outer);
    }
}

void Mat::setTo(const Scalar& s)
{
    if (empty())
        return;

    const int d = depth();
    const int cn = channels();
    if (cn > 4)
    {
        CV_Error_(Error::StsBadArg,
                 ("Mat::setTo(Scalar) supports channels <= 4 in v1, channels=%d", cn));
    }

    const size_t outer = dims > 1 ? static_cast<size_t>(size.p[0]) : 1;
    const size_t pixel_per_outer = dims > 1 ? total(1, dims) : total();
    const size_t outer_step = dims > 1 ? step(0) : pixel_per_outer * elemSize();

    for (size_t i = 0; i < outer; ++i)
    {
        uchar* row_ptr = data + i * outer_step;
        switch (d)
        {
            case CV_8U:
                fill_scalar_pattern(reinterpret_cast<uchar*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_8S:
                fill_scalar_pattern(reinterpret_cast<schar*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_16U:
                fill_scalar_pattern(reinterpret_cast<ushort*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_16S:
                fill_scalar_pattern(reinterpret_cast<short*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_32F:
                fill_scalar_pattern(reinterpret_cast<float*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_32S:
                fill_scalar_pattern(reinterpret_cast<int*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_32U:
                fill_scalar_pattern(reinterpret_cast<uint*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_16F:
                fill_scalar_pattern(reinterpret_cast<hfloat*>(row_ptr), pixel_per_outer, cn, s);
                break;
            default:
                CV_Error_(Error::StsNotImplemented, ("Unsupported type:%d in setTo(Scalar)", d));
        }
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
    CV_Assert(startDim >= 0 && startDim <= endDim);
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
    std::cout << "shape = " << displayShapeString() << std::endl;
}

void Mat::print(int len) const
{
    if (empty())
    {
        std::cout<<"mat is empty"<<std::endl;
        return;
    }

    // print shape
    std::cout << "shape = " << displayShapeString() << std::endl;

    // print value
    const size_t scalar_total = total() * static_cast<size_t>(channels());
    size_t printLen = len == -1 ? scalar_total :
            len > static_cast<int>(scalar_total) ? scalar_total : static_cast<size_t>(len);

    std::cout<<"value = ";
    if (depth() == CV_32F)
    {
        const float* p = (const float*)data;
        for (size_t i = 0; i < printLen; i++)
        {
            std::cout<<p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (depth() == CV_32S)
    {
        const int* p = (const int*)data;
        for (size_t i = 0; i < printLen; i++)
        {
            std::cout<<p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (depth() == CV_32U)
    {
        const uint* p = (const uint*)data;
        for (size_t i = 0; i < printLen; i++)
        {
            std::cout<<p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (depth() == CV_8S)
    {
        const char* p = (const char*)data;
        for (size_t i = 0; i < printLen; i++)
        {
            std::cout<<(int)p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (depth() == CV_8U)
    {
        const uchar* p = (const uchar*)data;
        for (size_t i = 0; i < printLen; i++)
        {
            std::cout<<(int)p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (depth() == CV_16U)
    {
        const ushort* p = (const ushort *)data;
        for (size_t i = 0; i < printLen; i++)
        {
            std::cout<<(int)p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else if (depth() == CV_16S)
    {
        const short* p = (const short *)data;
        for (size_t i = 0; i < printLen; i++)
        {
            std::cout<<(int)p[i]<<",";
        }
        std::cout<<std::endl;
    }
    else
        CV_Error_(Error::Code::StsBadType, ("Unsupported format at function \" Mat::print \" type = %d!", type()));
}

MatShape Mat::shape() const
{
    if (this->empty())
        return {};

    MatShape shape(dims, 0);
    memcpy(shape.data(), this->size.p, dims * sizeof (int));

    return shape;
}

std::string Mat::shapeString(ShapeDisplayOrder order) const
{
    if (empty())
        return "[]";

    if (order == ShapeDisplayOrder::ChannelFirst && channels() > 1 && dims == 2)
    {
        return format_shape_string({channels(), size.p[0], size.p[1]});
    }

    return format_shape_string(shape());
}

std::string Mat::displayShapeString() const
{
    if (!empty() && channels() > 1 && dims == 2)
    {
        return shapeString(ShapeDisplayOrder::ChannelFirst);
    }
    return shapeString(ShapeDisplayOrder::Geometry);
}

bool Mat::empty() const
{
    return data == 0 || total() == 0 || dims == 0;
}

uchar *Mat::ptr()
{
    return data;
}

const uchar* Mat::ptr() const
{
    return data;
}

const uchar* Mat::pixelPtr(int y, int x) const
{
    if (empty())
    {
        CV_Error(Error::StsBadArg, "pixelPtr expects a non-empty Mat");
    }
    if (dims != 2)
    {
        CV_Error_(Error::StsBadArg, ("pixelPtr expects a 2D Mat, dims=%d", dims));
    }
    if (y < 0 || y >= size.p[0] || x < 0 || x >= size.p[1])
    {
        CV_Error_(Error::StsOutOfRange,
                  ("pixelPtr index out of range, y=%d x=%d shape=[%d,%d]",
                   y, x, size.p[0], size.p[1]));
    }

    return data + static_cast<size_t>(y) * step(0) + static_cast<size_t>(x) * elemSize();
}

uchar* Mat::pixelPtr(int y, int x)
{
    return const_cast<uchar*>(static_cast<const Mat*>(this)->pixelPtr(y, x));
}

void Mat::addref()
{
    if (u)
        CV_XADD(&u->refcount, 1);
}

}
