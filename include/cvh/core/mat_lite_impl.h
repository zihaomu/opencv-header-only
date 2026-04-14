#ifndef CVH_CORE_MAT_LITE_IMPL_H
#define CVH_CORE_MAT_LITE_IMPL_H

#include "saturate.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace cvh
{

namespace lite_mat_detail
{

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
    {
        return;
    }

    size_t stride = m.elemSize();
    for (int i = m.dims - 1; i >= 0; --i)
    {
        m.stepBuf[i] = stride;
        stride *= static_cast<size_t>(m.size.p[i]);
    }
}

template<typename _Tp>
inline _Tp* alignPtr(_Tp* ptr, int n = static_cast<int>(sizeof(_Tp)))
{
    CV_Assert((n & (n - 1)) == 0);
    return reinterpret_cast<_Tp*>((reinterpret_cast<size_t>(ptr) + n - 1) & -static_cast<size_t>(n));
}

inline std::string format_shape_string(const std::vector<int>& dims)
{
    if (dims.empty())
    {
        return "[]";
    }

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

template<typename SrcT, typename DstT>
inline void convert_array(const uchar* src_, uchar* dst_, size_t count)
{
    const SrcT* src = reinterpret_cast<const SrcT*>(src_);
    DstT* dst = reinterpret_cast<DstT*>(dst_);
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = saturate_cast<DstT>(src[i]);
    }
}

template<typename SrcT>
inline bool convert_dispatch_dst(int ddepth, const uchar* src, uchar* dst, size_t count)
{
    switch (ddepth)
    {
        case CV_8U:  convert_array<SrcT, uchar>(src, dst, count); return true;
        case CV_8S:  convert_array<SrcT, schar>(src, dst, count); return true;
        case CV_16U: convert_array<SrcT, ushort>(src, dst, count); return true;
        case CV_16S: convert_array<SrcT, short>(src, dst, count); return true;
        case CV_32F: convert_array<SrcT, float>(src, dst, count); return true;
        case CV_32S: convert_array<SrcT, int>(src, dst, count); return true;
        case CV_32U: convert_array<SrcT, uint>(src, dst, count); return true;
        case CV_16F: convert_array<SrcT, hfloat>(src, dst, count); return true;
        default:
            return false;
    }
}

inline bool convert_dispatch(int sdepth, int ddepth, const uchar* src, uchar* dst, size_t count)
{
    switch (sdepth)
    {
        case CV_8U:  return convert_dispatch_dst<uchar>(ddepth, src, dst, count);
        case CV_8S:  return convert_dispatch_dst<schar>(ddepth, src, dst, count);
        case CV_16U: return convert_dispatch_dst<ushort>(ddepth, src, dst, count);
        case CV_16S: return convert_dispatch_dst<short>(ddepth, src, dst, count);
        case CV_32F: return convert_dispatch_dst<float>(ddepth, src, dst, count);
        case CV_32S: return convert_dispatch_dst<int>(ddepth, src, dst, count);
        case CV_32U: return convert_dispatch_dst<uint>(ddepth, src, dst, count);
        case CV_16F: return convert_dispatch_dst<hfloat>(ddepth, src, dst, count);
        default:
            return false;
    }
}

} // namespace lite_mat_detail

#define MAT_MALLOC_ALIGN 64

inline void* fastMalloc(size_t size)
{
    uchar* udata = static_cast<uchar*>(std::malloc(size + sizeof(void*) + MAT_MALLOC_ALIGN));
    if (!udata)
    {
        std::cerr << "Out of memory in fastMalloc" << std::endl;
        return nullptr;
    }

    uchar** adata = lite_mat_detail::alignPtr(reinterpret_cast<uchar**>(udata) + 1, MAT_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

inline void fastFree(void* ptr)
{
    if (ptr)
    {
        uchar* udata = (reinterpret_cast<uchar**>(ptr))[-1];
        CV_Assert(udata < reinterpret_cast<uchar*>(ptr));
        std::free(udata);
    }
}

inline MatData::MatData(const MatAllocator* _allocator)
    : allocator(_allocator), refcount(0), data(nullptr), size(0), flags(static_cast<MatData::MemoryFlag>(0))
{
}

inline MatData::~MatData()
{
    allocator = nullptr;
    refcount = 0;
    data = nullptr;
    flags = static_cast<MatData::MemoryFlag>(0);
}

class StdMatAllocator : public MatAllocator
{
public:
    MatData* allocate(int dims, const int* sizes, int type, void* data0) const override
    {
        size_t total_size = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; --i)
        {
            total_size *= static_cast<size_t>(sizes[i]);
        }

        uchar* data = data0 ? static_cast<uchar*>(data0) : static_cast<uchar*>(fastMalloc(total_size));
        MatData* u = new MatData(this);
        u->data = data;
        u->size = total_size;
        if (data0)
        {
            u->flags = MatData::USER_MEMORY;
        }

        return u;
    }

    bool allocate(MatData* u, MatDataUsageFlags) const override
    {
        return u != nullptr;
    }

    void deallocate(MatData* u) const override
    {
        if (!u)
        {
            return;
        }

        CV_Assert(u->refcount == 0);
        if (u->flags != MatData::USER_MEMORY)
        {
            fastFree(u->data);
            u->data = nullptr;
        }

        delete u;
    }
};

inline void MatAllocator::map(MatData*) const
{
}

inline void MatAllocator::unmap(MatData* u) const
{
    if (u && u->refcount == 0)
    {
        deallocate(u);
    }
}

inline Range::Range()
    : start(0), end(0)
{
}

inline Range::Range(int _start, int _end)
    : start(_start), end(_end)
{
}

inline Range Range::all()
{
    return Range(0, INT_MAX);
}

inline void _setSize(Mat& m, int _dim, const int* _sz)
{
    CV_Assert(_dim <= MAT_MAX_DIM && _dim >= 0);

    if (_dim != m.dims)
    {
        fastFree(m.size.p0);
        m.size.p0 = static_cast<int*>(fastMalloc(static_cast<size_t>(_dim + 1) * sizeof(int)));
        m.size.p = m.size.p0 + 1;
        m.size.p0[0] = _dim;
    }

    m.dims = _dim;

    if (!_sz)
    {
        return;
    }

    for (int i = _dim - 1; i >= 0; --i)
    {
        const int s = _sz[i];
        CV_Assert(s > 0);
        m.size.p[i] = s;
    }
}

inline MatAllocator*& getDefaultAllocatorMatRef()
{
    static MatAllocator* g_matAllocator = Mat::getStdAllocator();
    return g_matAllocator;
}

inline MatAllocator* Mat::getDefaultAllocator()
{
    return getDefaultAllocatorMatRef();
}

inline MatAllocator* Mat::getStdAllocator()
{
    static MatAllocator* const allocator = new StdMatAllocator();
    return allocator;
}

inline void Mat::copySize(const Mat& m)
{
    _setSize(*this, m.dims, nullptr);

    for (int i = 0; i < dims; ++i)
    {
        size[i] = m.size[i];
    }
}

inline Mat::Mat()
    : dims(0), data(nullptr), allocator(nullptr), u(nullptr), size(nullptr), matType(CV_32F)
{
}

inline Mat::Mat(int _dims, const int* _sizes, int _type)
    : dims(0), data(nullptr), allocator(nullptr), u(nullptr), size(nullptr), matType(CV_32F)
{
    create(_dims, _sizes, _type);
}

inline Mat::Mat(const std::vector<int> _sizes, int _type)
    : dims(0), data(nullptr), allocator(nullptr), u(nullptr), size(nullptr), matType(CV_32F)
{
    create(_sizes, _type);
}

inline Mat::Mat(int _dims, const int* _sizes, int _type, int v)
    : dims(0), data(nullptr), allocator(nullptr), u(nullptr), size(nullptr), matType(_type)
{
    const int depth_code = CV_MAT_DEPTH(_type);
    CV_Assert(depth_code == CV_32S || depth_code == CV_32U || depth_code == CV_32F);
    create(_dims, _sizes, _type);

    const size_t all_size = total() * static_cast<size_t>(channels());
    int* data_i = reinterpret_cast<int*>(data);
    for (size_t i = 0; i < all_size; ++i)
    {
        data_i[i] = v;
    }
}

inline Mat::Mat(const std::vector<int> _sizes, int _type, int v)
    : dims(0), data(nullptr), allocator(nullptr), u(nullptr), size(nullptr), matType(_type)
{
    const int depth_code = CV_MAT_DEPTH(_type);
    CV_Assert(depth_code == CV_32S || depth_code == CV_32U || depth_code == CV_32F);
    create(_sizes, _type);

    const size_t all_size = total() * static_cast<size_t>(channels());
    int* data_i = reinterpret_cast<int*>(data);
    for (size_t i = 0; i < all_size; ++i)
    {
        data_i[i] = v;
    }
}

inline Mat::Mat(const Mat& m)
    : dims(m.dims), data(m.data), allocator(m.allocator), u(m.u), size(nullptr), matType(m.matType)
{
    if (u)
    {
        CV_XADD(&u->refcount, 1);
    }

    dims = 0;
    copySize(m);
    lite_mat_detail::copy_steps(*this, m);
}

inline Mat::Mat(int _dims, const int* _sizes, int _type, void* _data)
    : dims(0), data(nullptr), allocator(nullptr), u(nullptr), size(nullptr), matType(_type)
{
    if (!lite_mat_detail::is_supported_type(_type))
    {
        CV_Error_(Error::StsNotImplemented, ("Unsupported Mat type=%d in external-memory constructor", _type));
    }
    CV_Assert(_data != nullptr);
    data = static_cast<uchar*>(_data);
    _setSize(*this, _dims, _sizes);
    lite_mat_detail::update_continuous_steps(*this);
}

inline Mat::Mat(const std::vector<int> _sizes, int _type, void* _data)
    : dims(0), data(nullptr), allocator(nullptr), u(nullptr), size(nullptr), matType(_type)
{
    if (!lite_mat_detail::is_supported_type(_type))
    {
        CV_Error_(Error::StsNotImplemented, ("Unsupported Mat type=%d in external-memory constructor", _type));
    }
    CV_Assert(_data != nullptr);
    data = static_cast<uchar*>(_data);
    _setSize(*this, static_cast<int>(_sizes.size()), _sizes.data());
    lite_mat_detail::update_continuous_steps(*this);
}

inline Mat::~Mat()
{
    release();
    fastFree(size.p0);
}

inline Mat& Mat::operator=(const Mat& m)
{
    if (this != &m)
    {
        if (m.u)
        {
            CV_XADD(&m.u->refcount, 1);
        }

        release();
        copySize(m);

        data = m.data;
        allocator = m.allocator;
        u = m.u;
        matType = m.matType;
        lite_mat_detail::copy_steps(*this, m);
    }

    return *this;
}

inline Mat& Mat::operator=(const float v)
{
    setTo(v);
    return *this;
}

inline Mat& Mat::operator=(const int v)
{
    setTo(static_cast<float>(v));
    return *this;
}

inline Mat& Mat::operator=(const Scalar& s)
{
    setTo(s);
    return *this;
}

inline Mat Mat::reshape(const std::vector<int> newSizes) const
{
    return reshape(static_cast<int>(newSizes.size()), newSizes.data());
}

inline Mat Mat::reinterpret(int rtype) const
{
    if (empty())
    {
        CV_Error(Error::StsBadArg, "reinterpret expects a non-empty Mat");
    }

    if (rtype < 0)
    {
        CV_Error_(Error::StsBadArg, ("reinterpret expects non-negative type, got=%d", rtype));
    }

    const int dtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), CV_MAT_CN(rtype));
    if (!lite_mat_detail::is_supported_type(dtype))
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

inline Mat Mat::rowRange(int startrow, int endrow) const
{
    if (dims < 1)
    {
        CV_Error(Error::StsBadArg, "rowRange expects dims >= 1");
    }
    if (startrow < 0 || endrow < startrow || endrow > size.p[0])
    {
        CV_Error_(Error::StsBadArg,
                  ("rowRange out of range, start=%d end=%d rows=%d", startrow, endrow, size.p[0]));
    }

    if (startrow == 0 && endrow == size.p[0])
    {
        return *this;
    }

    Mat out = *this;
    MatShape new_shape = shape();
    new_shape[0] = endrow - startrow;
    out.setSize(new_shape);
    lite_mat_detail::copy_steps(out, *this);
    out.data = data + static_cast<size_t>(startrow) * step(0);
    return out;
}

inline Mat Mat::colRange(int startcol, int endcol) const
{
    if (dims < 2)
    {
        CV_Error(Error::StsBadArg, "colRange expects dims >= 2");
    }
    if (startcol < 0 || endcol < startcol || endcol > size.p[1])
    {
        CV_Error_(Error::StsBadArg,
                  ("colRange out of range, start=%d end=%d cols=%d", startcol, endcol, size.p[1]));
    }

    if (startcol == 0 && endcol == size.p[1])
    {
        return *this;
    }

    Mat out = *this;
    MatShape new_shape = shape();
    new_shape[1] = endcol - startcol;
    out.setSize(new_shape);
    lite_mat_detail::copy_steps(out, *this);
    out.data = data + static_cast<size_t>(startcol) * step(1);
    return out;
}

inline Mat Mat::operator()(const Range& row_range, const Range& col_range) const
{
    CV_Assert(dims >= 2);

    const int row_start = row_range.start;
    const int row_end = row_range.end == INT_MAX ? size.p[0] : row_range.end;
    const int col_start = col_range.start;
    const int col_end = col_range.end == INT_MAX ? size.p[1] : col_range.end;

    return rowRange(row_start, row_end).colRange(col_start, col_end);
}

inline Mat Mat::reshape(int newDims, const int* newSizes) const
{
    CV_Assert(newDims > 0 && newDims <= MAT_MAX_DIM);
    CV_Assert(newSizes != nullptr);
    if (!isContinuous())
    {
        CV_Error(Error::StsNotImplemented, "reshape on non-contiguous Mat is not supported in v1");
    }

    size_t new_total = 1;
    for (int i = 0; i < newDims; ++i)
    {
        if (newSizes[i] <= 0)
        {
            CV_Error_(Error::StsBadSize, ("Invalid reshape size at dim=%d, value=%d", i, newSizes[i]));
        }
        if (new_total > std::numeric_limits<size_t>::max() / static_cast<size_t>(newSizes[i]))
        {
            CV_Error(Error::StsOutOfMem, "reshape size overflow");
        }
        new_total *= static_cast<size_t>(newSizes[i]);
    }

    if (new_total != total())
    {
        CV_Error(Error::StsBadSize, "The total size of the new Mat is not equal to the original Mat!");
    }

    Mat m;
    m = *this;
    m.setSize(newDims, newSizes);
    return m;
}

inline void Mat::copyTo(Mat& dst) const
{
    if (empty())
    {
        dst.release();
        return;
    }

    if (!dst.empty() && dst.type() != matType)
    {
        CV_Error_(Error::StsBadType, ("Mat::copyTo type mismatch, src=%d dst=%d", matType, dst.type()));
    }

    dst.create(dims, size.p, type());
    if (data == dst.data)
    {
        return;
    }

    if (total() == 0)
    {
        return;
    }

    if (isContinuous() && dst.isContinuous())
    {
        const size_t total_size = total() * elemSize();
        std::memcpy(dst.data, data, total_size);
        return;
    }

    const int outer = dims > 1 ? size.p[0] : 1;
    const size_t inner_bytes = (dims > 1 ? total(1, dims) : total()) * elemSize();
    const size_t src_step0 = dims > 1 ? step(0) : inner_bytes;
    const size_t dst_step0 = dims > 1 ? dst.step(0) : inner_bytes;

    for (int i = 0; i < outer; ++i)
    {
        const uchar* src_row = data + static_cast<size_t>(i) * src_step0;
        uchar* dst_row = dst.data + static_cast<size_t>(i) * dst_step0;
        std::memcpy(dst_row, src_row, inner_bytes);
    }
}

inline Mat Mat::clone() const
{
    Mat m;
    m.matType = type();
    copyTo(m);
    return m;
}

inline void Mat::create(int _dims, const int* _sizes, int _type)
{
    CV_Assert(0 < _dims && _dims <= MAT_MAX_DIM && _sizes);
    if (!lite_mat_detail::is_supported_type(_type))
    {
        CV_Error_(Error::StsNotImplemented, ("Unsupported Mat type=%d in Mat::create", _type));
    }

    if (data && _dims == dims && type() == _type)
    {
        int i = 0;
        for (; i < _dims; ++i)
        {
            if (size.p[i] != _sizes[i])
            {
                break;
            }
        }
        if (i == _dims)
        {
            return;
        }
    }

    int size_back[MAT_MAX_DIM];
    if (_sizes == size.p)
    {
        for (int i = 0; i < _dims; ++i)
        {
            size_back[i] = size.p[i];
        }
        _sizes = size_back;
    }

    release();

    matType = _type;
    _setSize(*this, _dims, _sizes);
    lite_mat_detail::update_continuous_steps(*this);

    if (total() > 0)
    {
        MatAllocator* a = allocator;
        if (!a)
        {
            a = getDefaultAllocator();
        }

        u = a->allocate(dims, size.p, type(), nullptr);
        CV_Assert(u != nullptr);
    }

    addref();

    if (u)
    {
        data = u->data;
    }
}

inline void Mat::create(const std::vector<int> _sizes, int _type)
{
    create(static_cast<int>(_sizes.size()), _sizes.data(), _type);
}

inline void Mat::release()
{
    if (u && CV_XADD(&u->refcount, -1) == 1)
    {
        deallocate();
    }

    u = nullptr;
    data = nullptr;

    for (int i = 0; i < dims; ++i)
    {
        size.p[i] = 0;
        stepBuf[i] = 0;
    }
}

inline void Mat::deallocate()
{
    if (u)
    {
        MatData* u_ = u;
        u = nullptr;
        (u_->allocator ? u_->allocator : allocator ? allocator : getDefaultAllocator())->unmap(u_);
    }
}

inline int Mat::type() const
{
    return matType;
}

inline int Mat::depth() const
{
    return CV_MAT_DEPTH(matType);
}

inline int Mat::channels() const
{
    return CV_MAT_CN(matType);
}

inline size_t Mat::elemSize1() const
{
    const int d = depth();
    if (!lite_mat_detail::is_supported_depth(d))
    {
        CV_Error_(Error::StsNotImplemented,
                  ("Mat::elemSize1 supports scalar depth in [CV_8U..CV_16F], depth=%d", d));
    }

    return static_cast<size_t>(CV_ELEM_SIZE1(d));
}

inline size_t Mat::elemSize() const
{
    return elemSize1() * static_cast<size_t>(channels());
}

inline size_t Mat::step(int dim) const
{
    if (empty())
    {
        return 0;
    }

    CV_Assert(dim >= 0 && dim < dims);
    return stepBuf[dim];
}

inline size_t Mat::step1(int dim) const
{
    if (empty())
    {
        return 0;
    }

    CV_Assert(dim >= 0 && dim < dims);
    return stepBuf[dim] / elemSize1();
}

inline bool Mat::isContinuous() const
{
    if (empty())
    {
        return false;
    }

    size_t expected = elemSize();
    for (int i = dims - 1; i >= 0; --i)
    {
        if (stepBuf[i] != expected)
        {
            return false;
        }
        expected *= static_cast<size_t>(size.p[i]);
    }
    return true;
}

inline void Mat::setSize(std::vector<int> _size)
{
    _setSize(*this, static_cast<int>(_size.size()), _size.data());
    lite_mat_detail::update_continuous_steps(*this);
}

inline void Mat::setSize(int dim, const int* _size)
{
    CV_Assert(dim > 0);
    _setSize(*this, dim, _size);
    lite_mat_detail::update_continuous_steps(*this);
}

inline void Mat::setSize(Mat& m)
{
    _setSize(*this, m.size.dims(), m.size.p);
    lite_mat_detail::update_continuous_steps(*this);
}

inline void Mat::setTo(float v)
{
    if (empty())
    {
        return;
    }

    const int d = depth();
    const size_t outer = dims > 1 ? static_cast<size_t>(size.p[0]) : 1;
    const size_t scalar_per_outer = (dims > 1 ? total(1, dims) : total()) * static_cast<size_t>(channels());
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

inline void Mat::setTo(const Scalar& s)
{
    if (empty())
    {
        return;
    }

    const int d = depth();
    const int cn = channels();
    if (cn > 4)
    {
        CV_Error_(Error::StsBadArg, ("Mat::setTo(Scalar) supports channels <= 4 in v1, channels=%d", cn));
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
                lite_mat_detail::fill_scalar_pattern(reinterpret_cast<uchar*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_8S:
                lite_mat_detail::fill_scalar_pattern(reinterpret_cast<schar*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_16U:
                lite_mat_detail::fill_scalar_pattern(reinterpret_cast<ushort*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_16S:
                lite_mat_detail::fill_scalar_pattern(reinterpret_cast<short*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_32F:
                lite_mat_detail::fill_scalar_pattern(reinterpret_cast<float*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_32S:
                lite_mat_detail::fill_scalar_pattern(reinterpret_cast<int*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_32U:
                lite_mat_detail::fill_scalar_pattern(reinterpret_cast<uint*>(row_ptr), pixel_per_outer, cn, s);
                break;
            case CV_16F:
                lite_mat_detail::fill_scalar_pattern(reinterpret_cast<hfloat*>(row_ptr), pixel_per_outer, cn, s);
                break;
            default:
                CV_Error_(Error::StsNotImplemented, ("Unsupported type:%d in setTo(Scalar)", d));
        }
    }
}

inline size_t Mat::total() const
{
    if (size.p == nullptr || dims == 0)
    {
        return 0;
    }

    size_t p = 1;
    for (int i = dims - 1; i >= 0; --i)
    {
        p *= static_cast<size_t>(size.p[i]);
    }
    return p;
}

inline size_t Mat::total(int startDim, int endDim) const
{
    CV_Assert(startDim >= 0 && startDim <= endDim);
    size_t p = 1;

    const int end_dim = endDim <= dims ? endDim : dims;
    for (int i = startDim; i < end_dim; ++i)
    {
        p *= static_cast<size_t>(size.p[i]);
    }
    return p;
}

inline void Mat::printShape() const
{
    std::cout << "shape = " << displayShapeString() << std::endl;
}

inline void Mat::print(int len) const
{
    if (empty())
    {
        std::cout << "mat is empty" << std::endl;
        return;
    }

    std::cout << "shape = " << displayShapeString() << std::endl;

    const size_t scalar_total = total() * static_cast<size_t>(channels());
    const size_t printLen = len == -1 ? scalar_total :
                            len > static_cast<int>(scalar_total) ? scalar_total : static_cast<size_t>(len);

    std::cout << "value = ";
    if (depth() == CV_32F)
    {
        const float* p = reinterpret_cast<const float*>(data);
        for (size_t i = 0; i < printLen; ++i)
        {
            std::cout << p[i] << ",";
        }
        std::cout << std::endl;
    }
    else if (depth() == CV_32S)
    {
        const int* p = reinterpret_cast<const int*>(data);
        for (size_t i = 0; i < printLen; ++i)
        {
            std::cout << p[i] << ",";
        }
        std::cout << std::endl;
    }
    else if (depth() == CV_32U)
    {
        const uint* p = reinterpret_cast<const uint*>(data);
        for (size_t i = 0; i < printLen; ++i)
        {
            std::cout << p[i] << ",";
        }
        std::cout << std::endl;
    }
    else if (depth() == CV_8S)
    {
        const schar* p = reinterpret_cast<const schar*>(data);
        for (size_t i = 0; i < printLen; ++i)
        {
            std::cout << static_cast<int>(p[i]) << ",";
        }
        std::cout << std::endl;
    }
    else if (depth() == CV_8U)
    {
        const uchar* p = reinterpret_cast<const uchar*>(data);
        for (size_t i = 0; i < printLen; ++i)
        {
            std::cout << static_cast<int>(p[i]) << ",";
        }
        std::cout << std::endl;
    }
    else if (depth() == CV_16U)
    {
        const ushort* p = reinterpret_cast<const ushort*>(data);
        for (size_t i = 0; i < printLen; ++i)
        {
            std::cout << static_cast<int>(p[i]) << ",";
        }
        std::cout << std::endl;
    }
    else if (depth() == CV_16S)
    {
        const short* p = reinterpret_cast<const short*>(data);
        for (size_t i = 0; i < printLen; ++i)
        {
            std::cout << static_cast<int>(p[i]) << ",";
        }
        std::cout << std::endl;
    }
    else if (depth() == CV_16F)
    {
        const hfloat* p = reinterpret_cast<const hfloat*>(data);
        for (size_t i = 0; i < printLen; ++i)
        {
            std::cout << static_cast<float>(p[i]) << ",";
        }
        std::cout << std::endl;
    }
    else
    {
        CV_Error_(Error::StsBadType, ("Unsupported format at function Mat::print type=%d", type()));
    }
}

inline MatShape Mat::shape() const
{
    if (empty())
    {
        return {};
    }

    MatShape out_shape(static_cast<size_t>(dims), 0);
    std::memcpy(out_shape.data(), size.p, static_cast<size_t>(dims) * sizeof(int));
    return out_shape;
}

inline std::string Mat::shapeString(ShapeDisplayOrder order) const
{
    if (empty())
    {
        return "[]";
    }

    if (order == ShapeDisplayOrder::ChannelFirst && channels() > 1 && dims == 2)
    {
        return lite_mat_detail::format_shape_string({channels(), size.p[0], size.p[1]});
    }

    return lite_mat_detail::format_shape_string(shape());
}

inline std::string Mat::displayShapeString() const
{
    if (!empty() && channels() > 1 && dims == 2)
    {
        return shapeString(ShapeDisplayOrder::ChannelFirst);
    }
    return shapeString(ShapeDisplayOrder::Geometry);
}

inline bool Mat::empty() const
{
    return data == nullptr || total() == 0 || dims == 0;
}

inline uchar* Mat::ptr()
{
    return data;
}

inline const uchar* Mat::ptr() const
{
    return data;
}

inline const uchar* Mat::pixelPtr(int y, int x) const
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
                  ("pixelPtr index out of range, y=%d x=%d shape=[%d,%d]", y, x, size.p[0], size.p[1]));
    }

    return data + static_cast<size_t>(y) * step(0) + static_cast<size_t>(x) * elemSize();
}

inline uchar* Mat::pixelPtr(int y, int x)
{
    return const_cast<uchar*>(static_cast<const Mat*>(this)->pixelPtr(y, x));
}

inline void Mat::addref()
{
    if (u)
    {
        CV_XADD(&u->refcount, 1);
    }
}

inline void Mat::convertTo(Mat& m, int type_) const
{
    if (empty())
    {
        m.release();
        return;
    }

    const int sdepth = depth();
    const int src_channels = channels();

    int ddepth = -1;
    int dchannels = src_channels;
    if (type_ < 0)
    {
        ddepth = sdepth;
    }
    else if (type_ < CV_DEPTH_MAX)
    {
        ddepth = type_;
    }
    else
    {
        ddepth = CV_MAT_DEPTH(type_);
        dchannels = CV_MAT_CN(type_);
    }

    if (!lite_mat_detail::is_supported_depth(sdepth) || !lite_mat_detail::is_supported_depth(ddepth))
    {
        CV_Error_(Error::StsNotImplemented,
                  ("Mat::convertTo supports only depths in [CV_8U..CV_16F], sdepth=%d ddepth=%d", sdepth, ddepth));
    }

    if (dchannels != src_channels)
    {
        CV_Error_(Error::StsBadArg, ("Mat::convertTo channel mismatch, src=%d dst=%d", src_channels, dchannels));
    }

    const int dtype = CV_MAKETYPE(ddepth, dchannels);
    if (type() == dtype)
    {
        copyTo(m);
        return;
    }

    m.create(dims, size.p, dtype);

    const size_t scalar_count = m.total() * static_cast<size_t>(m.channels());
    if (isContinuous() && m.isContinuous())
    {
        if (!lite_mat_detail::convert_dispatch(sdepth, ddepth, data, m.data, scalar_count))
        {
            CV_Error_(Error::StsNotImplemented,
                      ("Mat::convertTo conversion miss, sdepth=%d ddepth=%d", sdepth, ddepth));
        }
        return;
    }

    const int outer = dims > 1 ? size.p[0] : 1;
    const size_t scalar_per_outer = (dims > 1 ? total(1, dims) : total()) * static_cast<size_t>(channels());
    const size_t src_step0 = dims > 1 ? step(0) : scalar_per_outer * elemSize1();
    const size_t dst_step0 = m.dims > 1 ? m.step(0) : scalar_per_outer * m.elemSize1();

    for (int i = 0; i < outer; ++i)
    {
        const uchar* src_row = data + static_cast<size_t>(i) * src_step0;
        uchar* dst_row = m.data + static_cast<size_t>(i) * dst_step0;
        if (!lite_mat_detail::convert_dispatch(sdepth, ddepth, src_row, dst_row, scalar_per_outer))
        {
            CV_Error_(Error::StsNotImplemented,
                      ("Mat::convertTo conversion miss, sdepth=%d ddepth=%d", sdepth, ddepth));
        }
    }
}

} // namespace cvh

#endif // CVH_CORE_MAT_LITE_IMPL_H
