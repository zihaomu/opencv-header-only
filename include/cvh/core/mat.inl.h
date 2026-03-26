//
// Created by mzh on 2024/3/27.
//

#ifndef CVH_MAT_INL_H
#define CVH_MAT_INL_H

namespace cvh
{

inline
MatSize::MatSize(int* _p) : p0(_p)
{
    if (_p)
        p = _p + 1;
    else
        p = nullptr;
}

inline
int MatSize::dims() const
{
    return p0[0];
}

inline
const int& MatSize::operator[](int i) const
{
    return p[i];
}

inline
int& MatSize::operator[](int i)
{
    return p[i];
}

inline
bool MatSize::operator!=(const MatSize& sz) const
{
    return !(*this == sz);
}

inline
bool MatSize::operator==(const MatSize& sz) const
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

inline
size_t total(const Mat& m)
{
    return m.total();
}

inline
size_t total(const Mat& m, int startDim, int endDim)
{
    if (endDim == -1)
    {
        endDim = m.dims;
    }
    return m.total(startDim, endDim);
}

inline
size_t total(const MatShape shape)
{
    return total(shape, 0, static_cast<int>(shape.size()));
}

inline
size_t total(const MatShape shape, int startDim, int endDim)
{
    if (endDim == -1)
    {
        endDim = static_cast<int>(shape.size());
    }

    assert(startDim >= 0 && startDim <= endDim);
    size_t p = 1;
    int dims = static_cast<int>(shape.size());

    int endDim_ = endDim <= dims ? endDim : dims;
    for (int i = startDim; i < endDim_; i++)
    {
        p *= static_cast<size_t>(shape[i]);
    }
    return p;
}

inline
MatExpr::MatExpr()
        : op(0), flags(0), a(Mat()), b(Mat()), c(Mat()), alpha(0), beta(0)
{}

inline
MatExpr::MatExpr(const MatOp* _op, int _flags, const Mat& _a, const Mat& _b,
                 const Mat& _c, double _alpha, double _beta)
        : op(_op), flags(_flags), a(_a), b(_b), c(_c), alpha(_alpha), beta(_beta)
{}

inline
MatExpr::operator Mat() const
{
    Mat m;
    op->assign(*this, m);
    return m;
}

inline
Mat& Mat::operator = (const MatExpr& e)
{
    e.op->assign(e, *this);
    return *this;
}

template<typename _Tp>
inline _Tp& Mat::at(int i0)
{
    if (empty())
    {
        CV_Error(Error::StsBadArg, "Mat::at(i0) expects a non-empty Mat");
    }
    if (i0 < 0)
    {
        CV_Error_(Error::StsOutOfRange, ("Mat::at(i0) out of range, i0=%d", i0));
    }
    return reinterpret_cast<_Tp*>(data)[i0];
}

template<typename _Tp>
inline const _Tp& Mat::at(int i0) const
{
    if (empty())
    {
        CV_Error(Error::StsBadArg, "Mat::at(i0) expects a non-empty Mat");
    }
    if (i0 < 0)
    {
        CV_Error_(Error::StsOutOfRange, ("Mat::at(i0) out of range, i0=%d", i0));
    }
    return reinterpret_cast<const _Tp*>(data)[i0];
}

template<typename _Tp>
inline const _Tp& Mat::at(int y, int x, int ch) const
{
    if (empty())
    {
        CV_Error(Error::StsBadArg, "Mat::at(y,x,ch) expects a non-empty Mat");
    }
    if (dims != 2)
    {
        CV_Error_(Error::StsBadArg, ("Mat::at(y,x,ch) expects a 2D Mat, dims=%d", dims));
    }
    if (ch < 0 || ch >= channels())
    {
        CV_Error_(Error::StsOutOfRange,
                  ("Mat::at(y,x,ch) channel out of range, ch=%d channels=%d", ch, channels()));
    }
    if (sizeof(_Tp) != elemSize1())
    {
        CV_Error_(Error::StsBadType,
                  ("Mat::at(y,x,ch) type size mismatch, sizeof(T)=%zu elemSize1=%zu",
                   sizeof(_Tp), elemSize1()));
    }

    const uchar* p = pixelPtr(y, x) + static_cast<size_t>(ch) * elemSize1();
    return *reinterpret_cast<const _Tp*>(p);
}

template<typename _Tp>
inline _Tp& Mat::at(int y, int x, int ch)
{
    return const_cast<_Tp&>(static_cast<const Mat*>(this)->at<_Tp>(y, x, ch));
}

static inline
Mat& operator += (Mat& a, const MatExpr& b)
{
    b.op->augAssginAdd(b, a);
    return a;
}

static inline
const Mat& operator += (const Mat& a, const MatExpr& b)
{
    b.op->augAssginAdd(b, (Mat&)a);
    return a;
}

static inline
Mat& operator -= (Mat& a, const MatExpr& b)
{
    b.op->augAssginSubtract(b, a);
    return a;
}

static inline
const Mat& operator -= (const Mat& a, const MatExpr& b)
{
    b.op->augAssginSubtract(b, (Mat&)a);
    return a;
}

static inline
Mat& operator *= (Mat& a, const MatExpr& b)
{
    b.op->augAssginMultiply(b, a);
    return a;
}

static inline
const Mat& operator *= (const Mat& a, const MatExpr& b)
{
    b.op->augAssginMultiply(b, (Mat&)a);
    return a;
}

static inline
Mat& operator /= (Mat& a, const MatExpr& b)
{
    b.op->augAssginDivide(b, a);
    return a;
}

static inline
const Mat& operator /= (const Mat& a, const MatExpr& b)
{
    b.op->augAssginDivide(b, (Mat&)a);
    return a;
}

}

#endif //CVH_MAT_INL_H
