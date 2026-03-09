//
// Created by mzh on 2024/3/27.
//

#ifndef MINFER_MAT_INL_H
#define MINFER_MAT_INL_H

namespace minfer
{

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

#endif //MINFER_MAT_INL_H
