#ifndef CVH_CORE_DETAIL_MAT_EXPR_IMPL_HPP
#define CVH_CORE_DETAIL_MAT_EXPR_IMPL_HPP

namespace cvh {

// This and its overload below are used in various MatExpr operator overloads
// implemented to check that Matrix operands exists.
static void checkOperandsExist(const Mat& a)
{
    if (a.empty())
    {
        CV_Assert(0 && "Matrix operand is an empty matrix!");
    }
}

static void checkOperandsExist(const Mat& a, const Mat& b)
{
    if (a.empty() || b.empty())
    {
        CV_Assert(0 && "One or more matrix operands are empty!");
    }
}

static Mat evalExprToMat(const MatExpr& e)
{
    Mat out;
    e.op->assign(e, out);
    return out;
}

static MatExpr makeCmpExpr(const Mat& a, const Mat& b, int cmpop)
{
    checkOperandsExist(a, b);
    Mat out;
    compare(a, b, out, cmpop);
    return MatExpr(out);
}

static MatExpr makeCmpExprWithScalar(const Mat& a, const Scalar& s, int cmpop, bool scalar_first)
{
    checkOperandsExist(a);
    Mat out;
    if (scalar_first)
    {
        compare(s, a, out, cmpop);
    }
    else
    {
        compare(a, s, out, cmpop);
    }
    return MatExpr(out);
}

class MatOp_Identity : public MatOp
{
public:
    MatOp_Identity() {};
    virtual ~MatOp_Identity() {};

    bool elementWise(const MatExpr& ) const override {return true;}

    void assign(const MatExpr& expr, Mat& m, int type = -1) const override;

    static void makeExpr(MatExpr& res, const Mat& a);
};

static MatOp_Identity g_MatOp_Identity;

// Cover add and subtract operator
class MatOp_AddEx : public MatOp
{
public:
    MatOp_AddEx() {};
    virtual ~MatOp_AddEx() {};

    bool elementWise(const MatExpr& ) const override {return true;}

    void assign(const MatExpr& expr, Mat& m, int type = -1) const override;

    static void makeExpr(MatExpr& res, const Mat& a, const Mat& b, double alpha, double beta);
};

static MatOp_AddEx g_MatOp_AddEx;

class MatOp_Bin : public MatOp
{
public:
    MatOp_Bin() {};
    virtual ~MatOp_Bin() {};
    void assign(const MatExpr& expr, Mat& m, int type = -1) const override;

    static void makeExpr(MatExpr& res, char op, const Mat& a, const Mat& b);
};

static MatOp_Bin g_MatOp_Bin;


class MatOp_Cmp : public MatOp
{
public:
    MatOp_Cmp() {}
    virtual ~MatOp_Cmp() {};

    bool elementWise(const MatExpr& ) const override {return true;}

    void assign(const MatExpr& expr, Mat& m, int type = -1) const override;
    int type(const MatExpr& expr) const override;

    static void makeExpr(MatExpr& res, int cmpop, const Mat& a, const Mat& b);
};

static MatOp_Cmp g_MatOp_Cmp;

///////////// MatOp Implementation //////////////

inline void MatOp_Identity::assign(const MatExpr &e, Mat &m, int _type) const
{
    if (_type == -1 || _type == e.a.type())
        m = e.a;
    else
        CV_Assert(0 && "Unsupported data type convert!");
}

inline void MatOp_Identity::makeExpr(MatExpr &res, const Mat &a)
{
    res = MatExpr(&g_MatOp_Identity, 0, a, Mat(), Mat(), 1, 0);
}

inline void MatOp_AddEx::assign(const MatExpr &e, Mat &m, int _type) const
{
    Mat temp, &dst = _type == -1 || e.a.type() == _type ? m : temp;

    if (e.b.data)
    {
        if (e.beta == 1)
        {
            cvh::add(e.a, e.b, dst);
        }
        else if (e.beta == -1)
        {
            cvh::subtract(e.a, e.b, dst);
        }
        else
            cvh::addWeighted(e.a, e.alpha, e.b, e.beta, dst);
    }
    else if (e.alpha == 1)
    {
        dst = e.a;
    }
    else if (e.alpha == -1)
    {
        cvh::subtract(e.a, dst);
    }
    else
        CV_Assert(0 && "Unsupported type in MatOp_AddEx::assign!");
}

inline void MatOp_AddEx::makeExpr(MatExpr &res, const Mat &a, const Mat &b, double alpha, double beta)
{
    res = MatExpr(&g_MatOp_AddEx, 0, a, b, Mat(), alpha, beta);
}

inline void MatOp_Bin::assign(const MatExpr &e, Mat &m, int _type) const
{
    Mat temp, &dst = _type == -1 || e.a.type() == _type ? m : temp;

    if (e.flags == '*')
    {
        cvh::multiply(e.a, e.b, dst);
    }
    else if (e.flags == '/' && e.b.data)
    {
        cvh::divide(e.a, e.b, dst);
    }
    else
        CV_Error(Error::StsNotImplemented, "Unsupported value!");
}

inline void MatOp_Bin::makeExpr(MatExpr& res, char op, const Mat& a, const Mat& b)
{
    res = MatExpr(&g_MatOp_Bin, op, a, b, Mat());
}

inline void MatOp_Cmp::assign(const MatExpr &expr, Mat &m, int _type) const
{
    const int expected_type = CV_MAKETYPE(CV_8U, expr.a.channels());
    if (_type != -1 && _type != expected_type)
    {
        CV_Error_(Error::StsBadType,
                  ("MatOp_Cmp::assign only supports dst type=%d, got=%d", expected_type, _type));
    }

    cvh::compare(expr.a, expr.b, m, expr.flags);
}

inline int MatOp_Cmp::type(const MatExpr &expr) const
{
    if (expr.a.empty())
    {
        return -1;
    }
    return CV_MAKETYPE(CV_8U, expr.a.channels());
}

inline void MatOp_Cmp::makeExpr(MatExpr &res, int cmpop, const Mat &a, const Mat &b)
{
    res = MatExpr(&g_MatOp_Cmp, cmpop, a, b, Mat());
}

////////////// Implementation ////////////

inline MatOp::MatOp() {}

inline MatOp::~MatOp() {}

inline bool MatOp::elementWise(const MatExpr&) const
{
    return false;
}

inline void MatOp::augAssginAdd(const MatExpr &expr, Mat &m) const
{
    Mat temp;
    expr.op->assign(expr, temp);
    m += temp;
}

inline void MatOp::augAssginSubtract(const MatExpr &expr, Mat &m) const
{
    Mat temp;
    expr.op->assign(expr, temp);
    m -= temp;
}

inline void MatOp::augAssginMultiply(const MatExpr &expr, Mat &m) const
{
    Mat temp;
    expr.op->assign(expr, temp);
    m *= temp;
}

inline void MatOp::augAssginDivide(const MatExpr &expr, Mat &m) const
{
    Mat temp;
    expr.op->assign(expr, temp);
    m /= temp;
}

inline int MatOp::type(const MatExpr &expr) const
{
    return !expr.a.empty() ? expr.a.type() : expr.b.empty() ? expr.b.type() : expr.c.type();
}

static inline bool isAddEx(const MatExpr& e) {return e.op == & g_MatOp_AddEx; }
static inline bool isBin(const MatExpr& e, char c) {return e.op == &g_MatOp_Bin && e.flags == c; }
static inline bool isReciprocal(const MatExpr& e) {return isBin(e, '/') && (!e.b.data || e.beta == 0);}

inline void MatOp::add(const MatExpr &e1, const MatExpr &e2, MatExpr &res) const
{
    if (this == e2.op)
    {
        double alpha = 1, beta = 1;

        Mat m1, m2;

        if (isAddEx(e1) && (!e1.b.data || e1.beta == 0))
        {
            m1 = e1.a;
            alpha = e1.alpha;
        }
        else
            e1.op->assign(e1, m1);

        if(isAddEx(e2) && (!e2.b.data || e2.beta == 0))
        {
            m2 = e2.a;
            beta = e2.alpha;
        }
        else
            e2.op->assign(e2, m2);
        MatOp_AddEx::makeExpr(res, m1, m2, alpha, beta);
    }
    else
        e2.op->add(e1, e2, res);
}

inline void MatOp::subtract(const MatExpr &e1, const MatExpr &e2, MatExpr &res) const
{
    if (this == e2.op)
    {
        double alpha = 1, beta = -1;

        Mat m1, m2;

        if (isAddEx(e1) && (!e1.b.data || e1.beta == 0))
        {
            m1 = e1.a;
            alpha = e1.alpha;
        }
        else
            e1.op->assign(e1, m1);

        if(isAddEx(e2) && (!e2.b.data || e2.beta == 0))
        {
            m2 = e2.a;
            beta = -e2.alpha;
        }
        else
            e2.op->assign(e2, m2);
        MatOp_AddEx::makeExpr(res, m1, m2, alpha, beta);
    }
    else
        e2.op->subtract(e1, e2, res);
}

inline void MatOp::multiply(const MatExpr &e1, const MatExpr &e2, MatExpr &res) const
{
    if (this == e2.op)
    {
        Mat m1, m2;

        if (isReciprocal(e1))
        {
            e2.op->assign(e2, m2);
            MatOp_Bin::makeExpr(res, '/', m2, e1.a);
        }
        else
        {
            char op = '*';

            e1.op->assign(e1, m1);

            if (isReciprocal(e2))
            {
                m2 = e2.a;
                op = '/';
            }
            else
                e2.op->assign(e2, m2);

            MatOp_Bin::makeExpr(res, op, m1, m2);
        }
    }
    else
        e2.op->multiply(e1, e2, res);
}

inline void MatOp::divide(const MatExpr &e1, const MatExpr &e2, MatExpr &res) const
{
    if (this == e2.op)
    {
        if (isReciprocal(e1) && isReciprocal(e2))
        {
            MatOp_Bin::makeExpr(res, '/', e2.a, e1.a);
        }
        else
        {
            Mat m1, m2;
            char op = '/';

            e1.op->assign(e1, m1);
            if (isReciprocal(e2))
            {
                m2 = e2.a;
                op = '*';
            }
            else
                e2.op->assign(e2, m2);

            MatOp_Bin::makeExpr(res, op, m1, m2);
        }
    }
    else
        e2.op->divide(e1, e2, res);
}

//////////// MatExpr implementation /////////////

inline MatExpr::MatExpr(const Mat &m)
        : op(&g_MatOp_Identity), flags(0), a(m), b(Mat()), c(Mat()), alpha(1), beta(0)
{

}

inline int MatExpr::type() const
{
    return op ? op->type(*this) : -1;
}


//////////// MatExpr /////////
inline MatExpr operator + (const Mat& a, const Mat& b)
{
    checkOperandsExist(a, b);
    MatExpr e;
    MatOp_AddEx::makeExpr(e, a, b, 1, 1);
    return e;
}

inline MatExpr operator + (const Mat& a, const MatExpr& e)
{
    checkOperandsExist(a);
    MatExpr en;
    e.op->add(e, MatExpr(a), en);
    return en;
}

inline MatExpr operator + (const MatExpr& e, const Mat& b)
{
    MatExpr en;
    e.op->add(e, MatExpr(b), en);
    return en;
}

inline MatExpr operator + (const MatExpr& e1, const MatExpr& e2)
{
    MatExpr en;
    e1.op->add(e1, e2, en);
    return en;
}

inline MatExpr operator + (const MatExpr& e1, const float e)
{
    // convert float to specific type.
    int type = e1.type();
    Mat em = Mat({1}, type);
    em.setTo(e);

    MatExpr en;
    e1.op->add(e1, MatExpr(em), en);
    return en;
}

inline MatExpr operator + (const float e, const MatExpr& e2)
{
    // convert float to specific type.
    int type = e2.type();
    Mat em = Mat({1}, type);
    em.setTo(e);

    MatExpr en;
    e2.op->add(MatExpr(em), e2, en);
    return en;
}

inline MatExpr operator + (const Mat& a, const float e)
{
    checkOperandsExist(a);

    int type = a.type();
    Mat b = Mat({1}, type);
    b.setTo(e);

    MatExpr en;
    MatOp_AddEx::makeExpr(en, a, b, 1, 1);
    return en;
}

inline MatExpr operator + (const float e, const Mat& b)
{
    checkOperandsExist(b);

    int type = b.type();
    Mat a = Mat({1}, type);
    a.setTo(e);

    MatExpr en;
    MatOp_AddEx::makeExpr(en, a, b, 1, 1);
    return en;
}

inline MatExpr operator - (const Mat& a)
{
    checkOperandsExist(a);
    MatExpr en;
    MatOp_AddEx::makeExpr(en, a, Mat(), -1, 0);
    return en;
}

inline MatExpr operator - (const MatExpr& e)
{
    MatExpr en;
    e.op->subtract(MatExpr(Mat()), e, en);
    return en;
}

inline MatExpr operator - (const Mat& a, const Mat& b)
{
    checkOperandsExist(a, b);
    MatExpr e;
    MatOp_AddEx::makeExpr(e, a, b, 1, -1);
    return e;
}

inline MatExpr operator - (const Mat& a, const MatExpr& e)
{
    checkOperandsExist(a);
    MatExpr en;
    e.op->subtract(MatExpr(a), e, en);
    return en;
}

inline MatExpr operator - (const MatExpr& e, const Mat& b)
{
    MatExpr en;
    e.op->subtract(e, MatExpr(b), en);
    return en;
}

inline MatExpr operator - (const MatExpr& e1, const MatExpr& e2)
{
    MatExpr en;
    e1.op->subtract(e1, e2, en);
    return en;
}

inline MatExpr operator - (const MatExpr& e1, const float e)
{
    // convert float to specific type.
    int type = e1.type();
    Mat em = Mat({1}, type);
    em.setTo(e);

    MatExpr en;
    e1.op->subtract(e1, MatExpr(em), en);
    return en;
}

inline MatExpr operator - (const float e, const MatExpr& e2)
{
    // convert float to specific type.
    int type = e2.type();
    Mat em = Mat({1}, type);
    em.setTo(e);

    MatExpr en;
    e2.op->subtract(MatExpr(em), e2, en);
    return en;
}

inline MatExpr operator - (const Mat& a, const float e)
{
    checkOperandsExist(a);

    int type = a.type();
    Mat b = Mat({1}, type);
    b.setTo(e);

    MatExpr en;
    MatOp_AddEx::makeExpr(en, a, b, 1, -1);
    return en;
}

inline MatExpr operator - (const float e, const Mat& b)
{
    checkOperandsExist(b);

    int type = b.type();
    Mat a = Mat({1}, type);
    a.setTo(e);

    MatExpr en;
    MatOp_AddEx::makeExpr(en, a, b, 1, -1);
    return en;
}

inline MatExpr operator * (const Mat& a, const Mat& b)
{
    checkOperandsExist(a, b);
    MatExpr en;

    MatOp_Bin::makeExpr(en, '*', a, b);
    return en;
}

inline MatExpr operator * (const Mat& a, const MatExpr& e)
{
    checkOperandsExist(a);

    MatExpr en;
    e.op->multiply(MatExpr(a), e, en);
    return en;
}

inline MatExpr operator * (const MatExpr& e, const Mat& b)
{
    MatExpr en;
    e.op->multiply(e, MatExpr(b), en);
    return en;
}

inline MatExpr operator * (const MatExpr& e1, const MatExpr& e2)
{
    MatExpr en;
    e1.op->multiply(e1, e2, en);
    return en;
}

inline MatExpr operator * (const MatExpr& e1, const float e)
{
    Mat lhs = evalExprToMat(e1);
    Mat out;
    multiply(lhs, Scalar::all(static_cast<double>(e)), out);
    return MatExpr(out);
}

inline MatExpr operator * (const float e, const MatExpr& e2)
{
    Mat rhs = evalExprToMat(e2);
    Mat out;
    multiply(Scalar::all(static_cast<double>(e)), rhs, out);
    return MatExpr(out);
}

inline MatExpr operator * (const Mat& a, const float e)
{
    checkOperandsExist(a);
    Mat out;
    multiply(a, Scalar::all(static_cast<double>(e)), out);
    return MatExpr(out);
}

inline MatExpr operator * (const float e, const Mat& b)
{
    checkOperandsExist(b);
    Mat out;
    multiply(Scalar::all(static_cast<double>(e)), b, out);
    return MatExpr(out);
}

inline MatExpr operator / (const Mat& a, const Mat& b)
{
    checkOperandsExist(a, b);
    MatExpr e;
    MatOp_Bin::makeExpr(e, '/', a, b);
    return e;
}

inline MatExpr operator / (const Mat& a, const MatExpr& e)
{
    checkOperandsExist(a);

    MatExpr en;
    e.op->divide(MatExpr(a), e, en);
    return en;
}

inline MatExpr operator / (const MatExpr& e, const Mat& b)
{
    MatExpr en;
    e.op->divide(e, MatExpr(b), en);
    return en;
}

inline MatExpr operator / (const MatExpr& e1, const MatExpr& e2)
{
    MatExpr en;
    e1.op->divide(e1, e2, en);
    return en;
}

inline MatExpr operator / (const MatExpr& e1, const float e)
{
    Mat lhs = evalExprToMat(e1);
    Mat out;
    divide(lhs, Scalar::all(static_cast<double>(e)), out);
    return MatExpr(out);
}

inline MatExpr operator / (const float e, const MatExpr& e2)
{
    Mat rhs = evalExprToMat(e2);
    Mat out;
    divide(Scalar::all(static_cast<double>(e)), rhs, out);
    return MatExpr(out);
}

inline MatExpr operator / (const Mat& a, const float e)
{
    checkOperandsExist(a);
    Mat out;
    divide(a, Scalar::all(static_cast<double>(e)), out);
    return MatExpr(out);
}

inline MatExpr operator / (const float e, const Mat& b)
{
    checkOperandsExist(b);
    Mat out;
    divide(Scalar::all(static_cast<double>(e)), b, out);
    return MatExpr(out);
}

inline MatExpr operator == (const Mat& a, const Mat& b)
{
    return makeCmpExpr(a, b, CV_CMP_EQ);
}

inline MatExpr operator + (const MatExpr& e1, const Scalar& s)
{
    Mat lhs = evalExprToMat(e1);
    Mat out;
    add(lhs, s, out);
    return MatExpr(out);
}

inline MatExpr operator + (const Scalar& s, const MatExpr& e2)
{
    Mat rhs = evalExprToMat(e2);
    Mat out;
    add(s, rhs, out);
    return MatExpr(out);
}

inline MatExpr operator + (const Mat& a, const Scalar& s)
{
    checkOperandsExist(a);
    Mat out;
    add(a, s, out);
    return MatExpr(out);
}

inline MatExpr operator + (const Scalar& s, const Mat& b)
{
    checkOperandsExist(b);
    Mat out;
    add(s, b, out);
    return MatExpr(out);
}

inline MatExpr operator - (const MatExpr& e1, const Scalar& s)
{
    Mat lhs = evalExprToMat(e1);
    Mat out;
    subtract(lhs, s, out);
    return MatExpr(out);
}

inline MatExpr operator - (const Scalar& s, const MatExpr& e2)
{
    Mat rhs = evalExprToMat(e2);
    Mat out;
    subtract(s, rhs, out);
    return MatExpr(out);
}

inline MatExpr operator - (const Mat& a, const Scalar& s)
{
    checkOperandsExist(a);
    Mat out;
    subtract(a, s, out);
    return MatExpr(out);
}

inline MatExpr operator - (const Scalar& s, const Mat& b)
{
    checkOperandsExist(b);
    Mat out;
    subtract(s, b, out);
    return MatExpr(out);
}

inline MatExpr operator * (const MatExpr& e1, const Scalar& s)
{
    Mat lhs = evalExprToMat(e1);
    Mat out;
    multiply(lhs, s, out);
    return MatExpr(out);
}

inline MatExpr operator * (const Scalar& s, const MatExpr& e2)
{
    Mat rhs = evalExprToMat(e2);
    Mat out;
    multiply(s, rhs, out);
    return MatExpr(out);
}

inline MatExpr operator * (const Mat& a, const Scalar& s)
{
    checkOperandsExist(a);
    Mat out;
    multiply(a, s, out);
    return MatExpr(out);
}

inline MatExpr operator * (const Scalar& s, const Mat& b)
{
    checkOperandsExist(b);
    Mat out;
    multiply(s, b, out);
    return MatExpr(out);
}

inline MatExpr operator / (const MatExpr& e1, const Scalar& s)
{
    Mat lhs = evalExprToMat(e1);
    Mat out;
    divide(lhs, s, out);
    return MatExpr(out);
}

inline MatExpr operator / (const Scalar& s, const MatExpr& e2)
{
    Mat rhs = evalExprToMat(e2);
    Mat out;
    divide(s, rhs, out);
    return MatExpr(out);
}

inline MatExpr operator / (const Mat& a, const Scalar& s)
{
    checkOperandsExist(a);
    Mat out;
    divide(a, s, out);
    return MatExpr(out);
}

inline MatExpr operator / (const Scalar& s, const Mat& b)
{
    checkOperandsExist(b);
    Mat out;
    divide(s, b, out);
    return MatExpr(out);
}

inline MatExpr operator == (const Mat& a, const Scalar& s)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_EQ, false);
}

inline MatExpr operator == (const Scalar& s, const Mat& a)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_EQ, true);
}

inline MatExpr operator == (const Mat& a, const float s)
{
    return a == Scalar::all(static_cast<double>(s));
}

inline MatExpr operator == (const float s, const Mat& a)
{
    return Scalar::all(static_cast<double>(s)) == a;
}

inline MatExpr operator != (const Mat& a, const Mat& b)
{
    return makeCmpExpr(a, b, CV_CMP_NE);
}

inline MatExpr operator != (const Mat& a, const Scalar& s)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_NE, false);
}

inline MatExpr operator != (const Scalar& s, const Mat& a)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_NE, true);
}

inline MatExpr operator != (const Mat& a, const float s)
{
    return a != Scalar::all(static_cast<double>(s));
}

inline MatExpr operator != (const float s, const Mat& a)
{
    return Scalar::all(static_cast<double>(s)) != a;
}

inline MatExpr operator < (const Mat& a, const Mat& b)
{
    return makeCmpExpr(a, b, CV_CMP_LT);
}

inline MatExpr operator < (const Mat& a, const Scalar& s)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_LT, false);
}

inline MatExpr operator < (const Scalar& s, const Mat& a)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_LT, true);
}

inline MatExpr operator < (const Mat& a, const float s)
{
    return a < Scalar::all(static_cast<double>(s));
}

inline MatExpr operator < (const float s, const Mat& a)
{
    return Scalar::all(static_cast<double>(s)) < a;
}

inline MatExpr operator <= (const Mat& a, const Mat& b)
{
    return makeCmpExpr(a, b, CV_CMP_LE);
}

inline MatExpr operator <= (const Mat& a, const Scalar& s)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_LE, false);
}

inline MatExpr operator <= (const Scalar& s, const Mat& a)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_LE, true);
}

inline MatExpr operator <= (const Mat& a, const float s)
{
    return a <= Scalar::all(static_cast<double>(s));
}

inline MatExpr operator <= (const float s, const Mat& a)
{
    return Scalar::all(static_cast<double>(s)) <= a;
}

inline MatExpr operator > (const Mat& a, const Mat& b)
{
    return makeCmpExpr(a, b, CV_CMP_GT);
}

inline MatExpr operator > (const Mat& a, const Scalar& s)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_GT, false);
}

inline MatExpr operator > (const Scalar& s, const Mat& a)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_GT, true);
}

inline MatExpr operator > (const Mat& a, const float s)
{
    return a > Scalar::all(static_cast<double>(s));
}

inline MatExpr operator > (const float s, const Mat& a)
{
    return Scalar::all(static_cast<double>(s)) > a;
}

inline MatExpr operator >= (const Mat& a, const Mat& b)
{
    return makeCmpExpr(a, b, CV_CMP_GE);
}

inline MatExpr operator >= (const Mat& a, const Scalar& s)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_GE, false);
}

inline MatExpr operator >= (const Scalar& s, const Mat& a)
{
    return makeCmpExprWithScalar(a, s, CV_CMP_GE, true);
}

inline MatExpr operator >= (const Mat& a, const float s)
{
    return a >= Scalar::all(static_cast<double>(s));
}

inline MatExpr operator >= (const float s, const Mat& a)
{
    return Scalar::all(static_cast<double>(s)) >= a;
}

}

#endif  // CVH_CORE_DETAIL_MAT_EXPR_IMPL_HPP
