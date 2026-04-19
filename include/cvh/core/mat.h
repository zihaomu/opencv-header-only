//
// Created by mzh on 2024/3/27.
//

#ifndef CVH_MAT_H
#define CVH_MAT_H

#include "../detail/config.h"

#include <iostream>
#include <assert.h>
#include <string>
#include <vector>

#include "define.h"
#include "system.h"
#include "types.h"

namespace cvh
{

typedef std::vector<int> MatShape;
#define MAT_MAX_DIM 8 // Max mat dimension

enum class ShapeDisplayOrder
{
    Geometry = 0,
    ChannelFirst = 1,
};

struct Range
{
    Range();
    Range(int _start, int _end);
    static Range all();

    int start;
    int end;
};

class MatAllocator;
struct MatData;
class Mat;
class MatOp;
class MatExpr;

enum MatDataUsageFlags
{
    USAGE_DEFAULT = 0,
    USAGE_ALLOCATE_HOST_MEMORY = 1 << 0,
    USAGE_ALLOCATE_DEVICE_MEMORY = 1 << 1,
};

// API for mat memory allocator
// 支持不同后端，需要支持不同的Mat
class MatAllocator
{
public:
    MatAllocator() {}
    virtual ~MatAllocator() {}

    virtual MatData* allocate(int dims, const int* sizes, int type, void* data) const = 0;
    virtual bool allocate(MatData* data, MatDataUsageFlags usageFlags) const = 0;
    virtual void deallocate(MatData* data) const = 0;
    virtual void map(MatData* data) const;
    virtual void unmap(MatData* data) const;
};

// Some basic information
struct MatData
{
    enum MemoryFlag
    {
        HOST_MEMORY = 1,
        DEVICE_MEMORY = 2,
        HOST_DEVICE_MEMORY = 3,
        USER_MEMORY = 4,
    };

    MatData(const MatAllocator* allocator);
    ~MatData();

    const MatAllocator* allocator;
    int refcount;
    uchar* data;

    size_t size;
    MatData::MemoryFlag flags;
};

struct MatSize
{
    MatSize(int* _p);
    int dims() const;
    const int& operator[](int i) const;
    int& operator[](int i);
    bool operator == (const MatSize& sz) const;
    bool operator != (const MatSize& sz) const;

    int* p0; // p0[0] is dim
    int* p;  // p is p0+1
};

// Mat class, 从OpenCV处抄过来
class Mat
{
public:
    Mat();

    Mat(int dims, const int* sizes, int type);

    Mat(const std::vector<int> sizes, int type);

    // create specific dimension Mat with given default value.
    // Note the value is int type, if you wanna set the default value as other type
    // please reinterpret_cast it to int value first.
    Mat(int dims, const int* sizes, int type, int v);

    Mat(const std::vector<int> sizes, int type, int v);

    // when use reference mat, we need to create new memory for the mat size.
    Mat(const Mat& m);

    Mat(int dims, const int* sizes, int type, void* data);

    Mat(const std::vector<int> sizes, int type, void* data);

    ~Mat();

    // No data copy, just add reference counter.
    Mat& operator=(const Mat& m);

    // Mat& operator+=(const Mat& m);
    // Mat& operator-=(const Mat& m);
    // Mat& operator/=(const Mat& m);
    // Mat& operator*=(const Mat& m);

    Mat& operator=(const MatExpr& e);

    // set all mat value to given value, it will not change the mat type, just assign value according the data type.
    Mat& operator=(const float v);

    // set all mat value to given value
    Mat& operator=(const int v);
    Mat& operator=(const Scalar& s);

    // Create a full copy of the array and the underlying data.
    Mat clone() const;

    // reshape the mat to new shape
    Mat reshape(int newDims, const int* newSizes) const;

    Mat reshape(const std::vector<int> newSizes) const;

    // Reinterpret Mat element type without data conversion/copy.
    // v1 only supports byte-size-preserving reinterpret.
    Mat reinterpret(int rtype) const;

    Mat rowRange(int startrow, int endrow) const;
    Mat colRange(int startcol, int endcol) const;
    Mat operator()(const Range& rowRange, const Range& colRange) const;

    void copyTo(Mat& m) const;

    // convert mat to other mat with specific data type.
    void convertTo(Mat& m, int rtype) const;

    void create(int ndims, const int* sizes, int type);

    void create(const std::vector<int> sizes, int type);

    void release();

    void deallocate();

    int type() const;
    int depth() const;
    int channels() const;
    size_t elemSize() const;
    size_t elemSize1() const;
    size_t step(int dim = 0) const;
    size_t step1(int dim = 0) const;
    bool isContinuous() const;

    // 只设置 Mat的shape而不分配内存
    // ⚠️ 这个有可能打破Mat的安全性，double check
    void setSize(std::vector<int> size);

    void setSize(int dim, const int* size);

    // 设置和输入Mat 一样的shape
    void setSize(Mat& m);

    // 将Mat中的所有数值设置到给定的v中，注意，内部会将float v转换成Mat所对应的数据类型。
    void setTo(float v);
    void setTo(const Scalar& s);

    size_t total() const;

    size_t total(int startDim, int endDim=MAT_MAX_DIM) const;

    // print the mat value and shape.
    void print(int len = -1) const;

    void printShape() const;

    // Geometry-only view by default. ChannelFirst is a display-only projection.
    std::string shapeString(ShapeDisplayOrder order = ShapeDisplayOrder::Geometry) const;
    // Policy used by print()/printShape(): for 2D multi-channel mat show [C,H,W], otherwise geometry.
    std::string displayShapeString() const;

    MatShape shape() const;

    bool empty() const;

    void copySize(const Mat& m);

    uchar* ptr();
    const uchar* ptr() const;

    // Returns the address of the first channel of pixel (y, x) for 2D Mat.
    uchar* pixelPtr(int y = 0, int x = 0);
    const uchar* pixelPtr(int y = 0, int x = 0) const;

    template<typename _Tp> _Tp& at(int i0 = 0);
    template<typename _Tp> const _Tp& at(int i0 = 0) const;
    template<typename _Tp> _Tp& at(int y, int x, int ch = 0);
    template<typename _Tp> const _Tp& at(int y, int x, int ch = 0) const;

    // This a atomic operation. The method increments the reference counter associated with the matrix data.
    void addref();

    static MatAllocator* getStdAllocator();
    static MatAllocator* getDefaultAllocator();

    int dims;
    uchar* data;

    MatAllocator* allocator;

    MatData* u;
    MatSize size;
    size_t stepBuf[MAT_MAX_DIM] = {0};
    int matType;

    struct MatExtrInfo;
    std::shared_ptr<MatExtrInfo> extrInfo; // 保存backend的信息, TODO 暂时没用到
};

///////////////////////////////// Matrix Expressions //////////////

// virtual class for computation operator
class MatOp
{
public:
    MatOp();
    virtual ~MatOp();

    virtual bool elementWise(const MatExpr& expr) const;
    virtual void assign(const MatExpr& expr, Mat& m, int type = -1) const = 0;

    // augment assign? for compound assignment, +=, -=, *= ...
    virtual void augAssginAdd(const MatExpr& expr, Mat& m) const;
    virtual void augAssginSubtract(const MatExpr& expr, Mat& m) const;
    virtual void augAssginMultiply(const MatExpr& expr, Mat& m) const;
    virtual void augAssginDivide(const MatExpr& expr, Mat& m) const;

    virtual void add(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void subtract(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void multiply(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void divide(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;

    virtual int type(const MatExpr& expr) const;
};

class MatExpr
{
public:
    MatExpr();
    explicit MatExpr(const Mat& m);

    MatExpr(const MatOp* _op, int _flags, const Mat& _a = Mat(), const Mat& _b = Mat(),
            const Mat& _c = Mat(), double _alpha = 1, double _beta = 1);

    operator Mat() const;

    int type() const;

    const MatOp* op;
    int flags;

    Mat a, b, c;
    double alpha, beta;
};

MatExpr operator + (const Mat& a, const Mat& b);
MatExpr operator + (const Mat& a, const MatExpr& e);
MatExpr operator + (const MatExpr& e, const Mat& b);
MatExpr operator + (const MatExpr& e1, const MatExpr& e2);
MatExpr operator + (const MatExpr& e1, const float e);
MatExpr operator + (const float e, const MatExpr& e2);
MatExpr operator + (const Mat& a, const float e);
MatExpr operator + (const float e, const Mat& b);
MatExpr operator + (const MatExpr& e1, const Scalar& s);
MatExpr operator + (const Scalar& s, const MatExpr& e2);
MatExpr operator + (const Mat& a, const Scalar& s);
MatExpr operator + (const Scalar& s, const Mat& b);

MatExpr operator - (const Mat& a);
MatExpr operator - (const MatExpr& e);
MatExpr operator - (const Mat& a, const Mat& b);
MatExpr operator - (const Mat& a, const MatExpr& e);
MatExpr operator - (const MatExpr& e, const Mat& b);
MatExpr operator - (const MatExpr& e1, const MatExpr& e2);
MatExpr operator - (const MatExpr& e1, const float e);
MatExpr operator - (const float e, const MatExpr& e2);
MatExpr operator - (const Mat& a, const float e);
MatExpr operator - (const float e, const Mat& b);
MatExpr operator - (const MatExpr& e1, const Scalar& s);
MatExpr operator - (const Scalar& s, const MatExpr& e2);
MatExpr operator - (const Mat& a, const Scalar& s);
MatExpr operator - (const Scalar& s, const Mat& b);

MatExpr operator * (const Mat& a, const Mat& b);
MatExpr operator * (const Mat& a, const MatExpr& e);
MatExpr operator * (const MatExpr& e, const Mat& b);
MatExpr operator * (const MatExpr& e1, const MatExpr& e2);
MatExpr operator * (const MatExpr& e1, const float e);
MatExpr operator * (const float e, const MatExpr& e2);
MatExpr operator * (const Mat& a, const float e);
MatExpr operator * (const float e, const Mat& b);
MatExpr operator * (const MatExpr& e1, const Scalar& s);
MatExpr operator * (const Scalar& s, const MatExpr& e2);
MatExpr operator * (const Mat& a, const Scalar& s);
MatExpr operator * (const Scalar& s, const Mat& b);

MatExpr operator / (const Mat& a, const Mat& b);
MatExpr operator / (const Mat& a, const MatExpr& e);
MatExpr operator / (const MatExpr& e, const Mat& b);
MatExpr operator / (const MatExpr& e1, const MatExpr& e2);
MatExpr operator / (const MatExpr& e1, const float e);
MatExpr operator / (const float e, const MatExpr& e2);
MatExpr operator / (const Mat& a, const float e);
MatExpr operator / (const float e, const Mat& b);
MatExpr operator / (const MatExpr& e1, const Scalar& s);
MatExpr operator / (const Scalar& s, const MatExpr& e2);
MatExpr operator / (const Mat& a, const Scalar& s);
MatExpr operator / (const Scalar& s, const Mat& b);

MatExpr operator == (const Mat& a, const Mat& b);
MatExpr operator == (const Mat& a, const Scalar& s);
MatExpr operator == (const Scalar& s, const Mat& a);
MatExpr operator == (const Mat& a, const float s);
MatExpr operator == (const float s, const Mat& a);

MatExpr operator != (const Mat& a, const Mat& b);
MatExpr operator != (const Mat& a, const Scalar& s);
MatExpr operator != (const Scalar& s, const Mat& a);
MatExpr operator != (const Mat& a, const float s);
MatExpr operator != (const float s, const Mat& a);

MatExpr operator < (const Mat& a, const Mat& b);
MatExpr operator < (const Mat& a, const Scalar& s);
MatExpr operator < (const Scalar& s, const Mat& a);
MatExpr operator < (const Mat& a, const float s);
MatExpr operator < (const float s, const Mat& a);

MatExpr operator <= (const Mat& a, const Mat& b);
MatExpr operator <= (const Mat& a, const Scalar& s);
MatExpr operator <= (const Scalar& s, const Mat& a);
MatExpr operator <= (const Mat& a, const float s);
MatExpr operator <= (const float s, const Mat& a);

MatExpr operator > (const Mat& a, const Mat& b);
MatExpr operator > (const Mat& a, const Scalar& s);
MatExpr operator > (const Scalar& s, const Mat& a);
MatExpr operator > (const Mat& a, const float s);
MatExpr operator > (const float s, const Mat& a);

MatExpr operator >= (const Mat& a, const Mat& b);
MatExpr operator >= (const Mat& a, const Scalar& s);
MatExpr operator >= (const Scalar& s, const Mat& a);
MatExpr operator >= (const Mat& a, const float s);
MatExpr operator >= (const float s, const Mat& a);

// TODO support bitwise/min/max/abs operators for MatExpr.


// compute shape.
size_t total(const Mat& m);
size_t total(const Mat& m, int startDim, int endDim = -1);
size_t total(const MatShape shape);
size_t total(const MatShape shape, int startDim, int endDim = -1);

// for fast gemm
struct GemmPackedB
{
    MatShape shape;
    MatShape strides;
    int type = -1;
    int k = 0;
    int n = 0;
    size_t packed_step = 0;
    std::vector<float> packed_fp32;
    std::vector<hfloat> packed_fp16;

    bool empty() const;
};

GemmPackedB gemm_pack_b(const Mat& b, bool transB = false);
Mat gemm(const Mat& a, const Mat& b, bool transA = false, bool transB = false);
Mat gemm(const Mat& a, const GemmPackedB& packed_b, bool transA = false);
Mat gemm(const Mat& a, const Mat& b, const Mat& b_scales, bool transA = false, bool transB = false);

// TODO Mat inv, and other type mat operator.
}

#include "./mat.inl.h"

#if defined(CVH_LITE)
#include "./mat_lite_impl.h"
#endif

#endif //CVH_MAT_H
