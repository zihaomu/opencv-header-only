#include <cvh/cvh.h>
#include <cvh.h>

int main()
{
    cvh::MatShape shape = {2, 3, 4};
    if (cvh::total(shape) != 24)
    {
        return 1;
    }

    int dims_and_sizes[] = {3, 2, 3, 4};
    cvh::MatSize mat_size(dims_and_sizes);
    if (mat_size.dims() != 3 || mat_size[0] != 2 || mat_size[1] != 3 || mat_size[2] != 4)
    {
        return 2;
    }

    cvh::Mat m({2, 2}, CV_8UC3);
    m.setTo(cvh::Scalar(1, 2, 3, 0));
    if (m.empty() || m.channels() != 3 || m.step(0) != 6)
    {
        return 3;
    }

    cvh::Mat c = m.clone();
    if (c.empty() || c.data == m.data || c.type() != m.type())
    {
        return 4;
    }

    const uchar* p = c.pixelPtr(1, 1);
    if (p[0] != 1 || p[1] != 2 || p[2] != 3)
    {
        return 5;
    }

    return 0;
}
