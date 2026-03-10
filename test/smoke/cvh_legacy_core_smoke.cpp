#include "cvh.h"

int main()
{
    minfer::Mat input({1, 4}, CV_32F);
    input.setTo(1.0f);
    minfer::Mat output = input.clone();
    return (output.total() == input.total()) ? 0 : 1;
}
