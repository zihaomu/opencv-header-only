//
// Created by mzh on 2024/1/31.
//

#include "cvh/core/mat.h"
#include "cvh/core/basic_op.h"
#include "cvh/core/system.h"
#include "cvh/core/utils.h"

namespace cvh
{

// The following code has taken from https://github.com/opencv/opencv/modules/dnn/test/npy_blob.cpp
static std::string getType(const std::string& header)
{
    std::string field = "'descr':";
    int idx = header.find(field);
    M_Assert(idx != -1);

    int from = header.find('\'', idx + field.size()) + 1;
    int to = header.find('\'', from);
    return header.substr(from, to - from);
}

static std::string getFortranOrder(const std::string& header)
{
    std::string field = "'fortran_order':";
    int idx = header.find(field);
    M_Assert(idx != -1);

    int from = header.find_last_of(' ', idx + field.size()) + 1;
    int to = header.find(',', from);
    return header.substr(from, to - from);
}

static std::vector<int> getShape(const std::string& header)
{
    std::string field = "'shape':";
    size_t idx = header.find(field);
    if (idx == std::string::npos) return {1};

    size_t from = header.find('(', idx) + 1;
    size_t to   = header.find(')', from);
    std::string shapeStr = header.substr(from, to - from);

    std::vector<int> shape;
    std::istringstream ss(shapeStr);
    std::string token;
    while (std::getline(ss, token, ','))  // 用逗号分隔
    {
        // 去掉空格
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty())
            shape.push_back(std::stoi(token));
    }
    if (shape.empty()) shape.push_back(1);
    return shape;
}

// TODO current only support the fp32 data
Mat readMatFromNpy(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::binary);
    if (!ifs.is_open())
    {
        M_Error_(NULL, ("\n Can't open file: %s", path.c_str()));
        M_Assert(ifs.is_open());
        return Mat();
    }

    std::string magic(6, '*');
    ifs.read(&magic[0], magic.size());
    M_Assert(magic == "\x93NUMPY");

    ifs.ignore(1);  // Skip major version byte.
    ifs.ignore(1);  // Skip minor version byte.

    unsigned short headerSize;
    ifs.read((char *) &headerSize, sizeof(headerSize));

    std::string header(headerSize, '*');
    ifs.read(&header[0], header.size());

    // Extract data type.
    M_Assert(getType(header) == "<f4" || getType(header) == "<i4");
    auto data_type = getType(header) == "<f4" ? CV_32F : CV_32S;
    M_Assert(getFortranOrder(header) == "False");
    std::vector<int> shape = getShape(header);

    size_t typeSize = CV_ELEM_SIZE(data_type);
    Mat blob(shape, data_type);
    ifs.read((char *) blob.data, blob.total() * typeSize);
    M_Assert((size_t) ifs.gcount() == blob.total() * typeSize);

    return blob;
}

}