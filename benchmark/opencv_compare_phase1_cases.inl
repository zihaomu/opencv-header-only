double P1_BENCH_FUNCTION(Phase1OpId op,
                         int rows,
                         int cols,
                         int warmup,
                         int iters,
                         int repeats,
                         std::uint32_t seed)
{
    namespace api = P1_NAMESPACE;
    using Mat = P1_MAT;

    switch (op)
    {
        case Phase1OpId::Absdiff:
        case Phase1OpId::BitwiseAnd:
        case Phase1OpId::BitwiseOr:
        case Phase1OpId::BitwiseXor:
        case Phase1OpId::Min:
        case Phase1OpId::Max:
        {
            Mat first = p1_make_mat(rows, cols, CV_8UC3);
            Mat second = p1_make_mat(rows, cols, CV_8UC3);
            Mat dst;
            p1_fill_u8(first, seed);
            p1_fill_u8(second, seed + 17u);
            const auto run = [&]() {
                switch (op)
                {
                    case Phase1OpId::Absdiff:
                        api::absdiff(first, second, dst);
                        break;
                    case Phase1OpId::BitwiseAnd:
                        api::bitwise_and(first, second, dst);
                        break;
                    case Phase1OpId::BitwiseOr:
                        api::bitwise_or(first, second, dst);
                        break;
                    case Phase1OpId::BitwiseXor:
                        api::bitwise_xor(first, second, dst);
                        break;
                    case Phase1OpId::Min:
                        api::min(first, second, dst);
                        break;
                    case Phase1OpId::Max:
                        api::max(first, second, dst);
                        break;
                    default:
                        break;
                }
            };
            return measure_ms(
                run,
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::BitwiseNot:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC3);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() { api::bitwise_not(src, dst); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::InRange:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC3);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    api::inRange(
                        src,
                        api::Scalar(32.0, 48.0, 64.0),
                        api::Scalar(220.0, 210.0, 200.0),
                        dst);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::ScaleAdd:
        {
            Mat first = p1_make_mat(rows, cols, CV_32FC3);
            Mat second = p1_make_mat(rows, cols, CV_32FC3);
            Mat dst;
            p1_fill_f32(first, seed);
            p1_fill_f32(second, seed + 17u);
            return measure_ms(
                [&]() { api::scaleAdd(first, 0.75, second, dst); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::ConvertScaleAbs:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC3);
            Mat dst;
            p1_fill_f32(src, seed);
            return measure_ms(
                [&]() { api::convertScaleAbs(src, dst, 1.5, 2.0); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::ConvertFp16:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC1);
            Mat dst;
            p1_fill_f32(src, seed);
            return measure_ms(
                [&]() { api::convertFp16(src, dst); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Sqrt:
        case Phase1OpId::Pow:
        case Phase1OpId::Exp:
        case Phase1OpId::Log:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC1);
            Mat dst;
            p1_fill_f32(src, seed);
            p1_make_positive(src);
            const auto run = [&]() {
                switch (op)
                {
                    case Phase1OpId::Sqrt:
                        api::sqrt(src, dst);
                        break;
                    case Phase1OpId::Pow:
                        api::pow(src, 1.75, dst);
                        break;
                    case Phase1OpId::Exp:
                        api::exp(src, dst);
                        break;
                    case Phase1OpId::Log:
                        api::log(src, dst);
                        break;
                    default:
                        break;
                }
            };
            return measure_ms(
                run,
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::CheckRange:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC1);
            p1_fill_f32(src, seed);
            double result = 0.0;
            return measure_ms(
                [&]() {
                    result = api::checkRange(
                                 src, true, nullptr, -16.0, 16.0)
                                 ? 1.0
                                 : 0.0;
                },
                [&]() { return result; },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::PatchNaNs:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC1);
            p1_fill_f32(src, seed);
            return measure_ms(
                [&]() {
                    src.template at<float>(0, 0) =
                        std::numeric_limits<float>::quiet_NaN();
                    api::patchNaNs(src, 0.0);
                },
                [&]() { return p1_checksum(src); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Norm:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC1);
            p1_fill_f32(src, seed);
            double result = 0.0;
            return measure_ms(
                [&]() { result = api::norm(src, api::NORM_L2); },
                [&]() { return result; },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Sum:
        case Phase1OpId::Mean:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC3);
            p1_fill_f32(src, seed);
            api::Scalar result;
            return measure_ms(
                [&]() {
                    result = op == Phase1OpId::Sum
                                 ? api::sum(src)
                                 : api::mean(src);
                },
                [&]() { return result[0] + result[1] + result[2]; },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::MeanStdDev:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC3);
            p1_fill_f32(src, seed);
            api::Scalar mean_value;
            api::Scalar stddev_value;
            return measure_ms(
                [&]() {
                    api::meanStdDev(src, mean_value, stddev_value);
                },
                [&]() { return mean_value[0] + stddev_value[0]; },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::CountNonZero:
        case Phase1OpId::HasNonZero:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            p1_fill_u8(src, seed);
            double result = 0.0;
            return measure_ms(
                [&]() {
                    result =
                        op == Phase1OpId::CountNonZero
                            ? static_cast<double>(api::countNonZero(src))
                            : (api::hasNonZero(src) ? 1.0 : 0.0);
                },
                [&]() { return result; },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::FindNonZero:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            p1_fill_u8(src, seed);
            std::vector<P1_POINT_TYPE> points;
            return measure_ms(
                [&]() { api::findNonZero(src, points); },
                [&]() { return static_cast<double>(points.size()); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::MinMaxIdx:
        case Phase1OpId::MinMaxLoc:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC1);
            p1_fill_f32(src, seed);
            double minimum = 0.0;
            double maximum = 0.0;
            int min_index[2] = {};
            int max_index[2] = {};
            P1_POINT_TYPE min_point;
            P1_POINT_TYPE max_point;
            const auto run = [&]() {
                if (op == Phase1OpId::MinMaxIdx)
                {
                    api::minMaxIdx(
                        src,
                        &minimum,
                        &maximum,
                        min_index,
                        max_index);
                }
                else
                {
                    api::minMaxLoc(
                        src,
                        &minimum,
                        &maximum,
                        &min_point,
                        &max_point);
                }
            };
            return measure_ms(
                run,
                [&]() { return minimum + maximum; },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Reduce:
        case Phase1OpId::ReduceArgMax:
        case Phase1OpId::ReduceArgMin:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC1);
            Mat dst;
            p1_fill_f32(src, seed);
            const auto run = [&]() {
                if (op == Phase1OpId::Reduce)
                {
                    api::reduce(src, dst, 0, api::REDUCE_SUM, CV_32F);
                }
                else if (op == Phase1OpId::ReduceArgMax)
                {
                    api::reduceArgMax(src, dst, 0);
                }
                else
                {
                    api::reduceArgMin(src, dst, 0);
                }
            };
            return measure_ms(
                run,
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Normalize:
        {
            Mat src = p1_make_mat(rows, cols, CV_32FC1);
            Mat dst;
            p1_fill_f32(src, seed);
            return measure_ms(
                [&]() {
                    api::normalize(
                        src, dst, 1.0, 0.0, api::NORM_L2, CV_32F);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::BorderInterpolate:
        {
            double result = 0.0;
            return measure_ms(
                [&]() {
                    int accumulator = 0;
                    for (int index = -2048; index < 2048; ++index)
                    {
                        accumulator += api::borderInterpolate(
                            index, cols, api::BORDER_REFLECT_101);
                    }
                    result = accumulator;
                },
                [&]() { return result; },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::CopyTo:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC3);
            Mat mask = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            p1_fill_u8(mask, seed + 23u);
            return measure_ms(
                [&]() { api::copyTo(src, dst, mask); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::ExtractChannel:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC3);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() { api::extractChannel(src, dst, 1); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::InsertChannel:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst = p1_make_mat(rows, cols, CV_8UC3);
            p1_fill_u8(src, seed);
            p1_fill_u8(dst, seed + 23u);
            return measure_ms(
                [&]() { api::insertChannel(src, dst, 1); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::MixChannels:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC3);
            Mat dst = p1_make_mat(rows, cols, CV_8UC3);
            p1_fill_u8(src, seed);
            const int mapping[] = {0, 2, 1, 1, 2, 0};
            return measure_ms(
                [&]() {
                    api::mixChannels(&src, 1, &dst, 1, mapping, 3);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Flip:
        case Phase1OpId::FlipND:
        case Phase1OpId::Rotate:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC3);
            Mat dst;
            p1_fill_u8(src, seed);
            const auto run = [&]() {
                if (op == Phase1OpId::Flip)
                {
                    api::flip(src, dst, 1);
                }
                else if (op == Phase1OpId::FlipND)
                {
                    api::flipND(src, dst, 1);
                }
                else
                {
                    api::rotate(src, dst, api::ROTATE_90_CLOCKWISE);
                }
            };
            return measure_ms(
                run,
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Repeat:
        {
            Mat src = p1_make_mat(rows / 2, cols / 2, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() { api::repeat(src, 2, 2, dst); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Hconcat:
        case Phase1OpId::Vconcat:
        {
            Mat first = p1_make_mat(
                op == Phase1OpId::Hconcat ? rows : rows / 2,
                op == Phase1OpId::Hconcat ? cols / 2 : cols,
                CV_8UC1);
            Mat second = first.clone();
            Mat dst;
            p1_fill_u8(first, seed);
            p1_fill_u8(second, seed + 23u);
            return measure_ms(
                [&]() {
                    if (op == Phase1OpId::Hconcat)
                    {
                        api::hconcat(first, second, dst);
                    }
                    else
                    {
                        api::vconcat(first, second, dst);
                    }
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Broadcast:
        {
            Mat src = p1_make_mat(1, cols, CV_8UC1);
            Mat shape = p1_make_mat(2, 1, CV_32SC1);
            Mat dst;
            p1_fill_u8(src, seed);
            shape.template at<int>(0, 0) = rows;
            shape.template at<int>(1, 0) = cols;
            return measure_ms(
                [&]() { api::broadcast(src, shape, dst); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Swap:
        {
            Mat first = p1_make_mat(rows, cols, CV_8UC1);
            Mat second = p1_make_mat(rows, cols, CV_8UC1);
            p1_fill_u8(first, seed);
            p1_fill_u8(second, seed + 23u);
            return measure_ms(
                [&]() { api::swap(first, second); },
                [&]() {
                    return p1_checksum(first) + p1_checksum(second);
                },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::GetStructuringElement:
        {
            Mat dst;
            return measure_ms(
                [&]() {
                    dst = api::getStructuringElement(
                        api::MORPH_ELLIPSE,
                        P1_SIZE_TYPE(7, 7));
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::GetGaussianKernel:
        {
            Mat dst;
            return measure_ms(
                [&]() { dst = api::getGaussianKernel(15, 2.5, CV_32F); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::GetDerivKernels:
        {
            Mat first;
            Mat second;
            return measure_ms(
                [&]() {
                    api::getDerivKernels(
                        first, second, 1, 0, 5, true, CV_32F);
                },
                [&]() {
                    return p1_checksum(first) + p1_checksum(second);
                },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::GetGaborKernel:
        {
            Mat dst;
            return measure_ms(
                [&]() {
                    dst = api::getGaborKernel(
                        P1_SIZE_TYPE(15, 15),
                        3.0,
                        0.7,
                        5.0,
                        0.8,
                        0.25,
                        CV_32F);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::CreateHanningWindow:
        {
            Mat dst;
            return measure_ms(
                [&]() {
                    api::createHanningWindow(
                        dst,
                        P1_SIZE_TYPE(64, 64),
                        CV_32F);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Integral:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() { api::integral(src, dst, CV_32S); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Scharr:
        case Phase1OpId::Laplacian:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            const auto run = [&]() {
                if (op == Phase1OpId::Scharr)
                {
                    api::Scharr(
                        src,
                        dst,
                        CV_32F,
                        1,
                        0,
                        1.0,
                        0.0,
                        api::BORDER_REFLECT_101);
                }
                else
                {
                    api::Laplacian(
                        src,
                        dst,
                        CV_32F,
                        3,
                        1.0,
                        0.0,
                        api::BORDER_REFLECT_101);
                }
            };
            return measure_ms(
                run,
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::SpatialGradient:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dx;
            Mat dy;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    api::spatialGradient(
                        src, dx, dy, 3, api::BORDER_REFLECT_101);
                },
                [&]() { return p1_checksum(dx) + p1_checksum(dy); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::SqrBoxFilter:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    api::sqrBoxFilter(
                        src,
                        dst,
                        CV_32F,
                        P1_SIZE_TYPE(3, 3),
                        P1_POINT_TYPE(-1, -1),
                        true,
                        api::BORDER_REFLECT_101);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::MedianBlur:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() { api::medianBlur(src, dst, 5); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::BilateralFilter:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    api::bilateralFilter(
                        src,
                        dst,
                        5,
                        25.0,
                        3.0,
                        api::BORDER_REFLECT_101);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::StackBlur:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    api::stackBlur(
                        src, dst, P1_SIZE_TYPE(5, 5));
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::AdaptiveThreshold:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    api::adaptiveThreshold(
                        src,
                        dst,
                        255.0,
                        api::ADAPTIVE_THRESH_MEAN_C,
                        api::THRESH_BINARY,
                        11,
                        2.0);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::ThresholdWithMask:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat mask = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst = src.clone();
            p1_fill_u8(src, seed);
            p1_fill_u8(mask, seed + 23u);
            return measure_ms(
                [&]() {
                    api::thresholdWithMask(
                        src,
                        dst,
                        mask,
                        127.0,
                        255.0,
                        api::THRESH_BINARY);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::EqualizeHist:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() { api::equalizeHist(src, dst); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::ApplyColorMap:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    api::applyColorMap(src, dst, api::COLORMAP_JET);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Accumulate:
        case Phase1OpId::AccumulateSquare:
        case Phase1OpId::AccumulateWeighted:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst = p1_make_mat(rows, cols, CV_32FC1);
            p1_fill_u8(src, seed);
            dst.setTo(api::Scalar::all(0.0));
            const auto run = [&]() {
                if (op == Phase1OpId::Accumulate)
                {
                    api::accumulate(src, dst);
                }
                else if (op == Phase1OpId::AccumulateSquare)
                {
                    api::accumulateSquare(src, dst);
                }
                else
                {
                    api::accumulateWeighted(src, dst, 0.1);
                }
            };
            return measure_ms(
                run,
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::AccumulateProduct:
        {
            Mat first = p1_make_mat(rows, cols, CV_8UC1);
            Mat second = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst = p1_make_mat(rows, cols, CV_32FC1);
            p1_fill_u8(first, seed);
            p1_fill_u8(second, seed + 23u);
            dst.setTo(api::Scalar::all(0.0));
            return measure_ms(
                [&]() { api::accumulateProduct(first, second, dst); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::BlendLinear:
        {
            Mat first = p1_make_mat(rows, cols, CV_8UC3);
            Mat second = p1_make_mat(rows, cols, CV_8UC3);
            Mat weight1 = p1_make_mat(rows, cols, CV_32FC1);
            Mat weight2 = p1_make_mat(rows, cols, CV_32FC1);
            Mat dst;
            p1_fill_u8(first, seed);
            p1_fill_u8(second, seed + 23u);
            weight1.setTo(api::Scalar::all(0.35));
            weight2.setTo(api::Scalar::all(0.65));
            return measure_ms(
                [&]() {
                    api::blendLinear(
                        first, second, weight1, weight2, dst);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::PyrDown:
        case Phase1OpId::PyrUp:
        {
            const int source_rows =
                op == Phase1OpId::PyrDown ? rows : rows / 2;
            const int source_cols =
                op == Phase1OpId::PyrDown ? cols : cols / 2;
            Mat src = p1_make_mat(source_rows, source_cols, CV_8UC3);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    if (op == Phase1OpId::PyrDown)
                    {
                        api::pyrDown(src, dst);
                    }
                    else
                    {
                        api::pyrUp(src, dst);
                    }
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::BuildPyramid:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            std::vector<Mat> pyramid;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() { api::buildPyramid(src, pyramid, 3); },
                [&]() {
                    return pyramid.empty()
                               ? 0.0
                               : p1_checksum(pyramid.back());
                },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::CvtColorTwoPlane:
        {
            const int even_rows = rows & ~1;
            const int even_cols = cols & ~1;
            Mat y = p1_make_mat(even_rows, even_cols, CV_8UC1);
            Mat uv =
                p1_make_mat(even_rows / 2, even_cols / 2, CV_8UC2);
            Mat dst;
            p1_fill_u8(y, seed);
            p1_fill_u8(uv, seed + 23u);
            return measure_ms(
                [&]() {
                    api::cvtColorTwoPlane(
                        y, uv, dst, api::COLOR_YUV2BGR_NV12);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Demosaicing:
        {
            Mat src = p1_make_mat(rows, cols, CV_8UC1);
            Mat dst;
            p1_fill_u8(src, seed);
            return measure_ms(
                [&]() {
                    api::demosaicing(
                        src, dst, api::COLOR_BayerBG2BGR);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::ConvertMaps:
        {
            Mat map_x = p1_make_mat(rows, cols, CV_32FC1);
            Mat map_y = p1_make_mat(rows, cols, CV_32FC1);
            Mat first;
            Mat second;
            p1_fill_identity_maps(map_x, map_y);
            return measure_ms(
                [&]() {
                    api::convertMaps(
                        map_x,
                        map_y,
                        first,
                        second,
                        CV_16SC2,
                        false);
                },
                [&]() {
                    return p1_checksum(first) + p1_checksum(second);
                },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::GetAffineTransform:
        {
            const P1_POINT2F_TYPE src[] = {
                {0.0f, 0.0f},
                {100.0f, 0.0f},
                {0.0f, 80.0f},
            };
            const P1_POINT2F_TYPE dst_points[] = {
                {5.0f, 7.0f},
                {102.0f, 3.0f},
                {8.0f, 86.0f},
            };
            Mat dst;
            return measure_ms(
                [&]() { dst = api::getAffineTransform(src, dst_points); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::GetPerspectiveTransform:
        {
            const P1_POINT2F_TYPE src[] = {
                {0.0f, 0.0f},
                {100.0f, 0.0f},
                {100.0f, 80.0f},
                {0.0f, 80.0f},
            };
            const P1_POINT2F_TYPE dst_points[] = {
                {3.0f, 6.0f},
                {98.0f, 2.0f},
                {104.0f, 84.0f},
                {6.0f, 78.0f},
            };
            Mat dst;
            return measure_ms(
                [&]() {
                    dst = api::getPerspectiveTransform(
                        src, dst_points, api::DECOMP_LU);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::GetRotationMatrix2D:
        {
            Mat dst;
            return measure_ms(
                [&]() {
                    dst = api::getRotationMatrix2D(
                        P1_POINT2F_TYPE(31.5f, 24.5f),
                        17.0,
                        0.85);
                },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::GetRotationMatrix2DUnderscore:
        {
            double result = 0.0;
            volatile double angle = 17.0;
            return measure_ms(
                [&]() {
                    result = p1_rotation_value(
                        api::getRotationMatrix2D_(
                            P1_POINT2F_TYPE(31.5f, 24.5f),
                            angle,
                            0.85));
                },
                [&]() { return result; },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::InvertAffineTransform:
        {
            Mat src = api::getRotationMatrix2D(
                P1_POINT2F_TYPE(31.5f, 24.5f),
                17.0,
                0.85);
            Mat dst;
            return measure_ms(
                [&]() { api::invertAffineTransform(src, dst); },
                [&]() { return p1_checksum(dst); },
                warmup,
                iters,
                repeats);
        }
        case Phase1OpId::Remap:
        case Phase1OpId::WarpPerspective:
        case Phase1OpId::GetRectSubPix:
            return -1.0;
    }

    return -1.0;
}
