#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#include "tensorflow_addons/custom_ops/image/cc/kernels/distance_transform/distance_transform.hpp"
#include "tensorflow_addons/custom_ops/image/cc/kernels/distance_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>

using namespace dope;

namespace tensorflow
{

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class DistanceTransform : public OpKernel
{
  public:
    explicit DistanceTransform(OpKernelConstruction *ctx) : OpKernel(ctx)
    {
    }

    void Compute(OpKernelContext *ctx) override
    {
        const Tensor &input_t = ctx->input(0);
        const Tensor &cutoff_t = ctx->input(1);
        OP_REQUIRES(ctx, cutoff_t.shape().dims() != 1 || cutoff_t.shape().dim_size(0) != 1,
                    errors::InvalidArgument("cutoff must a scalar"));

        Tensor *distance_t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_t.shape(), &distance_t));

        //auto input = input_t.tensor<T>();
        //auto thresholds = cutoff_t.tensor<T>();
        //auto distance = distance_t->flat<float>();

        Index3 size3D = {64, 64, 64};
        Grid<float, 3> f3D(size3D);
        for (SizeType i = 0; i < size3D[0]; ++i)
            for (SizeType j = 0; j < size3D[1]; ++j)
                for (std::size_t k = 0; k < size3D[2]; ++k)
                    f3D[i][j][k] = std::numeric_limits<float>::max();
        f3D[10][10][10] = 0.0f;
        f3D[41][41][41] = 0.0f;

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        dt::DistanceTransform::distanceTransformL2(f3D, f3D, false, 1);
        std::cout << std::endl
                  << size3D[0] << 'x' << size3D[1] << 'x' << size3D[2] << " distance function computed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << " ms." << std::endl;

        std::cout << "build volume" << std::endl;

        std::fstream fs;
        fs.open("./out.vtk", std::ios_base::out);
        fs << "# vtk DataFile Version 3.0" << std::endl;
        fs << "distance" << std::endl;
        fs << "ASCII" << std::endl;
        fs << "DATASET STRUCTURED_POINTS" << std::endl;
        fs << "DIMENSIONS " << size3D[0] << " " << size3D[1] << " " << size3D[2] << std::endl;
        fs << "ORIGIN 0 0 0" << std::endl;
        fs << "SPACING 1 1 1" << std::endl;
        fs << "POINT_DATA " << size3D.prod() << std::endl;
        fs << "SCALARS distance float 1" << std::endl;
        fs << "LOOKUP_TABLE default" << std::endl;
        for (SizeType i = 0; i < size3D[0]; ++i)
            for (SizeType j = 0; j < size3D[1]; ++j)
                for (std::size_t k = 0; k < size3D[2]; ++k)
                    fs << f3D[i][j][k] << std::endl;
        fs.close();
    }
};

#define REGISTER(TYPE)                                          \
    REGISTER_KERNEL_BUILDER(Name("DistanceTransform")           \
                                .Device(DEVICE_CPU)             \
                                .TypeConstraint<TYPE>("dtype"), \
                            DistanceTransform<CPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

} // end namespace tensorflow
