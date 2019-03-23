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

template <typename Device, typename T, int D>
class DistanceTransform : public OpKernel
{
  public:
    explicit DistanceTransform(OpKernelConstruction *ctx) : OpKernel(ctx)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("threads", &threads_));
        OP_REQUIRES(ctx, threads_ >= 0, errors::InvalidArgument("Need threads >= 0"));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("squared", &squared_));
    }

    void Compute(OpKernelContext *ctx) override
    {
        const Tensor &input_t = ctx->input(0);
        OP_REQUIRES(ctx, input_t.shape().dims() == D,
                    errors::InvalidArgument("input shape dimension must be ", D,
                                            " get ", input_t.shape().dims()));

        const Tensor &cutoff_t = ctx->input(1);
        OP_REQUIRES(ctx, cutoff_t.shape().dims() == 0,
                    errors::InvalidArgument("cutoff must a scalar"));

        Tensor *distance_t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_t.shape(), &distance_t));

        auto cutoff = cutoff_t.flat<T>()(0);
        auto input_flat = input_t.flat<T>();
        auto distance_flat = distance_t->flat<float>();
        for (int i = 0; i < input_flat.size(); i++)
            distance_flat(i) = input_flat(i) <= cutoff ? 0 : std::numeric_limits<float>::max();

        Index<D> size;
        for (int i = 0; i < distance_t->shape().dims(); i++)
            size[i] = distance_t->dim_size(1);

        Grid<float, D> f(static_cast<float *>(distance_flat.data()), 0, size);
        dt::DistanceTransform::distanceTransformL2(f, f, squared_, threads_);
    }

  private:
    bool squared_;
    int threads_;
};

#define REGISTER_2D(TYPE)                                                                                 \
    REGISTER_KERNEL_BUILDER(Name("DistanceTransform2d").Device(DEVICE_CPU).TypeConstraint<TYPE>("dtype"), \
                            DistanceTransform<CPUDevice, TYPE, 2>)

#define REGISTER_3D(TYPE)                                                                                 \
    REGISTER_KERNEL_BUILDER(Name("DistanceTransform3d").Device(DEVICE_CPU).TypeConstraint<TYPE>("dtype"), \
                            DistanceTransform<CPUDevice, TYPE, 3>)

TF_CALL_uint8(REGISTER_2D);
TF_CALL_int32(REGISTER_2D);
TF_CALL_int64(REGISTER_2D);
TF_CALL_half(REGISTER_2D);
TF_CALL_float(REGISTER_2D);
TF_CALL_double(REGISTER_2D);

TF_CALL_uint8(REGISTER_3D);
TF_CALL_int32(REGISTER_3D);
TF_CALL_int64(REGISTER_3D);
TF_CALL_half(REGISTER_3D);
TF_CALL_float(REGISTER_3D);
TF_CALL_double(REGISTER_3D);

#undef REGISTER

} // end namespace tensorflow
