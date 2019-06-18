/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

// Sets output[0] to shape [batch_dim,height,width,channel_dim], where
// height and width come from the size_tensor.
Status SetOutputToSizedImage(InferenceContext *c, DimensionHandle batch_dim,
                             int size_input_idx, DimensionHandle channel_dim) {
  // Verify shape of size input.
  ShapeHandle size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(size_input_idx), 1, &size));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 2, &unused));

  // Get size values from the size tensor.
  const Tensor *size_tensor = c->input_tensor(size_input_idx);
  DimensionHandle width;
  DimensionHandle height;
  if (size_tensor == nullptr) {
    width = c->UnknownDim();
    height = c->UnknownDim();
  } else {
    // TODO(petewarden) - Remove once we have constant evaluation in C++ only.
    if (size_tensor->dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
          "but got ",
          DataTypeString(size_tensor->dtype()), " for input #", size_input_idx,
          " in ", c->DebugString());
    }
    auto vec = size_tensor->vec<int32>();
    height = c->MakeDim(vec(0));
    width = c->MakeDim(vec(1));
  }
  c->set_output(0, c->MakeShape({batch_dim, height, width, channel_dim}));
  return Status::OK();
}

// TODO(qyu): Move this to core/framework/common_shape_fns.h
Status ResizeShapeFn(InferenceContext *c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
  return SetOutputToSizedImage(c, c->Dim(input, 0), 2 /* size_input_idx */,
                               c->Dim(input, 3));
}

static const char EuclideanDistanceTransformDoc[] = R"doc(
Applies the euclidean distance transform to each of the images.

Input `image` is a `Tensor` in NHWC format (batch, rows, columns,
and channels). `image` must be a binary image with a single channel,
and of type `uint8`.

transformed_images: 4D `Tensor`, image(s) in NHWC format of type `tf.float32`
generated by applying the euclidean distance transform to `images`.
applying
)doc";

static const char kImageProjectiveTransformDoc[] = R"doc(
Applies the given transform to each of the images.

Input `image` is a `Tensor` in NHWC format (where the axes are image in batch,
rows, columns, and channels. Input `transforms` is a num_images x 8 or 1 x 8
matrix, where each row corresponds to a 3 x 3 projective transformation matrix,
with the last entry assumed to be 1. If there is one row, the same
transformation will be applied to all images.

If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
the *output* point `(x, y)` to a transformed *input* point
`(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
`k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
image, the output pixel is set to 0.

images: 4D `Tensor`, input image(s) in NHWC format.
transforms: 2D `Tensor`, projective transform(s) to apply to the image(s).

transformed_images: 4D `Tensor`, image(s) in NHWC format, generated by applying
the `transforms` to the `images`. Satisfies the description above.
)doc";

}  // namespace

REGISTER_OP("EuclideanDistanceTransform")
    .Input("images: dtype")
    .Attr("dtype: {float16, float32, float64}")
    .Output("transformed_images: dtype")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(EuclideanDistanceTransformDoc);

// V2 op supports output_shape.
REGISTER_OP("ImageProjectiveTransformV2")
    .Input("images: dtype")
    .Input("transforms: float32")
    .Input("output_shape: int32")
    .Attr("dtype: {uint8, int32, int64, float16, float32, float64}")
    .Attr("interpolation: string")
    .Output("transformed_images: dtype")
    .SetShapeFn(ResizeShapeFn)
    .Doc(kImageProjectiveTransformDoc);
}  // namespace tensorflow
