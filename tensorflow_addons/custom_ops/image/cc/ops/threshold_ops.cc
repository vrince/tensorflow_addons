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

namespace tensorflow
{

namespace
{

static const char ThresholdImageDoc[] = R"doc(
Applies the threholds to the input.

input: a `Tensor` any `dtype`.
threshold: 1D `Tensor` (max size `255`) threholds to apply to the input.

output: a `uint8` `Tensor` same size as `input`.
)doc";

} // namespace

// V2 op supports output_shape.
REGISTER_OP("ImageThreshold")
    .Input("input: dtype")
    .Input("threshold: float32")
    .Attr("dtype: {uint8, int32, int64, float16, float32, float64}")
    .Output("output: uint8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    })
    .Doc(ThresholdImageDoc);
} // namespace tensorflow
