/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scatterPlugin.h"
#include "cuda_fp16.h"
#include <thread>
#include <chrono>

using namespace nvinfer1;
using namespace plugin;


// Device code
__global__ void ScatterNDKernel(int *indices,
    int index_rank,
    const int *data_shape,
    int *updates,
    int vec_size,
    int tar_num,
    int* output
) 
{
    int tar_idx = blockIdx.x;

    // calculate the index part idx based on indice input
    int indices_base =  tar_idx * index_rank;
    int index_idx = 0;
    for (int i=0; i<index_rank; ++i){
	    //printf("data_shape[%d]: %d\n", i, data_shape[i]);
        index_idx = index_idx * data_shape[i] + indices[indices_base+i];
    }

    // calculate the source idx of params
    int data_idx_base = index_idx * vec_size;

    // calculate the target idx of updates
    int updates_idx_base = tar_idx * vec_size;

    int vec_idx, data_idx, updates_idx;
    for (int i=0; i<tar_num; ++i){
        vec_idx = threadIdx.x + blockDim.x * i;
        if (vec_idx>=vec_size){
            break;
        }
        data_idx = data_idx_base + vec_idx;
        updates_idx = updates_idx_base + vec_idx;

	    atomicExch(output+data_idx, updates[updates_idx]);
    }
}

PluginFieldCollection ScatterNDCreator::mFC{};
std::vector<PluginField> ScatterNDCreator::mPluginAttributes;

int ScatterND::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    dim3 dimBlock;
    dim3 dimGrid;


    int index_rank = inputDesc[1].dims.d[inputDesc[1].dims.nbDims-1];
    int tar_size = multiplyArray(inputDesc[1].dims.d, inputDesc[1].dims.nbDims-1);
    int vec_size = multiplyArray(inputDesc[0].dims.d+index_rank, inputDesc[0].dims.nbDims-index_rank);

    dimBlock.x = vec_size >= 1024 ? 1024 : vec_size;
    dimGrid.x = tar_size;

    cudaMemcpyAsync(outputs[0], inputs[0], multiplyArray(inputDesc[0].dims.d, inputDesc[0].dims.nbDims)*sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    // copy params shape to device to calculate gather position
    cudaMemcpyAsync((int*)workspace, inputDesc[0].dims.d, inputDesc[0].dims.nbDims * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // invoke kernel
    ScatterNDKernel<<<dimGrid, dimBlock, 0, stream>>>((int*)inputs[1],
                                                index_rank,
                                                (int*)workspace,
                                                (int*)inputs[2],
                                                vec_size,
                                                ceilf(vec_size/float(dimBlock.x)),
                                                (int*)outputs[0]);

    return 0;
}

REGISTER_TENSORRT_PLUGIN(ScatterNDCreator);