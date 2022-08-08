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

#include "NvInfer.h"
#include <iostream>
#include <cstring>
#include <assert.h>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class ScatterND: public IPluginV2DynamicExt {
public:
    virtual size_t getSerializationSize() const noexcept override {
        return 0;
    }
    virtual void serialize(void *buffer) const noexcept override {}

    nvinfer1::IPluginV2DynamicExt * clone() const noexcept override {
        return new ScatterND();
    }

    int getNbOutputs() const noexcept override {
        return 1;
    }

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        for (int i=0; i<inputs[0].nbDims; ++i)
            assert(inputs[0].d[i]->isConstant()==true); //data shape tensor must be constant to determine output dimensions

        return inputs[0];
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override
    {
	    return inputs[0].dims.nbDims * sizeof(int32_t);
    }

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int initialize() noexcept override {return 0;}
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* szNamespace) noexcept override {mNamespace = szNamespace;}
    const char* getPluginNamespace() const noexcept override {return mNamespace.c_str();}
    const char* getPluginType() const noexcept override {return "ScatterND";}
    const char* getPluginVersion() const noexcept override {return "1";}
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override
    {
        assert(in  && nbInputs  == 3); 
        assert(out && nbOutputs == 1);
        assert(in[0].desc.type == out[0].desc.type);  

        assert( in[0].desc.format == TensorFormat::kLINEAR); //data
        assert( in[1].desc.format == TensorFormat::kLINEAR); //indices
        assert( in[2].desc.format == TensorFormat::kLINEAR); //update
        assert(out[0].desc.format == TensorFormat::kLINEAR);
    }

    //! The combination of kLINEAR + kFLOAT is supported.
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override
    {
        assert(nbInputs == 3 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        if (pos == 1) // the datatype of the first input is kINT32
            condition &= inOut[pos].type == DataType::kINT32;
        else
            condition &= inOut[pos].type == DataType::kFLOAT;

        return condition;
    }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        assert(inputTypes && nbInputs == 3);
        return inputTypes[0];
    }

private:

    int multiplyArray(const int *arr, int len) {
        int i,temp=1;
        for(i=0;i<len;i++) {
            temp=temp*arr[i];
        }
        return temp;
    }

    const char* mPluginNamespace;
    std::string mNamespace;

    using nvinfer1::IPluginV2Ext::configurePlugin;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;
};

class ScatterNDCreator : public nvinfer1::IPluginCreator {
public:
    ScatterNDCreator()
    {
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        return new ScatterND();
    }

    const char* getPluginName() const noexcept override {return "ScatterND";}
    const char* getPluginVersion() const noexcept override {return "1";}

    void setPluginNamespace(const char* szNamespace) noexcept override {mNamespace = szNamespace;}
    const char* getPluginNamespace() const noexcept override {return mNamespace.c_str();}

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        std::cout << __FUNCTION__ << std::endl;
        return &mFC;
    }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        std::cout << __FUNCTION__ << std::endl;
        ScatterND* obj = new ScatterND{};
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;

    }

private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin

} // namespace nvinfer1
