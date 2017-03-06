/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
#pragma once
#include "tiny_cnn/layers/layer.h"
#include <fstream>
#include <string>

// right now functional during inference only
// pre-trained batchnorm params must be manually set
// TODO:
// - implement proper batch normalization during training

// during training, for each neuron output, batch normalization learns
// four statistics from the data that passes through it. we keep all
// four of these as part of the layer's weights, in the following order:
// * all of the shifts (beta)
// * all of the scales (gamma)
// * all of the means (mu)
// * all of the inverted stddevs (inv_std)

namespace tiny_cnn {
template<typename Activation>
class batchnorm_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    // channels: number of channels. each channel has a batchnorm parameter set.
    // dim: number of pixels/elements in each channel.
    batchnorm_layer(cnn_size_t channels, cnn_size_t dim = 1, std::string paramFile = "")
        : Base(dim*channels, dim*channels, 4*channels, 0), dim_(dim), channels_(channels)
    {
      if(paramFile != "") {
          loadFromBinaryFile(paramFile);
      }
    }

    void loadFromBinaryFile(std::string fileName) {
      // TODO this assumes the binary file always uses 4 bytes per batchnorm parameter

      std::ifstream wf(fileName, std::ios::binary | std::ios::in);
      for(unsigned int line = 0 ; line < Base::W_.size(); line++) {
        float e = 0;
        wf.read((char *)&e, sizeof(float));
        W_[line] = e;
      }
      wf.close();
    }

    size_t connection_size() const override {
        return in_size_;
    }

    size_t fan_in_size() const override {
        return dim_;
    }

    size_t fan_out_size() const override {
        return dim_;
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t &a = a_[index];
        vec_t &out = output_[index];

        for_i(parallelize_, channels_, [&](int ch) {
            for(unsigned int j = 0; j < dim_; j++) {
                unsigned int pos = ch*dim_ + j;
                a[pos] = gamma(ch) * (in[pos] - mean(ch)) * invstd(ch) + beta(ch);
            }
        });

        for_i(parallelize_, out_size_, [&](int i) {
            out[i] = h_.f(a, i);
        });
        CNN_LOG_VECTOR(out, "[bn]forward");

        return next_ ? next_->forward_propagation(out, index) : out;
    }

    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        throw "Not yet implemented";
        return curr_delta;
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        throw "Not yet implemented";
        return current_delta2;
    }

    std::string layer_type() const override { return "batchnorm"; }

protected:
    unsigned int dim_;
    unsigned int channels_;
    inline float_t beta(unsigned int ind) {
        return W_[(channels_ * 0) + ind];
    }

    inline float_t gamma(unsigned int ind) {
        return W_[(channels_ * 1) + ind];
    }

    inline float_t mean(unsigned int ind) {
        return W_[(channels_ * 2) + ind];
    }

    inline float_t invstd(unsigned int ind) {
        return W_[(channels_ * 3) + ind];
    }

};

} // namespace tiny_cnn
