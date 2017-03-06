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
#include "tiny_cnn/util/product.h"
#include <vector>
#include <string>
#include <iostream>

namespace tiny_cnn {

template<typename Activation>
class bnn_fc_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    bnn_fc_layer(cnn_size_t in_dim, cnn_size_t out_dim,
                 bool usePopcount = false, bool rowMajorWeights = false, std::string binaryParamFile = "")
        : Base(in_dim, out_dim, size_t(in_dim) * out_dim, 0), Wbin_(in_dim*out_dim, false),
          usePopcount_(usePopcount), rowMajorWeights_(rowMajorWeights) {
        if(binaryParamFile != "")
          loadFromBinaryFile(binaryParamFile);
    }

    void loadFromBinaryFile(std::string fileName) {
      // TODO this assumes the binary file always uses 8 bytes per threshold entry

      // load weights
      std::ifstream wf(fileName, std::ios::binary | std::ios::in);
      if(!wf.is_open())
        throw "Could not open file";
      for(unsigned int line = 0 ; line < Wbin_.size(); line++) {
        unsigned long long e = 0;
        wf.read((char *)&e, sizeof(unsigned long long));
        Wbin_[line] = e == 1 ? true : false;
      }
      wf.close();
    }

    size_t connection_size() const override {
        return size_t(in_size_) * out_size_;
    }

    size_t fan_in_size() const override {
        return in_size_;
    }

    size_t fan_out_size() const override {
        return out_size_;
    }

    virtual void post_update() {
        // once the weights have been updated, update the binarized versions too
        float2bipolar(W_, Wbin_);
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        std::vector<bool> in_bin(in_size_, false);
        // explicitly binarize the input
        float2bipolar(in, in_bin);
        vec_t &a = a_[index];
        vec_t &out = output_[index];

        for_i(parallelize_, out_size_, [&](int i) {
            a[i] = float_t(0);
            for (cnn_size_t c = 0; c < in_size_; c++) {
                // multiplication for binarized values is basically XNOR (equals)
                // i.e. if two values have the same sign (pos-pos or neg-neg)
                // the mul. result will be positive, otherwise negative
                // when using the popcount mode, consider positive results only
                const unsigned int wInd = rowMajorWeights_ ? i*in_size_+c : c*out_size_+i;
                if(usePopcount_)
                  a[i]  += (Wbin_[wInd] == in_bin[c]) ? +1 : 0;
                else
                  a[i]  += (Wbin_[wInd] == in_bin[c]) ? +1 : -1;
            }
        });

        for_i(parallelize_, out_size_, [&](int i) {
            out[i] = h_.f(a, i);
        });
        CNN_LOG_VECTOR(out, "[bfc]forward");

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

    std::string layer_type() const override { return "bnn_fc_layer"; }

protected:
    std::vector<bool> Wbin_;
    bool usePopcount_, rowMajorWeights_;

    // utility function to convert a vector of floats into a vector of bools, where the
    // output boolean represents the sign of the input value (false: negative,
    // true: positive)
    void float2bipolar(const vec_t & in, std::vector<bool> & out) {
        for(unsigned int i = 0; i < in.size(); i++)
            out[i] = in[i] >= 0 ? true : false;
    }

};

} // namespace tiny_cnn
