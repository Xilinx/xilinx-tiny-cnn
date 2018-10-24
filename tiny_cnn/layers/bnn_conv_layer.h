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
#include "tiny_cnn/activations/activation_function.h"
#include <vector>
#include <string>
#include <iostream>

namespace tiny_cnn {

class bnn_conv_layer : public layer<activation::identity> {
public:
    typedef layer<activation::identity> Base;

    // assumptions: padding::valid, wstride=hstride=1, no bias
    bnn_conv_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_size,
        cnn_size_t in_channels,
        cnn_size_t out_channels,
        bool usePopcount = false, std::string binaryParamFile = "")
        : Base(in_width*in_height*in_channels, (in_width-window_size+1)*(in_height-window_size+1)*out_channels,
               out_channels*in_channels*window_size*window_size, 0),
          in_width_(in_width), in_height_(in_height), window_size_(window_size), in_channels_(in_channels), out_channels_(out_channels),
          Wbin_(out_channels*in_channels*window_size*window_size, false),
          usePopcount_(usePopcount)
    {
        // TODO re-enable parallelization -- need to support worker index in forward prop
        Base::set_parallelize(false);
        out_width_ = (in_width-window_size+1);
        out_height_ = (in_height-window_size+1);

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

    ///< number of incoming connections for each output unit
    virtual size_t fan_in_size() const override
    {
        return in_channels_ * window_size_ * window_size_;
    }

    ///< number of outgoing connections for each input unit
    virtual size_t fan_out_size() const override
    {
        return out_channels_ * window_size_ * window_size_;
    }

    ///< number of connections
    virtual size_t connection_size() const override
    {
        return out_height_ * out_width_ * fan_in_size();
    }

    std::string layer_type() const override { return "bnn_conv_layer"; }

    virtual void post_update() override {
        // once the weights have been updated, update the binarized versions too
        float2bipolar(W_, Wbin_);
    }

    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        throw "Not implemented";
    }

    virtual const vec_t& forward_propagation(const vec_t& in_raw, size_t worker_index) override
    {
        // turn the input into a vector of bools
        std::vector<bool> in_bin(in_raw.size(), false);
        float2bipolar(in_raw, in_bin);
        vec_t &out = output_[worker_index];

        // TODO implement actual binarized version
        // TODO support padding modes
        // TODO support worker index for parallelization
        for(cnn_size_t oc = 0; oc < out_channels_; oc++) {
            unsigned int output_base = oc * out_height_ * out_width_;
            for(cnn_size_t oy = 0; oy < out_height_; oy++) {
                for(cnn_size_t ox = 0; ox < out_width_; ox++) {
                    int acc = 0;
                    for(cnn_size_t ic = 0; ic < in_channels_; ic++) {
                        unsigned int weight_base = oc*(window_size_*window_size_*in_channels_) + (window_size_*window_size_*ic);
                        unsigned int input_base = ic*(in_width_*in_height_) + oy*in_width_ + ox;
                        for(cnn_size_t ky = 0; ky < window_size_; ky++) {
                            for(cnn_size_t kx = 0; kx < window_size_; kx++) {
                                unsigned int weight_ind = weight_base + ky*window_size_ + kx;
                                unsigned int input_ind = input_base + ky*in_width_ + kx;
                                if(usePopcount_) {
                                    // accumulate popcount (+1 bits) only
                                    acc += Wbin_[weight_ind] == in_bin[input_ind] ? +1 : 0;
                                } else {
                                    // accumulate sum of +1 and -1s
                                    acc += Wbin_[weight_ind] == in_bin[input_ind] ? +1 : -1;
                                }
                            }
                        }
                    }
                    unsigned int output_ind = output_base + oy * out_width_ + ox;
                    out[output_ind] = acc;
                }
            }
        }

        CNN_LOG_VECTOR(out, "[bnn_conv_layer] forward ");

        return next_ ? next_->forward_propagation(out, worker_index) : out;
    }

    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        throw "Not implemented";
    }

protected:
    bool usePopcount_;
    std::vector<bool> Wbin_;
    cnn_size_t in_width_;
    cnn_size_t in_height_;
    cnn_size_t window_size_;
    cnn_size_t in_channels_;
    cnn_size_t out_channels_;
    cnn_size_t out_width_;
    cnn_size_t out_height_;

    // utility function to convert a vector of floats into a vector of bools, where the
    // output boolean represents the sign of the input value (false: negative,
    // true: positive)
    void float2bipolar(const vec_t & in, std::vector<bool> & out) {
        for(unsigned int i = 0; i < in.size(); i++)
            out[i] = in[i] >= 0 ? true : false;
    }
};

}
