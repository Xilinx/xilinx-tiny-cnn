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

// offloaded_layer -- simply calls a hook function every time its forward pass is called

#include "tiny_cnn/activations/activation_function.h"
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/util.h"
#include <vector>



namespace tiny_cnn {

typedef struct {
    cnn_size_t in_width;            // input feature map width
    cnn_size_t in_height;           // input feature map height
    cnn_size_t window_size;         // convolution kernel window size
    cnn_size_t in_channels;         // # input feature maps
    cnn_size_t out_channels;        // # output feature maps
} OffloadConvParams;

#ifdef SOLITAIRE
// function type for offload handling. args are (input, output, offloadID, conv params if any or 0, target set of weigths)
typedef void (*OffloadHandler)(const vec_t &, vec_t &, unsigned int, OffloadConvParams *, unsigned int);

class offloaded_layer : public layer<activation::identity> {
public:
    typedef layer<activation::identity> Base;

    offloaded_layer(cnn_size_t in_dim, cnn_size_t out_dim, OffloadHandler handler,
                    unsigned int offloadID, OffloadConvParams * convParams = 0, unsigned int targetSet = 0) :
        Base(in_dim, out_dim, 0, 0), offloadHandler_(handler), offloadID_(offloadID),
        offloadConvParams_(convParams), targetSet_(targetSet)
		
#else
// function type for offload handling. args are (input, output, offloadID, conv params if any or 0)	
typedef void (*OffloadHandler)(const vec_t &, vec_t &, unsigned int, OffloadConvParams *);

class offloaded_layer : public layer<activation::identity> {
public:
    typedef layer<activation::identity> Base;

    offloaded_layer(cnn_size_t in_dim, cnn_size_t out_dim, OffloadHandler handler,
                    unsigned int offloadID, OffloadConvParams * convParams = 0) :
        Base(in_dim, out_dim, 0, 0), offloadHandler_(handler), offloadID_(offloadID),
        offloadConvParams_(convParams)
#endif
    {
        // disable parallelization for offloaded layers
        Base::set_parallelize(false);
    }

    std::string layer_type() const override { return "offloaded"; }

    size_t param_size() const override {
        return 0;
    }

    size_t connection_size() const override {
        return 0;
    }

    size_t fan_in_size() const override {
        return in_size_;
    }

    size_t fan_out_size() const override {
        return out_size_;
    }

    // forward prop does nothing except calling the
    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t & out = output_[index];
#ifdef SOLITAIRE
        offloadHandler_(in, out, offloadID_, offloadConvParams_, targetSet_);
#else
	offloadHandler_(in, out, offloadID_, offloadConvParams_);
#endif
        return next_ ? next_->forward_propagation(out, index) : out;
    }

    // offloaded layer is feedforward only, does not support training
    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        throw "Not implemented";
        return curr_delta;
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        throw "Not implemented";
        return current_delta2;
    }

protected:
    OffloadHandler offloadHandler_;
    OffloadConvParams * offloadConvParams_;
    unsigned int offloadID_;
#ifdef SOLITAIRE
    unsigned int targetSet_;
#endif
};

} // namespace tiny_cnn

