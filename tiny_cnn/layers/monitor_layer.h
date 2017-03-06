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

// monitor layer - print out contents passing through

#include "tiny_cnn/activations/activation_function.h"
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/util.h"
#include <vector>
#include <iostream>
#include <string>

namespace tiny_cnn {

class monitor_layer : public layer<activation::identity> {
public:
    typedef layer<activation::identity> Base;

    monitor_layer(cnn_size_t dim, std::string monitorName) :
        Base(dim, dim, 0, 0), monitorName_(monitorName)
    {
        // disable parallelization for monitor layers
        Base::set_parallelize(false);
    }

    std::string layer_type() const override { return "monitor"; }

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

        std::cout << monitorName_ << std::endl;

        for(unsigned int i = 0; i < in.size(); i++) {
            out[i] = in[i];
            std::cout << i << " " << in[i] << std::endl;
        }

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
    std::string monitorName_;

};

} // namespace tiny_cnn
