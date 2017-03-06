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
#include "tiny_cnn/util/util.h"

// each channel's feature maps are either contiguous or interleaved,
// and this layer switches between those two modes. does not touch
// row major / column major order of the pixels.

namespace tiny_cnn {


template<typename Activation>
class chaninterleave_layer : public layer<Activation> {
public:
    CNN_USE_LAYER_MEMBERS;

    typedef layer<Activation> Base;

    explicit chaninterleave_layer(cnn_size_t channels, cnn_size_t pixelsPerChan, bool deinterleave)
        : Base(channels*pixelsPerChan, channels*pixelsPerChan, 0, 0), channels_(channels),
          pixelsPerChan_(pixelsPerChan), deinterleave_(deinterleave)
    {

    }

    size_t param_size() const override {
        return 0;
    }

    size_t connection_size() const override {
        return this->in_size();
    }

    size_t fan_in_size() const override {
        return 1;
    }

    size_t fan_out_size() const override {
        return 1;
    }

    std::string layer_type() const override { return "chaninterleave_layer"; }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t& a = a_[index];
        vec_t& out = output_[index];

        for(unsigned int c = 0; c < channels_; c++) {
            for(unsigned int pix = 0; pix < pixelsPerChan_; pix++) {
                if(deinterleave_) {
                    out[c * pixelsPerChan_ + pix] = in[pix*channels_ + c];
                } else {
                    out[pix*channels_ + c] = in[c * pixelsPerChan_ + pix];
                }
            }
        }

        return next_ ? next_->forward_propagation(out, index) : out;
    }

    virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) override {
        throw "Not implemented";

        return current_delta;
    }

    const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        throw "Not implemented";

        return current_delta2;
    }

protected:
    cnn_size_t channels_, pixelsPerChan_;
    bool deinterleave_;
};

} // namespace tiny_cnn
