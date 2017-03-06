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

// function type for offload handling. args are (input, thresholds, weights, output)
typedef void (*BinMatVecMult)(std::vector<bool>&, std::vector<unsigned int>&, std::vector<bool>&, std::vector<bool>&);

// implements a binarized fully-connected layer and "compacted" batch normalization
// pretrained only, i.e. does not support training in tiny-cnn
// use the set_threshold_from_batchnorm function for each neuron to absorb the
// batchnorm parameters into thresholds

namespace tiny_cnn {

template<typename Activation>
class binarynet_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    binarynet_layer(cnn_size_t in_dim, cnn_size_t out_dim, BinMatVecMult offload = 0)
        : Base(in_dim, out_dim, size_t(in_dim) * out_dim, 0), Offload_(offload) {
        // initialize all binarized weights and thresholds
        for(unsigned int i = 0; i < in_size_ * out_size_; i++) {
            Wbin_.push_back(false);
        }

        for(unsigned int i = 0; i < out_size_; i++) {
            Threshold_.push_back(0);
        }
    }

    // save/load
    virtual void save(std::ostream& os) const {
        for (auto w : Wbin_) os << (w ? 1 : 0) << "\n";
        for (auto thr : Threshold_) os << thr << "\n";
    }

    virtual void load(std::istream& is) {
        bool w;
        for(unsigned int i = 0; i < in_size_ * out_size_; i++) {
            is >> w;
            Wbin_[i] = w;
        }
        for (auto& thr : Threshold_) is >> thr;
    }

    size_t connection_size() const override {
        // number of connections/parameters in this layer
        // - one for each synaptic weight
        // - one for each neuron's threshold
        // - one for each neuron's flip indicator
        return size_t(in_size_) * out_size_ + 2*out_size_;
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

    void set_threshold_from_batchnorm(size_t index, float_t mean, float_t gamma, float_t invstd, float_t beta) {
        // "compact" the batchnorm parameters into a single threshold
        // how does this work? we exploit the fact that the batchnorm parameters are constant during inference.
        // let fc = sum(weights*inputs) for a binarized neuron
        // let beta, gamma, mean and invstd be the learned batchnorm parameters
        // a binarized neuron with batch normalization and the sign activation function
        // computes the following:
        // sign(gamma*(fc - mean)*invstd+beta)
        // if we plot the neuron output as a function of the fc output, it looks like this:
        //        threshold
        //        |
        //        v
        //        |--------
        //        |
        // _______|
        // so for some value of fc (which we call the threshold), the neuron's output will change sign.
        // since the batchnorm output itself is linear, we can compute where the sign changes
        // by solving for the fc value that sets the batch-normalized output value to zero:
        // gamma*(fc - mean)*invstd+beta = 0
        // fc = mean-beta/(gamma*invstd)
        int thres = mean - (beta / (gamma*invstd));

        // depending on the sign of the multiplicative factor (gamma*invstd) the neuron output plot
        // may reverse direction, e.g:
        //        |--------           --------|
        //        |           or              |
        // _______|                           |_________
        // this could be handled by keeping an extra bit per neuron and flipping the output sign,
        // but we flip the signs of all weights and the threshold instead.
        if((gamma*invstd) < 0) {
            thres = -thres;
            for (cnn_size_t c = 0; c < in_size_; c++) {
                Wbin_[c*out_size_ + index] = !Wbin_[c*out_size_ + index];
            }
        }
        // ensure a positive threshold by averaging with the neuron fan-in
        // by ensuring a positive threshold, it becomes possible to use popcount (instead of signed add)
        // for the addition, followed by a greater than comparison for the threshold
        Threshold_[index] = (thres + fan_in_size()) / 2;
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        std::vector<bool> in_bin(in_size_, false);
        // explicitly binarize the input
        float2bipolar(in, in_bin);
        vec_t &a = a_[index];
        vec_t &out = output_[index];

        if(Offload_ != 0) {
            // call offload hook to perform actual computation
            std::vector<bool> res(out_size_, false);
            Offload_(in_bin, Threshold_, Wbin_, res);
            for(unsigned int i = 0; i < out_size_; i++)
                out[i] = res[i] == 1 ? +1 : -1;
        } else {
            for_i(parallelize_, out_size_, [&](int i) {
                a[i] = 0;
                for (cnn_size_t c = 0; c < in_size_; c++) {
                    // multiplication for binarized values is basically XNOR (equals)
                    // i.e. if two values have the same sign (pos-pos or neg-neg)
                    // we increment the popcount for this row
                    a[i]  += (Wbin_[c*out_size_ + i] == in_bin[c]) ? +1 : 0;
                }
                // compute the activation by comparing against the threshold
                // (the tiny-cnn specified act.fn. becomes unnecessary)
                out[i] = a[i] >= Threshold_[i] ? +1 : -1;
            });
        }



        CNN_LOG_VECTOR(out, "[binarynet]forward");

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

    std::string layer_type() const override { return "binarynet-fully-connected"; }

protected:
    std::vector<bool> Wbin_;
    std::vector<unsigned int> Threshold_;
    BinMatVecMult Offload_;

    // utility function to convert a vector of floats into a vector of bools, where the
    // output boolean represents the sign of the input value (false: negative,
    // true: positive)
    void float2bipolar(const vec_t & in, std::vector<bool> & out) {
        for(unsigned int i = 0; i < in.size(); i++) {
            out[i] = in[i] >= 0 ? true : false;
        }
    }



};

} // namespace tiny_cnn
