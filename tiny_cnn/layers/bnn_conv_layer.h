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
        cnn_size_t out_channels)
        : Base(in_width*in_height*in_channels, (in_width-window_size+1)*(in_height-window_size+1)*out_channels,
               out_channels*in_channels*window_size*window_size, 0),
          in_width_(in_width), in_height_(in_height), window_size_(window_size), in_channels_(in_channels), out_channels_(out_channels),
          Wbin_(out_channels*in_channels*window_size*window_size, false)
    {
        // TODO re-enable parallelization -- need to support worker index in forward prop
        Base::set_parallelize(false);
        out_width_ = (in_width-window_size+1);
        out_height_ = (in_height-window_size+1);

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

    virtual void post_update() {
        // once the weights have been updated, update the binarized versions too
        // TODO re-enable binarization once we are ready
        //float2bipolar(W_, Wbin_);
    }

    virtual const vec_t& back_propagation_2nd(const vec_t& current_delta2) override {
        throw "Not implemented";
    }

    virtual const vec_t& forward_propagation(const vec_t& in_raw, size_t worker_index) override
    {
        vec_t &out = output_[worker_index];

        // TODO implement actual binarized version
        // TODO support padding modes
        // TODO support worker index for parallelization
        for(cnn_size_t oc = 0; oc < out_channels_; oc++) {
            unsigned int output_base = oc * out_height_ * out_width_;
            for(cnn_size_t oy = 0; oy < out_height_; oy++) {
                for(cnn_size_t ox = 0; ox < out_width_; ox++) {
                    float_t acc = 0;
                    for(cnn_size_t ic = 0; ic < in_channels_; ic++) {
                        unsigned int weight_base = oc*(window_size_*window_size_*in_channels_) + (window_size_*window_size_*ic);
                        unsigned int input_base = ic*(in_width_*in_height_) + oy*in_width_ + ox;
                        for(cnn_size_t ky = 0; ky < window_size_; ky++) {
                            for(cnn_size_t kx = 0; kx < window_size_; kx++) {
                                unsigned int weight_ind = weight_base + ky*window_size_ + kx;
                                unsigned int input_ind = input_base + ky*in_width_ + kx;
                                acc += W_[weight_ind] * in_raw[input_ind];
                            }
                        }
                    }
                    unsigned int output_ind = output_base + oy * out_width_ + ox;
                    out[output_ind] = acc;
                }
            }
        }

        return next_ ? next_->forward_propagation(out, worker_index) : out;
    }

    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        throw "Not implemented";
    }

protected:
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
