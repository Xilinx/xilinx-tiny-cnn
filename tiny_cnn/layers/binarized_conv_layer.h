#pragma once

#include "tiny_cnn/layers/convolutional_layer.h"
#include "tiny_cnn/util/product.h"
#include <vector>

namespace tiny_cnn {

template<typename Activation>
class binarized_conv_layer : public convolutional_layer<Activation> {
public:
    typedef convolutional_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    // assumptions: padding::same, wstride=hstride=1, with bias
    binarized_conv_layer(cnn_size_t in_width,
        cnn_size_t in_height,
        cnn_size_t window_size,
        cnn_size_t in_channels,
        cnn_size_t out_channels)
        : Base(in_width, in_height, window_size, in_channels, out_channels, padding::same,
               true, 1, 1)
    {
    }

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
        // TODO
    }

    const vec_t& back_propagation(const vec_t& curr_delta, size_t index) override {
        throw "Not implemented";
    }

protected:
    // utility function to convert a vector of floats into a vector of bools, where the
    // output boolean represents the sign of the input value (false: negative,
    // true: positive)
    void float2bipolar(const vec_t & in, std::vector<bool> & out) {
        for(unsigned int i = 0; i < in.size(); i++)
            out[i] = in[i] >= 0 ? true : false;
    }
};

}
