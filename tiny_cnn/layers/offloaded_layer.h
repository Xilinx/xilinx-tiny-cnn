#pragma once

// offloaded_layer -- simply calls a hook function every time its forward pass is called

#include "tiny_cnn/activations/activation_function.h"
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/util/product.h"
#include "tiny_cnn/util/util.h"
#include <vector>

// function type for offload handling. args are (input, output)
typedef void (*OffloadHandler)(const float_t *, unsigned int, float_t *, unsigned int, unsigned int);

namespace tiny_cnn {

class offloaded_layer : public layer<activation::identity> {
public:
    typedef layer<activation::identity> Base;

    offloaded_layer(cnn_size_t in_dim, cnn_size_t out_dim, OffloadHandler handler,
                    unsigned int offloadID) :
        Base(in_dim, out_dim, 0, 0), offloadHandler_(handler), offloadID_(offloadID)
    {
        // disable parallelization for offloaded layers
        outBuf_ = new float_t[out_dim];
        Base::set_parallelize(false);
    }

    ~offloaded_layer() {
        delete [] outBuf_;
    }

    std::string layer_type() const override { return "offloaded"; }

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

    // forward prop does nothing except calling the
    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        const float_t * inP = &in[index];

        offloadHandler_(inP, in_dim(), outBuf_, out_dim(), offloadID_);

        vec_t & out = output_[index];

        for(size_t i = 0; i < out_dim(); i++) {
            out[i] = outBuf_[i];
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
    OffloadHandler offloadHandler_;
    unsigned int offloadID_;
    float_t * outBuf_;

};

} // namespace tiny_cnn
