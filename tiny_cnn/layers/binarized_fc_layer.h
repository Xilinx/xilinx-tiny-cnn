#pragma once
#include "tiny_cnn/layers/layer.h"
#include "tiny_cnn/util/product.h"

namespace tiny_cnn {

template<typename Activation>
class binarized_fc_layer : public layer<Activation> {
public:
    typedef layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    binarized_fc_layer(cnn_size_t in_dim, cnn_size_t out_dim)
        : Base(in_dim, out_dim, size_t(in_dim) * out_dim, 0) {}

    size_t connection_size() const override {
        return size_t(in_size_) * out_size_;
    }

    size_t fan_in_size() const override {
        return in_size_;
    }

    size_t fan_out_size() const override {
        return out_size_;
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t &a = a_[index];
        vec_t &out = output_[index];

        for_i(parallelize_, out_size_, [&](int i) {
            a[i] = float_t(0);
            /*unsigned int reduced = 0;
            for (cnn_size_t c = 0; c < in_size_; c++) {
                bool signW = (W_[c*out_size_ + i] >= 0);
                bool signIn = (in[c] >= 0);
                if(signW == signIn)
                    reduced++;
                else
                    reduced--;
            }*/

            for (cnn_size_t c = 0; c < in_size_; c++) {
                a[i] += (W_[c*out_size_ + i] > 0 ? 1 : -1) * (in[c] > 0 ? 1 : -1);
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

    std::string layer_type() const override { return "binarized-fully-connected"; }

};

} // namespace tiny_cnn
