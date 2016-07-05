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
