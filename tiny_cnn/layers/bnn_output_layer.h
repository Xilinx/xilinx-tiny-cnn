#pragma once
#include "tiny_cnn/layers/bnn_threshold_layer.h"

namespace tiny_cnn {
class bnn_output_layer : public bnn_threshold_layer {
public:
    using bnn_threshold_layer::bnn_threshold_layer;

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {
        vec_t &out = output_[index];

        for(unsigned int ch = 0; ch < channels_; ch++) {
          for(unsigned int j = 0; j < dim_; j++) {
              unsigned int pos = ch*dim_ + j;
              out[pos] = in[pos] - thresholds_[ch];

              if(invertOutput_[ch])
                  out[pos] = -out[pos];
          }
        }

        return next_ ? next_->forward_propagation(out, index) : out;
    }
};
}

