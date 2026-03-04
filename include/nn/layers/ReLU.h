#ifndef RELU_H
#define RELU_H

#include "nn/layers/Layer.h"

class ReLU : public Layer {
public:
	std::vector<Value> operator()(const std::vector<Value>& input) override;
	std::vector<Value> parameters() const override;
	std::string description() const override;
};


#endif // RELU_H