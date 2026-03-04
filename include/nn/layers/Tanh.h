#ifndef TANH_H
#define TANH_H

#include "nn/layers/Layer.h"

class Tanh : public Layer {
public:
	std::vector<Value> operator()(const std::vector<Value>& input) override;
	std::vector<Value> parameters() const override;
	std::string description() const override;
};


#endif // TANH_H