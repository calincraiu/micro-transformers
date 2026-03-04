#include "core/Value.h"

#include "nn/layers/ReLU.h"

std::vector<Value> ReLU::operator()(const std::vector<Value>& input) {
	size_t out = input.size();

	std::vector<Value> out_values;
	out_values.reserve(out);

	for (size_t i = 0; i < out; i++) {
		out_values.push_back(input[i].relu());
	}

	return out_values;
}

std::vector<Value> ReLU::parameters() const {
	return std::vector<Value>();
}

std::string ReLU::description() const {
	return "ReLU()";
}