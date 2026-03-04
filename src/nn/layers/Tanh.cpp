#include "core/Value.h"

#include "nn/layers/Tanh.h"

std::vector<Value> Tanh::operator()(const std::vector<Value>& input) {
	size_t out = input.size();

	std::vector<Value> out_values;
	out_values.reserve(out);

	for (size_t i = 0; i < out; i++) {
		out_values.push_back(input[i].tanh());
	}

	return out_values;
}

std::vector<Value> Tanh::parameters() const {
	return std::vector<Value>();
}

std::string Tanh::description() const {
	return "Tanh()";
}