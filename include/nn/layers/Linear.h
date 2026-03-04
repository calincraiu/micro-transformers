#ifndef LINEAR_H
#define LINEAR_H

#include <vector>

#include "core/Node.h"
#include "core/Value.h"
#include "core/Tape.h"

#include "nn/layers/Layer.h"

class Linear : public Layer {
public:
	Linear(Tape* tape, size_t features_in, size_t features_out, bool use_bias = true);

	std::vector<Value> operator()(const std::vector<Value>& input) override;
	std::vector<Value> parameters() const override;
	std::string description() const override;

private:
	Tape* m_tape = nullptr;
	std::vector<Value> m_weights;
	std::vector<Value> m_biases;
	size_t m_features_in;
    size_t m_features_out;
};

#endif // LINEAR_H