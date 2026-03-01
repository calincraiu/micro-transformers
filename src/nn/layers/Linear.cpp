#include "nn/layers/Linear.h"

Linear::Linear(Tape* tape, size_t features_in, size_t features_out, bool use_bias) : 
	Layer(), m_tape(tape)
{
	m_weights.reserve(features_out * features_in);
	if (use_bias) {
		m_biases.reserve(features_out);
	}

	for (size_t o = 0; o < features_out; o++)  {
		for (size_t i = 0; i < features_in; i++) {
			float val = (float(rand() % 1000) / 500.0f) - 1.0f; // -1 to 1
			m_weights.emplace_back(m_tape->create_leaf(val), m_tape);
		}

		if (use_bias) { 
			m_biases.emplace_back(m_tape->create_leaf(0.0f), m_tape);
		}
	}
}

std::vector<Value> Linear::operator()(const std::vector<Value>& input) {
	size_t in = input.size();
	if (in == 0) {
		return {};
	}
	size_t out = m_weights.size() / in;

	std::vector<Value> out_values;
	out_values.reserve(out);

	for (size_t o = 0; o < out; o++) {
		Value sum = Value(m_tape->create_leaf(0.0f), m_tape);  // start at zero

		for (size_t i = 0; i < in; i++) {
			size_t w_idx = o * in + i; // weight index
			sum = sum + m_weights[w_idx] * input[i];
		}

		if (!m_biases.empty()) {
			sum = sum + m_biases[o];
		}

		out_values.push_back(sum.relu()); // with activation function
	}
	return out_values;
}

std::vector<Value> Linear::parameters() const {
	std::vector<Value> params = m_weights;
	params.insert(params.end(), m_biases.begin(), m_biases.end());
	return params;
}