#include <iostream>
#include <sstream>

#include "nn/layers/Linear.h"
#include "nn/layers/ReLU.h"
#include "nn/models/MLP.h"

MLP::MLP(Tape* tape, const std::vector<size_t>& sizes) :
	m_tape(tape)
{
	for (size_t i = 0; i < sizes.size() - 1; i++)
	{
		bool is_last_layer = (i == sizes.size() - 2);
		bool activate = !is_last_layer; // No activation on final layer

		m_layers.push_back(std::make_unique<Linear>(m_tape, sizes[i], sizes[i + 1], true));

		if (activate) {
			m_layers.push_back(std::make_unique<ReLU>());
		}
	}
}

std::vector<Value> MLP::operator()(const std::vector<Value>& input) {
	std::vector<Value> out = input;
	for (size_t i = 0; i < m_layers.size(); i++) {
		out = (*m_layers[i])(out);
	}
	return out;
}

std::vector<Layer*> MLP::layers() const {
	std::vector<Layer*> layersP;
	for (const std::unique_ptr<Layer>& layer : m_layers) {
		layersP.push_back(layer.get());
	}
	return layersP;
}

std::vector<Value> MLP::parameters() const {
	std::vector<Value> all_params;
	for (const std::unique_ptr<Layer>& layer : m_layers) {
		std::vector<Value> layer_params = layer->parameters();
		all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
	}
	return all_params;
}

std::string MLP::description() const {
	std::ostringstream oss;

	oss << "MLP(\n";
	for (size_t i = 0; i < m_layers.size(); ++i) {
		oss << "  (" << i << "): "
			<< m_layers[i]->description()
			<< "\n";
	}
	oss << ")";

	return oss.str();
}