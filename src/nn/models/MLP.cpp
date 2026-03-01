#include "nn/layers/Linear.h"
#include "nn/models/MLP.h"

MLP::MLP(Tape* tape, const std::vector<size_t>& sizes) :
	m_tape(tape)
{
	for (size_t i = 0; i < sizes.size() - 1; i++)
	{
		m_layers.push_back(std::make_unique<Linear>(m_tape, sizes[i], sizes[i + 1], true));
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
	std::vector<Value> params;
	for (const auto& layer : m_layers) {
		auto param = layer->parameters();
		params.insert(params.end(), param.begin(), param.end());
	}
	return params;
}