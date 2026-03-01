#ifndef MLP_H
#define MLP_H

#include <vector>
#include <memory>

#include <core/Value.h>
#include <core/Tape.h>

#include "nn/layers/Layer.h"

class MLP {
public:
	MLP(Tape* tape, const std::vector<size_t>& sizes);

	std::vector<Value> operator()(const std::vector<Value>& input);

	std::vector<Layer*> layers() const;
	std::vector<Value> parameters() const;

private:
	Tape* m_tape = nullptr;
	std::vector<std::unique_ptr<Layer>> m_layers;
};


#endif