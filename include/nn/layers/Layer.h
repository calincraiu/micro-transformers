#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include "core/Value.h"

class Layer {
public:
	virtual ~Layer() = default;
	virtual std::vector<Value> operator()(const std::vector<Value>& input) = 0;
	virtual std::vector<Value> parameters() const = 0;
};

#endif // LAYER_H