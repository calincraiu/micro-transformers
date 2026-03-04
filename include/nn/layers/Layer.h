#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>

#include "core/Value.h"

class Layer {
public:
	virtual std::vector<Value> operator()(const std::vector<Value>& input) = 0;
	virtual std::vector<Value> parameters() const = 0;
	virtual std::string description() const = 0;

	virtual ~Layer() = default;

};

#endif // LAYER_H