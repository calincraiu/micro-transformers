#ifndef SGD_H
#define SGD_H

#include <vector>

#include "core/Value.h"

class SGD {
public:
	SGD(std::vector<Value> params, float lr);
	void step();

private:
	float m_lr; // learning_rate
	std::vector<Value> m_params; 
};

#endif // SGD_H