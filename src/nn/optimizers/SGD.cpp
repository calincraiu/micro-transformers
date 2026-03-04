#include "nn/optimizers/SGD.h"

SGD::SGD(std::vector<Value> params, float lr) :
	m_params(params), m_lr(lr) 
{
}

void SGD::step() {
	for (Value param : m_params) {
		param.set_data(param.get_data() - m_lr * param.get_grad());
	}
}