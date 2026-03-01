#ifndef LINEAR_H
#define LINEAR_H

#include <vector>

#include "core/Node.h"
#include "core/Value.h"
#include "core/Tape.h"

#include "nn/layers/Layer.h"

class Linear : public Layer {
public:
	/// <summary>
	/// Constructor - creates parameters (leaf nodes on the tape) immediately.
	/// </summary>
	/// <param name="tape">Tape that stores computation graph.</param>
	/// <param name="features_in">number of features in.</param>
	/// <param name="features_out">number of features out.</param>
	Linear(Tape* tape, size_t features_in, size_t features_out, bool use_bias = true);

	/// <summary>
	/// Calling the Layer over the inputs - creates all intermediary nodes on the Tape.
	/// </summary>
	/// <param name="input">input weight Values that plug into the current layer.</param>
	/// <returns>the result of the (essentially) matrix multiplication.</returns>
	std::vector<Value> operator()(const std::vector<Value>& input) override;

	/// <summary>
	/// Gets all the parameters of this Layer (weights and biases).
	/// </summary>
	/// <returns>the Value parameters of this Layer.</returns>
	std::vector<Value> parameters() const override;
	

private:
	Tape* m_tape = nullptr;
	std::vector<Value> m_weights;
	std::vector<Value> m_biases;
};

#endif // LINEAR_H