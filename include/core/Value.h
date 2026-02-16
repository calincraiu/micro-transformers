#ifndef VALUE_H
#define VALUE_H

#include <vector>
#include <memory>
#include <optional>

class Value : public std::enable_shared_from_this<Value> {
public:
	// Constructors
	explicit Value(float data);
	Value(float data, std::vector<std::shared_ptr<Value>> children);

	// Methods
	void print();
	int num_children();

	// Static Methods

	/// <summary>
	/// Create a shared_ptr of a Value.
	/// </summary>
	/// <param name="data">the Value data.</param>
	/// <returns>shared_ptr of the Value.</returns>
	static std::shared_ptr<Value> create(float data);
	static std::shared_ptr<Value> add(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
	static std::shared_ptr<Value> multiply(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
	static std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);

private:
	// Data/parameter stored in this node
	float m_data;
	// Derivative of the loss w.r.t. this node
	float m_grad = 0;
	// Children of this node in the computation graph
	std::optional<std::vector<std::shared_ptr<Value>>> m_children;
	// Local derivative of this node w.r.t. its children in the computation graph
	std::optional<std::vector<float>> m_local_grads;
};

#endif VALUE_H