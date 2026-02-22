#ifndef NODE_H
#define NODE_H


enum class Op
{
	Leaf,
	Neg,
	Add,
	Sub,
	Mul,
	Div,
	Pow,
	PowConst,
	Log,
	Exp,
	Relu,
};

class Node {
public:
	float get_data() { return m_data; }
	float get_grad() { return m_grad; }
	float get_const() { return m_const; }
	Op get_op() { return m_op; }
	Node* get_left() { return m_left; }
	Node* get_right() { return m_right; }

	void set_data(float x) { m_data = x; }
	void set_grad(float x) { m_grad = x; }
	void set_const(float x) { m_const = x; }
	void set_op(Op op) { m_op = op; }
	void set_left(Node* node) { m_left = node; }
	void set_right(Node* node) { m_right = node; }

private:
	// Data/parameter stored in this node
	float m_data = 0.0f;
	// Loss gradient w.r.t this node
	float m_grad = 0.0f;
	// Const in case of operations that use const - like raising to the power of a const
	float m_const = 0.0f;
	// Children of this node in the computation graph
	Node* m_left = nullptr;
	Node* m_right = nullptr;
	// Operation that created this Node
	Op m_op = Op::Leaf;
};


#endif // NODE_H