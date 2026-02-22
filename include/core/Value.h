#ifndef VALUE_H
#define VALUE_H

#include <memory>
#include "core/Node.h"
#include "core/Tape.h"


class Value {
public:
	// Constructors
	Value(Node* n, Tape* t);

	// Methods
	float get_data() const;
	float get_grad() const;
	Node* get_node() const;

	// Operators
	Value operator-() const;
	Value operator+(const Value& a) const;
	Value operator-(const Value& a) const;
	Value operator*(const Value& a) const;
	Value operator/(const Value& a) const;
	
	// Operations
	Value pow(const Value& a) const;
	Value pow(const float a) const;
	Value log() const;
	Value exp() const;
	Value relu() const;

private:
	Node* m_node;
	Tape* m_tape;
};

#endif // VALUE_H