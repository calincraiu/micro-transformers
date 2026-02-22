#include <string>
#include <cmath>
#include <limits>

#include "core/Value.h"


Value::Value(Node* n, Tape* t) : m_node(n), m_tape(t)
{
}

float Value::get_data() const {
    return m_node->get_data();
}

float Value::get_grad() const {
    return m_node->get_grad();
}

Node* Value::get_node() const { 
    return m_node; 
}

Value Value::operator-() const
{
    return Value(
        m_tape->create_node(Op::Neg, m_node, nullptr),
        m_tape
    );
}

Value Value::operator+(const Value& a) const
{
    return Value(
        m_tape->create_node(Op::Add, m_node, a.m_node),
        m_tape
    );
}

Value Value::operator-(const Value& a) const
{
    return Value(
        m_tape->create_node(Op::Sub, m_node, a.m_node),
        m_tape
    );
}

Value Value::operator*(const Value& a) const
{
    return Value(
        m_tape->create_node(Op::Mul, m_node, a.m_node),
        m_tape
    );
}

Value Value::operator/(const Value& a) const
{
    return Value(
        m_tape->create_node(Op::Div, m_node, a.m_node),
        m_tape
    );
}

Value Value::pow(const Value& a) const
{
    return Value(
        m_tape->create_node(Op::Pow, m_node, a.m_node),
        m_tape
    );
}

Value Value::pow(const float a) const
{
    return Value(
        m_tape->create_node(Op::PowConst, m_node, nullptr, a),
        m_tape
    );
}

Value Value::log() const
{
    return Value(
        m_tape->create_node(Op::Log, m_node, nullptr),
        m_tape
    );
}

Value Value::exp() const
{
    return Value(
        m_tape->create_node(Op::Exp, m_node, nullptr),
        m_tape
    );
}

Value Value::relu() const
{
    return Value(
        m_tape->create_node(Op::Relu, m_node, nullptr),
        m_tape
    );
}