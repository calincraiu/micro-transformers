#include <string>
#include <cmath>

#include "core/Value.h"


Value::Value(float data) :
	m_data(data), m_children(std::nullopt), m_local_grads(std::nullopt)
{
}

Value::Value(float data, std::vector<std::shared_ptr<Value>> children) :
	m_data(data), m_children(std::move(children)), m_local_grads(std::nullopt)
{
}

void Value::print()
{
    std::printf(
        "Value(Data: %s ; Grad: %s ; Children: %i)\n", 
        std::to_string(m_data).c_str(), 
        std::to_string(m_grad).c_str(),
        num_children()
    );
}

int Value::num_children()
{
    return m_children.has_value() ? m_children.value().size() : 0;
}

std::shared_ptr<Value> Value::create(float data)
{
    return std::make_shared<Value>(data);
}

std::shared_ptr<Value> Value::add(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b)
{
    std::shared_ptr<Value> out = std::make_shared<Value>(
        a->m_data + b->m_data,
        std::vector<std::shared_ptr<Value>>{ a, b }
    );

    out->m_local_grads = std::vector<float>{ 
        1.0f, // w.r.t a
        1.0f, // w.r.t b
    };
    return out;
}

std::shared_ptr<Value> Value::multiply(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b)
{
    std::shared_ptr<Value> out = std::make_shared<Value>(
        a->m_data * b->m_data,
        std::vector<std::shared_ptr<Value>>{ a, b }
    );

    out->m_local_grads = std::vector<float>{ 
        b->m_data, // w.r.t a
        a->m_data, // w.r.t b
    };
    return out;
}

std::shared_ptr<Value> Value::pow(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b)
{
    std::shared_ptr<Value> out = std::make_shared<Value>(
        std::pow(a->m_data, b->m_data),
        std::vector<std::shared_ptr<Value>>{ a, b }
    );

    out->m_local_grads = std::vector<float>{ 
        b->m_data * std::pow(a->m_data, b->m_data - 1.0f), // w.r.t a
        a->m_data * std::pow(b->m_data, a->m_data - 1.0f), // w.r.t b
    };
    return out;
}