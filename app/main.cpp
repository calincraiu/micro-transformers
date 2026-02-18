#include "core/Value.h"

int main() {

    std::shared_ptr<Value> x1 = Value::create(1.5f);
    std::shared_ptr<Value> x2 = Value::create(2.0f);
    std::shared_ptr<Value> b = Value::create(5.0f);

    std::shared_ptr<Value> y = Value::multiply(x1, x2);
    std::shared_ptr<Value> yb = Value::add(y, b);

    std::shared_ptr<Value> powTest = Value::pow(x2, b);

    std::shared_ptr<Value> logTest = Value::log(x2);
    std::shared_ptr<Value> expTest = Value::exp(x2);
    std::shared_ptr<Value> reluTest = Value::relu(x2);
    std::shared_ptr<Value> negTest = Value::neg(x2);

    yb->print();
    powTest->print();
    logTest->print();
    expTest->print();
    reluTest->print();
    negTest->print();
}