#include <iostream>
#include "core/Value.h"
#include "core/Tape.h"
#include "core/Node.h"


int main() {
    Tape tape;

    // Create leaf nodes via Tape
    Value x(tape.create_leaf(2.0f), &tape);
    Value y(tape.create_leaf(4.0f), &tape);
    Value z(tape.create_leaf(3.0f), &tape);

    // Build computation using Value operators
    Value f = (x.pow(2.0f) + y) * z;
    Value result = f.relu();

    // Loss
    Value expected(tape.create_leaf(20.0f), &tape); // Mock expected result
    Value error = expected - result; 
    Value loss = error * error; // mean squared error

    // Backprop
    tape.zero_grad();
    tape.backward(loss.get_node());

    // Output results
    std::cout << "loss = " << loss.get_data() << "\n";
    std::cout << "dloss/dx = " << x.get_grad() << "\n";
    std::cout << "dloss/dy = " << y.get_grad() << "\n";

    return 0;
}