var math = require('mathjs');

var Neuron = function(MaxWeights) {
    //this.value = 0;
    this.weights = [];
    for (let i = 0; i < MaxWeights; i++) {
        this.weights.push(Math.random() * 2 - 1);
    }
    this.bias = Math.random() * 2 - 1;
    //this.adjust = 0;
}
Neuron.prototype.Process = function(inputs) {
    this.lastInputs = inputs;
    var sum = 0;
    for (let i = 0; i < inputs.length; i++) {
        sum += inputs[i] * this.weights[i];
    }
    sum += this.bias;
    return this.lastOutputs = new Functions().Sigmoid(sum);
}



var Layer = function(MaxNeurons, MaxInputs) {
    this.neurons = [];
    for (let i = 0; i < MaxNeurons; i++) {
        this.neurons[i] = new Neuron(MaxInputs);
    }
}

Layer.prototype.Process = function(inputs) {
    return this.neurons.map(function(neuron) {
        return neuron.Process(inputs);
    })
}



var Network = function() {
    this.network = [];
}
Network.prototype.Process = function(inputs) {
    var outputs;
    this.network.forEach(function(layer) {
        outputs = layer.Process(inputs);
        inputs = outputs;
    })
    return outputs;
}
Network.prototype.AddLayer = function(MaxNeurons, MaxInputs) {
    var index = 0;
    if (MaxInputs == null) {
        var prevLayer = this.network[this.network.length - 1];
        MaxInputs = prevLayer.neurons.length;
    }
    var layer = new Layer(MaxNeurons, MaxInputs);
    this.network.push(layer);
}
Network.prototype.Train = function(examples) {
    var outLayer = this.network[this.network.length - 1];
    var LearningRate = 0.3;
    var epochs = 10000;
    var Threshold = 0.00001;

    for (let i = 0; i < epochs; i++) {
        for (let j = 0; j < examples.length; j++) {
            var targets = examples[j][1];
            var inputs = examples[j][0];
            var outputs = this.Process(inputs);
            for (let k = 0; k < outLayer.neurons.length; k++) {
                var neuron = outLayer.neurons[k];
                neuron.error = targets[k] - outputs[k];
                neuron.adjust = new Functions().SigmoidDerivative(neuron.lastOutputs) * neuron.error;
            }

            for (let h = this.network.length - 2; h >= 0; h--) {
                for (let f = 0; f < this.network[h].neurons.length; f++) {
                    var neuron = this.network[h].neurons[f];
                    neuron.error = math.sum(this.network[h + 1].neurons.map(function(n) {
                        return n.weights[f] * n.adjust;
                    }));

                    neuron.adjust = new Functions().SigmoidDerivative(neuron.lastOutputs) * neuron.error;

                    for (let m = 0; m < this.network[h + 1].neurons.length; m++) {
                        var nextNeuron = this.network[h + 1].neurons[m];
                        for (let w = 0; w < nextNeuron.weights.length; w++) {
                            nextNeuron.weights[w] += LearningRate * nextNeuron.lastInputs[w] * nextNeuron.adjust;
                        }
                        nextNeuron.bias += LearningRate * nextNeuron.adjust;
                    }
                }
            }
        }
        var error = new Functions().Mse(outLayer.neurons.map(function(n) {
            return n.error;
        }));

        if (i % 10000 == 0) {
            console.log("Iteration : ", i, " error : ", error);
        }

        if (error < Threshold) {
            console.log("Stopped at iteration n°", i);
            return;
        }
    }
}
Network.prototype.Serialize = function() {
    return JSON.stringify(this);
}
Network.prototype.Deserialize = function(inputs, serial) {
        var serialData = JSON.parse(serial);
        if (serialData) {
            this.network.length = 0;
            this.AddLayer(serialData.network[0].neurons.length, inputs);

            for (let i = 1; i < serialData.network.length; i++) {
                this.AddLayer(serialData.network[i].neurons.length)
            }

            for (let i = 0; i < serialData.network.length; i++) {
                for (let j = 0; j < serialData.network[i].neurons.length; j++) {
                    this.network[i].neurons[j].bias = serialData.network[i].neurons[j].bias
                    this.network[i].neurons[j].error = serialData.network[i].neurons[j].error
                    this.network[i].neurons[j].lastOutput = serialData.network[i].neurons[j].lastOutput
                    this.network[i].neurons[j].adjust = serialData.network[i].neurons[j].adjust
                }

                for (var w = 0; w < serialData.network[i].neurons[j].weights.length; w++) {
                    this.network[i].neurons[j].weights[w] = serialData.network[i].neurons[j].weights[w];
                }

                this.network[i].neurons[j].lastInputs = [];
                for (var v = 0; v < serialData.network[i].neurons[j].weights.length; v++) {
                    this.network[i].neurons[j].lastInputs.push(serialData.network[i].neurons[j].lastInputs[v]);
                }
            }
        } else {
            return false;
        }
    }
    /*Network.prototype.Forwardpropagation = function(network, index) {
        var inputs = index;
        var Func = new Functions();
        for (let i = 0; i < network.length; i++) {
            NewInputs = [];
            for (let j = 0; j < network[i].neurons.length; j++) {
                activation = Func.Activation(network[i].neurons[j].weights, inputs);
                network[i].neurons[j].value = Func.Sigmoid(activation);
                NewInputs.push(network[i].neurons[j].value);
            }
            inputs = NewInputs;
        }
        return inputs;
    }
    Network.prototype.UpdateWeights = function(network, index, LearningRate, res) {
        for (let i = 0; i < network.length; i++) {
            for (let j = 0; j < network[i].neurons.length; j++) {
                for (let k = 0; k < res.length; k++) {
                    network[i].neurons[j].weights[k] += LearningRate * res[k];
                }
            }
        }
    }
    Network.prototype.Backpropagation = function(network, DesiredOut) {
        var Func = new Functions();
        var back = [];
        var error = [];
        for (let i = network.length - 1; i >= 0; i--) {
            for (let j = 0; j < network[i].neurons.length; j++) {

            }
        }
        /*for (let i = 0; i < out.length; i++) {
            error[i] = DesiredOut[i] - out[i];
            network[network.length - 1].neurons[i].adjust = error[i] * Func.SigmoidDerivative(out[i]);
            back[i] = network[network.length - 1].neurons[i].adjust;
        }
        return back;
    }
    Network.prototype.Train = function(network, train, LearningRate, epochs, outputs) {
        /*for (let epoch = 0; epoch < epochs; epoch++) {
            var ErrorSum = 0;
            for (let index = 0; index < train.length; index++) {
                var output = new Network().Forwardpropagation(network, train[index]);
                var expected = [];
                for (let i = 0; i < outputs; i++) {
                    expected[i] = 0;
                }
                expected[train[index][train[index].length - 1]] = 1;
                for (let i = 0; i < expected.length; i++) {
                    ErrorSum += math.sum([Math.pow((expected[i] - output[i]), 2)])
                }
                var res = new Network().Backpropagation(network, expected, output);
                new Network().UpdateWeights(network, train[index], LearningRate, res);
            }
            console.log("Epoch:" + epoch + "    LearningRate:" + LearningRate + "       Err:" + ErrorSum);
        }
    }*/



var Functions = function() {};
Functions.prototype.Sigmoid = function(activation) {
    return 1.0 / (1.0 + Math.exp(-activation));
}
Functions.prototype.Activation = function(weights, inputs) {
    var activation = 0;
    for (let i = 0; i < weights.length; i++) {
        activation += weights[i] * inputs[i];
    }
    return activation;
}
Functions.prototype.SigmoidDerivative = function(out) {
    return out * (1.0 - out);
}
Functions.prototype.Predict = function(network, row) {
    outputs = new Network().Forwardpropagation(network, row);
    return outputs.indexOf(Math.max(outputs));
}
Functions.prototype.Mse = function(errors) {
    var sum = errors.reduce(function(sum, i) {
        return sum + i * i;
    }, 0);
    return sum / errors.length;
}

var dataset = [
        [2.7810836, 2.550537003, 0],
        [1.465489372, 2.362125076, 0],
        [3.396561688, 4.400293529, 0],
        [1.38807019, 1.850220317, 0],
        [3.06407232, 3.005305973, 0],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [7.673756466, 3.508563011, 1]
    ]
    /*
    var network = new Network().InitializeNetwork(5, [5], 5);
    new Network().Train(network, dataset, 0.5, 1, 5);
    /*for (let row = 0; row < dataset.length; row++) {
        var prediction = new Functions().Predict(network, dataset[row]);
        console.log("Expected: " + dataset[row][dataset[row].length - 1] + "     predicted: " + prediction);
    }*/
net = new Network();
net.AddLayer(10, 20);
net.AddLayer(2);
var zero = [
    0, 1, 1, 0,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 0, 0, 1,
    0, 1, 1, 0
]

var one = [
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0,
    0, 0, 1, 0
]

var two = [
    0, 1, 1, 0,
    1, 0, 0, 1,
    0, 0, 1, 0,
    0, 1, 0, 0,
    1, 1, 1, 1
]

var three = [
    1, 1, 1, 1,
    0, 0, 0, 1,
    0, 1, 1, 1,
    0, 0, 0, 1,
    1, 1, 1, 1
]
var three_one = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1];

var three_two = [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1];

var three_three = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1];



var two_one = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1];

var two_two = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1];

var two_three = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1];



var one_one = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1];

var one_two = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0];

var one_three = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1];



var zero_one = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0];

var zero_two = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1];

var zero_three = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0];

net.Train([
    [zero, [0, 0]],

    [one, [0, 1]],

    [two, [1, 0]],

    [three, [1, 1]],

    [one_one, [0, 1]],

    [one_two, [0, 1]],

    [one_three, [0, 1]],

    [two_one, [1, 0]],

    [two_two, [1, 0]],

    [two_three, [1, 0]],

    [three_one, [1, 1]],

    [three_two, [1, 1]],

    [three_three, [1, 1]],

    [zero_one, [0, 0]],

    [zero_two, [0, 0]],

    [zero_three, [0, 0]]

])

var outputs = net.Process(two)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized : ", decimal, outputs)



var outputs = net.Process(three)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized :", decimal, outputs)



var outputs = net.Process(one)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized :", decimal, outputs)



console.log(net.Serialize());