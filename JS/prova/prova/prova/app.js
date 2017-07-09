var math = require('mathjs');

var Neuron = function(MaxWeights) {
    this.weights = [];
    for (let i = 0; i < MaxWeights; i++) {
        this.weights.push(Math.random() * 2 - 1);
    }
    this.bias = Math.random() * 2 - 1;
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
    var epochs = 100000;
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


net = new Network();
net.AddLayer(15, 20);
net.AddLayer(10, 20);
net.AddLayer(5, 20);
net.AddLayer(4);
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

var four = [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0];

var five = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1];

var ten = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

net.Train([
    [zero, [0, 0, 0, 0]],

    [one, [0, 0, 0, 1]],

    [two, [0, 0, 1, 0]],

    [three, [0, 0, 1, 1]],

    [four, [0, 1, 0, 0]],

    [five, [0, 1, 0, 1]],

    [ten, [1, 0, 1, 0]]

])

var outputs = net.Process(one)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized : ", decimal, outputs)



var outputs = net.Process(zero)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized :", decimal, outputs)



var outputs = net.Process(two)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized :", decimal, outputs)


var outputs = net.Process(four)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized :", decimal, outputs)


var outputs = net.Process(five)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized :", decimal, outputs)


var outputs = net.Process(ten)

var binary = outputs.map(function(v) { return Math.round(v) }).join("")

var decimal = parseInt(binary, 2)

console.log("Digit recognized :", decimal, outputs)


//console.log(net.Serialize());