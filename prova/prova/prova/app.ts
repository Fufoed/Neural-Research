function SigmoidDerivative(number): number
{
    return number * (1 - number);
}

function Sigmoid(number): number
{
    return (1 / (1 + Math.exp(-number)));
}

class Neuron
{
    value: number = 0;
    weights: Array<any> = [];

    populate(WeightsNumber)
    {
        this.weights = [];
        for (let i = 0; i < WeightsNumber; i++)
        {
            this.weights.push(Math.random() * 2 - 1);
        }
    }
}

class Layer
{
    id;
    neurons: Array<any> = [];
    constructor(index)
    {
        this.id = index || 0;
    }

    populate(NeuronsNumber, InputsNumber)
    {
        this.neurons = [];
        for (let i = 0; i < NeuronsNumber; i++)
        {
            var neuron = new Neuron();
            neuron.populate(InputsNumber);
            this.neurons.push(neuron);
        }
    }
}

class Network
{
    layers: Array<any> = [];
    error: any;

    PerceptronGen(input, hiddens, output)
    {
        var index = 0;
        var previousNeurons = 0;
        var head = new Layer(index);
        head.populate(input, previousNeurons);
        previousNeurons = input;
        this.layers.push(head);
        index++;
        for (var i in hiddens) {
            var body = new Layer(index);
            body.populate(hiddens[i], previousNeurons);
            previousNeurons = hiddens[i];
            this.layers.push(body);
            index++;
        }
        var tail = new Layer(index);
        tail.populate(output, previousNeurons);
        this.layers.push(tail);
        return this.layers
    }

    Compute(inputs, outputs): void
    {
        for (let i in inputs) {
            if (this.layers[0] && this.layers[0].neurons[i]) {
                this.layers[0].neurons[i].value = inputs[i];
            }
        }
        var prevLayer = this.layers[0];
        for (let i = 1; i < this.layers.length; i++) {
            for (let j in this.layers[i].neurons) {
                var sum = 0;
                for (let k in prevLayer.neurons) {
                    sum += prevLayer.neurons[k].value * this.layers[i].neurons[j].weights[k];
                    console.log("Sum: " + sum);
                }
                this.layers[i].neurons[j].value = Sigmoid(sum);
                console.log("\nSig: " + this.layers[i].neurons[j].value + '\n')
            }
            prevLayer = this.layers[i];
        }

        var out: Array<any> = [];
        var lastLayer = this.layers[this.layers.length - 1];
        for (let i in lastLayer.neurons) {
            out.push(lastLayer.neurons[i].value);
        }
        console.log(out);

    }
}

var net = new Network();
var x = net.PerceptronGen(3, [5], 3);
for (var i in x)
{
    console.log(x[i]);
}