package neural

import (
	"encoding/json"
	"fmt"
	"gonum.org/v1/plot/plotter"
	"io"
	"math"
	"os"
)

type transferFunction func(float64) float64

type Network struct {
	Layers             []Layer          `json:"layers"`
	LRate              float64          `json:"learning_rate"`
	Transfer           transferFunction `json:"-"`
	TransferDerivative transferFunction `json:"-"`
}

func NewNetwork(l []int, Lrate float64, tf, trd transferFunction) (n *Network) {
	n = &Network{
		Layers:             make([]Layer, len(l)),
		LRate:              Lrate,
		Transfer:           tf,
		TransferDerivative: trd,
	}
	for i := 0; i < len(l); i++ {
		if i == 0 {
			n.Layers[i] = NewLayer(l[i], 0)
		} else {
			n.Layers[i] = NewLayer(l[i], l[i-1])
		}
	}
	return
}

func (n *Network) Execute(s *Pattern) (r []float64) {
	// new value
	nv := 0.0

	// result of execution for each OUTPUT NeuronUnit in OUTPUT NeuralLayer
	r = make([]float64, n.Layers[len(n.Layers)-1].Length)

	// show pattern to network =>
	for i := 0; i < len(s.Features); i++ {

		// setup value of each neurons in first layers to respective features of pattern
		n.Layers[0].Neurons[i].Value = s.Features[i]

	}

	// execute - hiddens + output
	// for each layers from first hidden to output
	for k := 1; k < len(n.Layers); k++ {

		// for each neurons in focused level
		for i := 0; i < n.Layers[k].Length; i++ {

			// init new value
			nv = 0.0

			// for each neurons in previous level (for k = 1, INPUT)
			for j := 0; j < n.Layers[k-1].Length; j++ {

				// sum output value of previous neurons multiplied by weight between previous and focused neuron
				nv += n.Layers[k].Neurons[i].Weights[j] * n.Layers[k-1].Neurons[j].Value

			}

			// add neuron bias
			nv += n.Layers[k].Neurons[i].Bias

			// compute activation function to new output value
			n.Layers[k].Neurons[i].Value = n.Transfer(nv)

		}

	}

	// get output values
	for i := 0; i < n.Layers[len(n.Layers)-1].Length; i++ {
		// simply accumulate values of all neurons in last level
		r[i] = n.Layers[len(n.Layers)-1].Neurons[i].Value
	}

	return r
}

func (n *Network) BackPropagate(s *Pattern, o []float64) (r float64) {
	no := n.Execute(s)
	fmt.Println(no)
	fmt.Println(o)
	e := 0.0
	for i := 0; i < n.Layers[len(n.Layers)-1].Length; i++ {

		// compute error in output: output for given pattern - output computed by network
		e = o[i] - no[i]
		fmt.Println(e)
		// compute delta for each neuron in output layer as:
		// error in output * derivative of transfer function of network output
		n.Layers[len(n.Layers)-1].Neurons[i].Delta = e * n.TransferDerivative(no[i])

	}

	// backpropagate error to previous layers
	// for each layers starting from the last hidden (len(n.Layers)-2)
	for k := len(n.Layers) - 2; k >= 0; k-- {

		// compute actual layer errors and re-compute delta
		for i := 0; i < n.Layers[k].Length; i++ {

			// reset error accumulator
			e = 0.0

			// for each link to next layer
			for j := 0; j < n.Layers[k+1].Length; j++ {

				// sum delta value of next neurons multiplied by weight between focused neuron and all neurons in next level
				e += n.Layers[k+1].Neurons[j].Delta * n.Layers[k+1].Neurons[j].Weights[i]

			}

			// compute delta for each neuron in focused layer as error * derivative of transfer function
			n.Layers[k].Neurons[i].Delta = e * n.TransferDerivative(n.Layers[k].Neurons[i].Value)

		}

		// compute weights in the next layer
		// for each link to next layer
		for i := 0; i < n.Layers[k+1].Length; i++ {

			// for each neurons in actual level (for k = 0, INPUT)
			for j := 0; j < n.Layers[k].Length; j++ {

				// sum learning rate * next level next neuron Delta * actual level actual neuron output value
				n.Layers[k+1].Neurons[i].Weights[j] +=
					n.LRate * n.Layers[k+1].Neurons[i].Delta * n.Layers[k].Neurons[j].Value

			}

			// learning rate * next level next neuron Delta * actual level actual neuron output value
			n.Layers[k+1].Neurons[i].Bias += n.LRate * n.Layers[k+1].Neurons[i].Delta

		}

	}
	for i := 0; i < len(o); i++ {
		r += math.Abs(no[i] - o[i])
	}
	return r / float64(len(o))
}

func (n *Network) Train(patterns []Pattern, epochs int) {
	fmt.Println("Starting training...")
	for i := 0; i < epochs; i++ {
		deltasum := 0.0
		for j := 0; j < len(patterns); j++ {
			d := n.BackPropagate(&patterns[j], patterns[j].MultipleExpectation)
			deltasum += d
		}
		fmt.Printf("Finished epoch, average delta value of: %f \n", deltasum/float64(len(patterns)))
	}
}

func (n *Network) Test(patterns []Pattern) (plotter.XYs, plotter.XYs) {
	fmt.Println("Starting Testing...")
	expecteds := make(plotter.XYs, len(patterns))
	outputs := make(plotter.XYs, len(patterns))
	deltas := 0.0
	correct := 0
	for j := 0; j < len(patterns); j++ {
		no := n.Execute(&patterns[j])
		o := patterns[j].MultipleExpectation
		for i := 0; i < n.Layers[len(n.Layers)-1].Length; i++ {
			// compute error in output: output for given pattern - output computed by network
			deltas += o[i] - no[i]
		}
		opredict := convertVecToY(o)
		nopredict := convertVecToY(no)
		expecteds[j].X = float64(j)
		expecteds[j].Y = opredict
		outputs[j].X = float64(j)
		outputs[j].Y = nopredict
		if opredict == nopredict {
			correct++
		}
	}
	fmt.Println("Finished Testing...")
	fmt.Printf("Average delta value of: %f \n", deltas/float64(len(patterns)))
	fmt.Println("Accuracy: " + fmt.Sprintf("%f", float64(correct)/float64(len(patterns))*100) + "%\n")
	return expecteds, outputs
}

func convertVecToY(vec []float64) float64 {
	m := 0.0
	mi := 0
	for i, v := range vec {
		if v > m {
			m = v
			mi = i
		}
	}
	return float64(mi)
}

func (n *Network) Export(file *os.File) {
	data, err := json.Marshal(n)
	if err != nil {
		panic(err)
	}
	_, err = file.Write(data)
	if err != nil {
		panic(err)
	}
}

func ImportNetwork(file *os.File, transfer, transferD transferFunction) (n *Network) {
	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(data, &n)
	if err != nil {
		panic(err)
	}
	n.Transfer = transfer
	n.TransferDerivative = transferD
	return
}
