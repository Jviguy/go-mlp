package neural

import "math/rand"

const (
	SCALING_FACTOR = 0.0000000000001
)

type NeuronUnit struct {
	Weights []float64
	Bias    float64
	Lrate   float64
	Value   float64 `json:"-"`
	Delta   float64 `json:"-"`
}

func NewNeuron(prev int) (n NeuronUnit) {
	n = NeuronUnit{
		Weights: make([]float64, prev),
	}
	n.Bias = rand.NormFloat64() * SCALING_FACTOR
	n.Lrate = rand.NormFloat64() * SCALING_FACTOR
	n.Value = rand.NormFloat64() * SCALING_FACTOR
	n.Delta = rand.NormFloat64() * SCALING_FACTOR

	for i := 0; i < prev; i++ {
		n.Weights[i] = rand.NormFloat64() * SCALING_FACTOR
	}
	return
}
