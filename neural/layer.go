package neural

type Layer struct {
	Neurons []NeuronUnit
	Length  int
}

func NewLayer(n int, p int) (l Layer) {
	l = Layer{
		Neurons: make([]NeuronUnit, n),
		Length:  n,
	}
	for i := 0; i < n; i++ {
		l.Neurons[i] = NewNeuron(p)
	}
	return
}
