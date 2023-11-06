package data

import (
	"go-preceptron/neural"
	"math/rand"
)

func TrainAndTest(patterns []neural.Pattern, testlen int) ([]neural.Pattern, []neural.Pattern) {
	train := make([]neural.Pattern, len(patterns)-testlen)
	test := make([]neural.Pattern, testlen)
	// sample 1000 random patterns for testing
	for i := 0; i < testlen; i++ {
		r := rand.Intn(len(patterns) - 1)
		test[i] = patterns[r]
		patterns = append(patterns[:r], patterns[r+1:]...)
	}
	for i, p := range patterns {
		train[i] = p
	}
	return train, test
}
