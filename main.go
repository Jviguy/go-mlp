package main

import (
	"encoding/csv"
	"go-preceptron/data"
	"go-preceptron/neural"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"os"
)

func main() {
	f, err := os.Open("IRIS.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	if _, err := reader.Read(); err != nil {
		panic(err)
	}
	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}
	var net *neural.Network
	patterns := data.CSVRecordsToPatterns(records)
	train, test := data.TrainAndTest(patterns, 25)
	_, err = os.Stat("export.json")
	if err != nil {
		net = neural.NewNetwork([]int{4, 30, 15, 3}, .5, neural.Sigmoid, neural.SigmoidDerivative)
		net.Train(train, 10000)
		export, err := os.OpenFile("export.json", os.O_RDWR|os.O_CREATE, 0644)
		if err != nil {
			panic(err)
		}
		net.Export(export)
	} else {
		export, err := os.OpenFile("export.json", os.O_RDWR|os.O_CREATE, 0644)
		if err != nil {
			panic(err)
		}
		net = neural.ImportNetwork(export, neural.Sigmoid, neural.SigmoidDerivative)
		export.Close()
	}
	exp, out := net.Test(test)
	p := plot.New()
	p.Title.Text = "Testing Data"
	p.X.Label.Text = "Pattern"
	p.Y.Label.Text = "Flower type"
	// Make a scatter plotter and set its style.
	s, err := plotter.NewScatter(exp)
	s.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	if err != nil {
		panic(err)
	}
	p.Add(s)
	p.Legend.Add("Expected", s)
	s, err = plotter.NewScatter(out)
	s.GlyphStyle.Color = color.RGBA{B: 255, A: 255}
	if err != nil {
		panic(err)
	}
	p.Add(s)
	p.Legend.Add("Output", s)
	// Save the plot to a PNG file.
	if err := p.Save(10*vg.Inch, 10*vg.Inch, "testing.png"); err != nil {
		panic(err)
	}
}
