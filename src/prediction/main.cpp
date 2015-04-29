//
//  main.cpp
//  prediction
//
//  Created by Karol Kuna on 09/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include <iostream>
#include <assert.h>
#include "FeedforwardNetwork.h"
#include "CWRecurrentNetwork.h"
#include "SimpleRecurrentNetwork.h"
#include "TBPTT.h"
#include "RTRL.h"
#include "MemoryBlockView.h"

using std::cout;
using std::endl;

void activationFunctionTests() {
	ActivationFunction* logistic = new LogisticFunction();
	LogisticFunction logistic2;
	ActivationFunction& logistic3 = *logistic;
	
	assert((*logistic)(0.5) != 0);
	assert((*logistic)(0.5) == logistic2(0.5));
	assert((*logistic)(0.5) == logistic3(0.5));
}

void memoryBlockTests() {
	MemoryBlock a = MemoryBlock(5); assert(a.size == 5);
	a.Fill(1); assert(a.Sum() == 5);
	a.LeftShift(0); assert(a.Sum() == 5);
	a.RightShift(0); assert(a.Sum() == 5);
	a.RightShift(2); assert(a.data[0] == 0 && a.data[1] == 0 && a.data[2] == 1 && a.data[3] == 1 & a.data[4] == 1);
	a.LeftShift(2); assert(a.data[0] == 1 && a.data[1] == 1 && a.data[2] == 1 && a.data[3] == 0 & a.data[4] == 0);
}

void runAllTests() {
	activationFunctionTests();
	memoryBlockTests();
	std::cout << "All tests passed\n";
}

/*
class FeedforwardPrediction {
	FeedforwardNetwork* network;

	
	MemoryBlock slidingWindow;
	MemoryBlock inputToNetwork;
	
	int horizon;
	int slidingWindowSize;
	int inputSize;
	int outputSize;
	
	FeedforwardPrediction(FeedforwardNetwork* network, int inputSize, int outputSize, int slidingWindowSize, int horizon) {
		if (network == NULL) {
			throw std::logic_error("Network missing!");
		}
		if (horizon < 1) {
			throw std::logic_error("Prediction horizon must be >= 1!");
		}
		if (slidingWindowSize < 1) {
			throw std::logic_error("Sliding window size must be >= 1!");
		}
		if (inputSize < 0 || outputSize < 0) {
			throw std::logic_error("Input and output size must be >= 0!");
		}
		this->network = network;
		this->slidingWindowSize = slidingWindowSize;
		this->horizon = horizon;
		
		this->inputSize = inputSize;
		this->outputSize = outputSize;
		
		slidingWindow = MemoryBlock((inputSize + outputSize) * (slidingWindowSize + horizon));
		slidingWindow.Fill(0);
		inputToNetwork = MemoryBlock((inputSize + outputSize) * slidingWindowSize);
		
		if (network->layers.front()->units != inputToNetwork.size) {
			throw std::logic_error("Network's input layer dimension is invalid!");
		}
		if (network->layers.back()->units != outputSize) {
			throw std::logic_error("Network's output layer dimension is invalid!");
		}
	}
	
	void Predict(MemoryBlock& input, MemoryBlock& output, MemoryBlock& prediction) {
		if (input.size != inputSize || output.size != outputSize || prediction.size != outputSize) {
			throw std::logic_error("Invalid arguments passed!");
		}
		
		slidingWindow.RightShift(inputSize + outputSize);
		input.CopyTo(slidingWindow, 0, 0, inputSize);
		output.CopyTo(slidingWindow, 0, inputSize, outputSize);
		
		slidingWindow.CopyTo(inputToNetwork, 0, 0, inputToNetwork.size);
		network->Propagate(inputToNetwork);
		network->output.CopyTo(prediction);
		
		slidingWindow.CopyTo(inputToNetwork, horizon * (inputSize + outputSize), 0, inputToNetwork.size);
		network->Propagate(inputToNetwork);
		network->PropagateBackward(output);
	}
	
};
*/
 
int main(int argc, const char * argv[]) {	
	std::vector<int> layers;
	layers.push_back(128);
	
	LogisticFunction logistic;
	
	//FeedforwardNetwork network(5, layers, 1, &logistic, 0.01, 0.99);
	SimpleRecurrentNetwork network(1, 20, 1, &logistic);
	
	LearningAlgorithm* learning = new TBPTT(&network, 0.005, 0.999, 5);
	//LearningAlgorithm* learning = new RTRL(&network, 1, 0.0);
	
	
	MemoryBlock input(1);
	input.data[0] = 0;
	
	MemoryBlock target(1);
	target.data[0] = 0.7;
	
	for (int step = 0; step < 10000; step++) {
		input.data[0] = 0.5 + 0.3*sinf(step * 0.3);
		target.data[0] = 0.5 + 0.3*sinf((step+5) * 0.3);
		
		network.Propagate(input);
		learning->Train(target);		
		
		std::cout << "IN: "; input.Print();
		std::cout << "TAR: "; target.Print();
		std::cout << "OUT: "; network.output.Print();
		std::cout << std::endl;
	}
	
    return 0;
}
