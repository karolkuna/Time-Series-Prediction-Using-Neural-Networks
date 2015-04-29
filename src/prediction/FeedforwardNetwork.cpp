//
//  FeedforwardNetwork.cpp
//  prediction
//
//  Created by Karol Kuna on 09/02/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "FeedforwardNetwork.h"

FeedforwardNetwork::FeedforwardNetwork(int inputUnits, std::vector<int> hiddenLayerUnits, int outputUnits, ActivationFunction* activationFunction, float learningRate, float momentumRate) {
	if (hiddenLayerUnits.size() == 0) {
		throw std::logic_error("At least one hidden layer is required");
	}
	
	this->activationFunction = activationFunction;
	
	output = MemoryBlock(outputUnits);
	
	//input layer
	layers.push_back(new NeuralLayer(inputUnits, activationFunction, learningRate, momentumRate));
	
	//hidden layers
	for (int i = 0; i < hiddenLayerUnits.size(); i++) {
		layers.push_back(new NeuralLayer(hiddenLayerUnits[i], activationFunction, learningRate, momentumRate));
	}
	
	//output layer
	layers.push_back(new NeuralLayer(outputUnits, activationFunction, learningRate, momentumRate));
	
	//threshold
	threshold = new NeuralLayer(1, NULL, 0, 0);
	threshold->activation.data[0] = 1;
	
	for (int i = 1; i < layers.size(); i++) {
		threshold->ConnectTo(layers[i]);
	}
	
	for (int i = 0; i < layers.size() - 1; i++) {
		layers[i]->ConnectTo(layers[i+1]);
	}
}

FeedforwardNetwork::~FeedforwardNetwork() {
	for (int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
	delete threshold;
}

void FeedforwardNetwork::Propagate(MemoryBlock& input) {
	//copy input to input layer activation
	layers.front()->SetActivation(input);
	
	//propagate signal through hidden layers
	for (int i = 1; i < layers.size(); i++) {
		layers[i]->PropagateForward();
	}
	
	//copy activation of last layer to output
	layers.back()->GetActivation(output);
}
