//
//  CWRecurrentNetwork.cpp
//  prediction
//
//  Created by Karol Kuna on 30/03/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "CWRecurrentNetwork.h"

CWRecurrentNetwork::CWRecurrentNetwork(int inputUnits, int hiddenModuleUnits, int outputUnits, const vector<int>& modulesClockRate, ActivationFunction* activationFunction) {
	this->step = -1;
	
	this->inputUnits = inputUnits;
	this->hiddenUnits = hiddenModuleUnits;
	this->outputUnits = outputUnits;
	this->modules = (int) modulesClockRate.size();
	
	this->activationFunction = activationFunction;
	
	inputLayer = new NeuralLayer(inputUnits, activationFunction, 0, 0);
	outputLayer = new NeuralLayer(outputUnits, activationFunction, 0, 0);
	thresholdLayer = new NeuralLayer(1, NULL, 0, 0);
	
	for (int i = 0; i < modules; i++) {
		hiddenLayerModules.push_back(new NeuralLayer(hiddenUnits, activationFunction, 0, 0));
		contextLayerModules.push_back(new NeuralLayer(hiddenUnits, activationFunction, 0, 0));
		this->modulesClockRate.push_back(modulesClockRate[i]);
	}
	
	thresholdLayer->activation.data[0] = 1;
	thresholdLayer->ProjectTo(outputLayer);
	
	for (int i = 0; i < modules; i++) {
		thresholdLayer->ProjectTo(hiddenLayerModules[i]);
		inputLayer->ProjectTo(hiddenLayerModules[i]);
		for (int j = i; j < modules; j++) { //recurrent connection + connections to slower modules
			contextLayerModules[j]->ProjectTo(hiddenLayerModules[i]);
		}
		hiddenLayerModules[i]->ProjectTo(outputLayer);
	}
	
	output = MemoryBlock(outputUnits);
}

CWRecurrentNetwork::~CWRecurrentNetwork() {
	delete inputLayer;
	delete outputLayer;
	delete thresholdLayer;
	
	for (int i = 0; i < hiddenLayerModules.size(); i++) {
		delete hiddenLayerModules[i];
		delete contextLayerModules[i];
	}
}

void CWRecurrentNetwork::Propagate(const MemoryBlock& input) {
	//step counter determining which modules to execute
	step += 1;
	
	inputLayer->SetActivation(input);
	
	for (int i = 0; i < modules; i++) {
		if (step % modulesClockRate[i] == 0) {
			hiddenLayerModules[i]->activation.CopyTo(contextLayerModules[i]->activation);
		}
	}
	
	for (int i = 0; i < modules; i++) {
		if (step % modulesClockRate[i] == 0) {
			hiddenLayerModules[i]->PropagateForward();
		}
	}
	outputLayer->PropagateForward();
	
	outputLayer->GetActivation(output);
}
