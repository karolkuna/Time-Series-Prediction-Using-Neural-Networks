//
//  TBPTT.cpp
//  prediction
//
//  Created by Karol Kuna on 20/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "TBPTT.h"

void TBPTT::UpdateUnfoldedWeights() {
	for (int i = 0; i < depth; i++) {
		network->outputLayer->weights.CopyTo(m_unfoldedOutputLayers[i]->weights);
		for (int j = 0; j < network->modules; j++) {
			network->hiddenLayerModules[j]->weights.CopyTo(m_unfoldedHiddenLayerModules[i][j]->weights);
		}
	}
}

TBPTT::TBPTT(float learningRate, float momentumRate, int depth) {
	this->learningRate = learningRate;
	this->momentumRate = momentumRate;
	this->depth = depth;
}

TBPTT::~TBPTT() {
	for (int i = 0; i < m_unfoldedInputLayers.size(); i++) {
		delete m_unfoldedInputLayers[i];
	}
	for (int i = 0; i < m_unfoldedHiddenLayerModules.size(); i++) {
		for (int j = 0; j < m_unfoldedHiddenLayerModules[i].size(); j++) {
			delete m_unfoldedHiddenLayerModules[i][j];
		}
	}
	for (int i = 0; i < m_unfoldedOutputLayers.size(); i++) {
		delete m_unfoldedOutputLayers[i];
	}
	delete m_unfoldedThresholdLayer;
}

void TBPTT::Init(CWRecurrentNetwork* network) {
	this->network = network;
	
	//create unfolded network
	for (int i = 0; i < depth; i++) {
		//momentum is 0 in unfolded copies
		m_unfoldedInputLayers.push_back(new NeuralLayer(network->inputUnits, network->activationFunction, learningRate, 0));
		m_unfoldedOutputLayers.push_back(new NeuralLayer(network->outputUnits, network->activationFunction, learningRate, 0));
	}
	
	m_unfoldedHiddenLayerModules = std::vector< std::vector<NeuralLayer*> >(depth + 1);
	for (int i = 0; i < depth + 1; i++) {
		for (int j = 0; j < network->modules; j++) {
			m_unfoldedHiddenLayerModules[i].push_back(new NeuralLayer(network->hiddenUnits, network->activationFunction, learningRate, 0));
		}
	}
	
	m_unfoldedThresholdLayer = new NeuralLayer(1, NULL, 0, 0);
	m_unfoldedThresholdLayer->activation.data[0] = 1;
	
	//connect the unfolded network
	for (int i = 0; i < depth; i++) {
		m_unfoldedThresholdLayer->ConnectTo(m_unfoldedOutputLayers[i]);
		
		for (int j = 0; j < network->modules; j++) {
			m_unfoldedThresholdLayer->ConnectTo(m_unfoldedHiddenLayerModules[i][j]);
			m_unfoldedInputLayers[i]->ConnectTo(m_unfoldedHiddenLayerModules[i][j]);
			for (int k = j; k < network->modules; k++) {
				m_unfoldedHiddenLayerModules[i + 1][k]->ConnectTo(m_unfoldedHiddenLayerModules[i][j]);
			}
			m_unfoldedHiddenLayerModules[i][j]->ConnectTo(m_unfoldedOutputLayers[i]);
		}
	}
	
	for (int i = 0; i < depth; i++) {
		m_unfoldedTargets.push_back(MemoryBlock(network->outputUnits));
	}
	
	UpdateUnfoldedWeights();
}

void TBPTT::Train(MemoryBlock& target) {
	//shift state of the network one step deeper (to the past)
	for (int i = depth - 2; i >= 0; i--) {
		m_unfoldedInputLayers[i]->activation.CopyTo(m_unfoldedInputLayers[i + 1]->activation);
		m_unfoldedTargets[i].CopyTo(m_unfoldedTargets[i+1]);
	}
	for (int i = 0; i < network->modules; i++) {
		m_unfoldedHiddenLayerModules[depth - 1][i]->activation.CopyTo(m_unfoldedHiddenLayerModules[depth][i]->activation);
	}
	
	//copy current state of the network to the most recent copy
	network->inputLayer->activation.CopyTo(m_unfoldedInputLayers[0]->activation);
	target.CopyTo(m_unfoldedTargets[0]);
	
	if (network->step <= depth) {
		//unfolded network is not filled yet
		return;
	}
	
	//forward pass in the unfolded network (it's needed because the weights are changed inbetween copies)
	for (int i = depth - 1; i >= 0; i--) {
		int localStep = network->step - i;
		for (int j = 0; j < network->modules; j++) {
			if (localStep % network->modulesClockRate[j] == 0) {
				m_unfoldedHiddenLayerModules[i][j]->PropagateForward();
			}
		}
		m_unfoldedOutputLayers[i]->PropagateForward();
	}
	
	//backpropagation in the unfolded network
	for (int i = 0; i < depth; i++) {
		m_unfoldedOutputLayers[i]->SetTarget(m_unfoldedTargets[i]);
	}
	for (int i = 0; i < depth; i++) {
		int localStep = network->step - i;
		for (int j = 0; j < network->modules; j++) {
			if (localStep % network->modulesClockRate[j] == 0) {
				m_unfoldedHiddenLayerModules[i][j]->PropagateBackward();
			}
		}
	}
	
	//weights momentum
	network->outputLayer->weightsDelta.Multiply(momentumRate);
	for (int i = 0; i < network->modules; i++) {
		network->hiddenLayerModules[i]->weightsDelta.Multiply(momentumRate);
	}
	
	//sum weight changes in unfolded copies
	for (int i = 0; i < depth; i++) {
		m_unfoldedOutputLayers[i]->UpdateWeights();
		network->outputLayer->weightsDelta.Add(m_unfoldedOutputLayers[i]->weightsDelta);
		for (int j = 0; j < network->modules; j++) {
			m_unfoldedHiddenLayerModules[i][j]->UpdateWeights();
			network->hiddenLayerModules[j]->weightsDelta.Add(m_unfoldedHiddenLayerModules[i][j]->weightsDelta);
		}
	}
	
	//update weights in the actual network
	network->outputLayer->weights.Add(network->outputLayer->weightsDelta);
	for (int i = 0; i < network->modules; i++) {
		network->hiddenLayerModules[i]->weights.Add(network->hiddenLayerModules[i]->weightsDelta);
	}
	
	//change weights in the unfolded network to the new ones
	UpdateUnfoldedWeights();
}
