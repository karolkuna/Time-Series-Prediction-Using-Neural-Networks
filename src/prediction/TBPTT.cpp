//
//  TBPTT.cpp
//  prediction
//
//  Created by Karol Kuna on 20/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "TBPTT.h"

TBPTT::TBPTT(CWRecurrentNetwork* network, float learningRate, float momentumRate, int depth) {
	if (network == NULL) {
		throw std::logic_error("No network provided!");
	}
	
	m_network = network;
	m_learningRate = learningRate;
	m_momentumRate = momentumRate;
	m_depth = depth;
	
	CreateUnfoldedNetwork();
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

void TBPTT::CreateUnfoldedNetwork() {
	for (int i = 0; i < m_depth; i++) {
		//momentum is 0 in unfolded copies
		m_unfoldedInputLayers.push_back(new NeuralLayer(m_network->inputUnits, m_network->activationFunction, m_learningRate, 0));
		m_unfoldedOutputLayers.push_back(new NeuralLayer(m_network->outputUnits, m_network->activationFunction, m_learningRate, 0));
	}
	
	m_unfoldedHiddenLayerModules = std::vector< std::vector<NeuralLayer*> >(m_depth + 1);
	for (int i = 0; i < m_depth + 1; i++) {
		for (int j = 0; j < m_network->modules; j++) {
			m_unfoldedHiddenLayerModules[i].push_back(new NeuralLayer(m_network->hiddenUnits, m_network->activationFunction, m_learningRate, 0));
		}
	}
	
	m_unfoldedThresholdLayer = new NeuralLayer(1, NULL, 0, 0);
	m_unfoldedThresholdLayer->activation.data[0] = 1;
	
	//connect the unfolded network
	for (int i = 0; i < m_depth; i++) {
		m_unfoldedThresholdLayer->ProjectTo(m_unfoldedOutputLayers[i]);
		
		for (int j = 0; j < m_network->modules; j++) {
			m_unfoldedThresholdLayer->ProjectTo(m_unfoldedHiddenLayerModules[i][j]);
			m_unfoldedInputLayers[i]->ProjectTo(m_unfoldedHiddenLayerModules[i][j]);
			for (int k = j; k < m_network->modules; k++) {
				m_unfoldedHiddenLayerModules[i + 1][k]->ProjectTo(m_unfoldedHiddenLayerModules[i][j]);
			}
			m_unfoldedHiddenLayerModules[i][j]->ProjectTo(m_unfoldedOutputLayers[i]);
		}
	}
	
	for (int i = 0; i < m_depth; i++) {
		m_unfoldedTargets.push_back(MemoryBlock(m_network->outputUnits));
	}
	
	UpdateUnfoldedWeights();
}

void TBPTT::UpdateUnfoldedWeights() {
	for (int i = 0; i < m_depth; i++) {
		m_network->outputLayer->weights.CopyTo(m_unfoldedOutputLayers[i]->weights);
		for (int j = 0; j < m_network->modules; j++) {
			m_network->hiddenLayerModules[j]->weights.CopyTo(m_unfoldedHiddenLayerModules[i][j]->weights);
		}
	}
}

void TBPTT::Train(const MemoryBlock& target) {
	//shift state of the network one step deeper (to the past)
	for (int i = m_depth - 2; i >= 0; i--) {
		m_unfoldedInputLayers[i]->activation.CopyTo(m_unfoldedInputLayers[i + 1]->activation);
		m_unfoldedTargets[i].CopyTo(m_unfoldedTargets[i+1]);
	}
	for (int i = 0; i < m_network->modules; i++) {
		m_unfoldedHiddenLayerModules[m_depth - 1][i]->activation.CopyTo(m_unfoldedHiddenLayerModules[m_depth][i]->activation);
	}
	
	//copy current state of the network to the most recent copy
	m_network->inputLayer->activation.CopyTo(m_unfoldedInputLayers[0]->activation);
	target.CopyTo(m_unfoldedTargets[0]);
	
	if (m_network->step <= m_depth) {
		//unfolded network is not filled yet
		return;
	}
	
	//forward pass in the unfolded network (it's needed because the weights are changed inbetween copies)
	for (int i = m_depth - 1; i >= 0; i--) {
		int localStep = m_network->step - i;
		for (int j = 0; j < m_network->modules; j++) {
			if (localStep % m_network->modulesClockRate[j] == 0) {
				m_unfoldedHiddenLayerModules[i][j]->PropagateForward();
			}
		}
		m_unfoldedOutputLayers[i]->PropagateForward();
	}
	
	//backpropagation in the unfolded network
	for (int i = 0; i < m_depth; i++) {
		m_unfoldedOutputLayers[i]->SetTarget(m_unfoldedTargets[i]);
	}
	for (int i = 0; i < m_depth; i++) {
		int localStep = m_network->step - i;
		for (int j = 0; j < m_network->modules; j++) {
			if (localStep % m_network->modulesClockRate[j] == 0) {
				m_unfoldedHiddenLayerModules[i][j]->PropagateBackward();
			}
		}
	}
	
	//weights momentum
	m_network->outputLayer->weightsDelta.Multiply(m_momentumRate);
	for (int i = 0; i < m_network->modules; i++) {
		m_network->hiddenLayerModules[i]->weightsDelta.Multiply(m_momentumRate);
	}
	
	//sum weight changes in unfolded copies
	for (int i = 0; i < m_depth; i++) {
		m_unfoldedOutputLayers[i]->UpdateWeights();
		m_network->outputLayer->weightsDelta.Add(m_unfoldedOutputLayers[i]->weightsDelta);
		for (int j = 0; j < m_network->modules; j++) {
			m_unfoldedHiddenLayerModules[i][j]->UpdateWeights();
			m_network->hiddenLayerModules[j]->weightsDelta.Add(m_unfoldedHiddenLayerModules[i][j]->weightsDelta);
		}
	}
	
	//update weights in the actual network
	m_network->outputLayer->weights.Add(m_network->outputLayer->weightsDelta);
	for (int i = 0; i < m_network->modules; i++) {
		m_network->hiddenLayerModules[i]->weights.Add(m_network->hiddenLayerModules[i]->weightsDelta);
	}
	
	//change weights in the unfolded network to the new ones
	UpdateUnfoldedWeights();
	
	error = 0;
	for (int i = 0; i < target.size; i++) {
		error += (target.data[i] - m_network->output.data[i]) * (target.data[i] - m_network->output.data[i]);
	}
}
