//
//  Backpropagation.cpp
//  prediction
//
//  Created by Karol Kuna on 29/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "Backpropagation.h"

Backpropagation::Backpropagation(FeedforwardNetwork* network) {
	if (network == NULL) {
		throw std::logic_error("No network provided!");
	}
	
	m_network = network;
}

void Backpropagation::Train(const MemoryBlock& target) {
	m_network->layers.back()->SetTarget(target);
	//std::cout << "LB" << " "; layers.back()->delta.Print();
	//std::cout << "AD" << " "; layers.back()->activationDerivative.Print();
	
	for (int i = (int)m_network->layers.size() - 2; i >= 1; i--) {
		m_network->layers[i]->PropagateBackward();
		//std::cout << "L" << i << " "; layers[i]->delta.Print();
	}
	
	for (int i = 1; i < m_network->layers.size(); i++) {
		m_network->layers[i]->UpdateWeights();
	}
	
	error = 0;
	for (int i = 0; i < target.size; i++) {
		error += (target.data[i] - m_network->output.data[i]) * (target.data[i] - m_network->output.data[i]);
	}
}
