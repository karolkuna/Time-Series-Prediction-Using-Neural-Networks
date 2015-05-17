//
//  Backpropagation.cpp
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
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
