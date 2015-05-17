//
//  NeuralLayer.cpp
//  NeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#include "NeuralLayer.h"

using std::vector;

NeuralLayer::NeuralLayer(int units, ActivationFunction* activationFunction, float learningRate, float momentumRate) {
	this->units = units;
	this->activationFunction = activationFunction;
	this->learningRate = learningRate;
	this->momentumRate = momentumRate;
	
	this->activation = MemoryBlock(units);
	this->activationDerivative = MemoryBlock(units);
	this->delta = MemoryBlock(units);
	
	this->activation.Fill(0);
	this->activationDerivative.Fill(0);
	this->delta.Fill(0);
	
	this->inputUnits = 0;
}

void NeuralLayer::ProjectTo(NeuralLayer* destinationLayer) {
	if (destinationLayer == NULL) {
		throw std::logic_error("Invalid destination layer!");
	}
	if (this == destinationLayer) {
		throw std::logic_error("Recurrent connections are not allowed!");
	}
	
	//update lists of connections
	this->connectionsOut.push_back(destinationLayer);
	destinationLayer->connectionsIn.push_back(this);
	
	//weights are stored in one memory block, therefore offsets have to be stored
	this->outputWeightsOffsets.push_back(destinationLayer->inputUnits);
	destinationLayer->inputWeightsOffsets.push_back(destinationLayer->inputUnits);
	destinationLayer->inputUnits += this->units;
	
	//expand weights memory block
	destinationLayer->weights = MemoryBlock(destinationLayer->weights.size + this->units * destinationLayer->units);
	destinationLayer->weightsDelta = MemoryBlock(destinationLayer->weightsDelta.size + this->units * destinationLayer->units);
	
	//generate random weights
	destinationLayer->weights.GenerateUniform(-0.25, 0.25);
	destinationLayer->weightsDelta.Fill(0);
}

void NeuralLayer::SetActivation(const MemoryBlock& input) {
	if (input.size != activation.size) {
		throw std::logic_error("Invalid input size!");
	}
	
	for (int i = 0; i < units; i++) {
		activation.data[i] = input.data[i];
	}
}

void NeuralLayer::GetActivation(MemoryBlock& output) {
	if (output.size != activation.size) {
		throw std::logic_error("Invalid output size!");
	}
	
	for (int i = 0; i < units; i++) {
		output.data[i] = activation.data[i];
	}
}

void NeuralLayer::PropagateForward() {
	//for every unit
	for (int to = 0; to < units; to++) {
		float weightedSum = 0.0f;
		
		//loop through all input layers
		for (int layer = 0; layer < connectionsIn.size(); layer++) {
			//loop through all their units
			for (int from = 0; from < connectionsIn[layer]->units; from++) {
				//calculate weighted sum of activations of input layer units
				weightedSum += connectionsIn[layer]->activation.data[from] * weights.data[to * inputUnits + inputWeightsOffsets[layer] + from];
			}
		}
		
		//apply activation function to the weighted sum
		activation.data[to] = (*activationFunction)(weightedSum);
		activationDerivative.data[to] = (*activationFunction).Derivative(weightedSum);
	}
}

void NeuralLayer::SetTarget(const MemoryBlock& target) {
	if (target.size != delta.size) {
		throw std::logic_error("Target doesn't match layer size!");
	}
	
	target.CopyTo(delta);
	delta.Subtract(activation);
	delta.Multiply(activationDerivative);
}

void NeuralLayer::PropagateBackward() {
	//loop through all units
	for (int from = 0; from < units; from++) {
		float weightedSum = 0.0f;
		
		//loop through all output layers
		for (int layer = 0; layer < connectionsOut.size(); layer++) {
			//loop through all their units
			for (int to = 0; to < connectionsOut[layer]->units; to++) {
				//calculate weighted sum of error of output layer units
				weightedSum += connectionsOut[layer]->delta.data[to] * connectionsOut[layer]->weights.data[to * connectionsOut[layer]->inputUnits + outputWeightsOffsets[layer] + from];
			}
		}
		
		delta.data[from] = activationDerivative.data[from] * weightedSum;
	}
	
	//std::cout << ' ' << error.SquareSum() << std::endl;
}

void NeuralLayer::UpdateWeights() {
	//loop through all input layers
	for (int layer = 0; layer < connectionsIn.size(); layer++) {
		//loop through all units
		for (int to = 0; to < units; to++) {
			//loop through all input layer units
			for (int from = 0; from < connectionsIn[layer]->units; from++) {
				//adjust connection weight
				int weightId = to * inputUnits + inputWeightsOffsets[layer] + from;
				weightsDelta.data[weightId] = momentumRate * weightsDelta.data[weightId] + learningRate * delta.data[to] * connectionsIn[layer]->activation.data[from];
				weights.data[weightId] += weightsDelta.data[weightId];
			}
		}
	}
}
