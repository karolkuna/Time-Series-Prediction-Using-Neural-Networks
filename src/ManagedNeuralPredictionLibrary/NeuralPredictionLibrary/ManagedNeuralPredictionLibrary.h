//
//  ManagedNeuralPredictionLibrary.h
//  ManagedNeuralPredictionLibrary
//
//  Copyright 2015 Karol Kuna
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
//

#pragma once
#include "../../NeuralPredictionLibrary/MemoryBlock.h"
#include "../../NeuralPredictionLibrary/ActivationFunctions.h"
#include "../../NeuralPredictionLibrary/NeuralNetwork.h"
#include "../../NeuralPredictionLibrary/FeedForwardNetwork.h"
#include "../../NeuralPredictionLibrary/SimpleRecurrentNetwork.h"
#include "../../NeuralPredictionLibrary/CWRecurrentNetwork.h"
#include "../../NeuralPredictionLibrary/Backpropagation.h"
#include "../../NeuralPredictionLibrary/TBPTT.h"
#include "../../NeuralPredictionLibrary/RTRL.h"

using namespace System;
namespace ManagedNeuralPredictionLibrary {
   
	public ref class ManagedMemoryBlock  {
 
	public:
		MemoryBlock* nativeMemoryBlock;

		ManagedMemoryBlock () {
			nativeMemoryBlock = new MemoryBlock();
		};
	
		ManagedMemoryBlock(int size) {
			nativeMemoryBlock = new MemoryBlock(size);
		};

		void Fill(float value) {
			nativeMemoryBlock->Fill(value);
		}

		void GenerateNormal(float mean, float stdev) {
			nativeMemoryBlock->GenerateNormal(mean, stdev);
		}
		void GenerateUniform(float min, float max) {
			nativeMemoryBlock->GenerateUniform(min, max);
		}
	
		float Sum() {
			return nativeMemoryBlock->Sum();
		}

		float SquareSum() {
			return nativeMemoryBlock->SquareSum();
		}
	
		void Add(ManagedMemoryBlock^ operand) {
			nativeMemoryBlock->Add(*(operand->nativeMemoryBlock));
		}
		
		void Add(float operand) {
			nativeMemoryBlock->Add(operand);
		}

		void Subtract(ManagedMemoryBlock^ operand) {
			nativeMemoryBlock->Subtract(*(operand->nativeMemoryBlock));
		}

		void Subtract(float operand) {
			nativeMemoryBlock->Subtract(operand);
		}

		void Multiply(const MemoryBlock& operand) {
			nativeMemoryBlock->Multiply(operand);
		}

		void Multiply(float operand) {
			nativeMemoryBlock->Multiply(operand);
		}
	
		void LeftShift(unsigned int shift) {
			nativeMemoryBlock->LeftShift(shift);
		}

		void RightShift(unsigned int shift) {
			nativeMemoryBlock->RightShift(shift);
		}
	
		void CopyTo(MemoryBlock& target) {
			nativeMemoryBlock->CopyTo(target);
		}

		void CopyTo(ManagedMemoryBlock^ target, int sourceOffset, int targetOffset, int count) {
			nativeMemoryBlock->CopyTo(*(target->nativeMemoryBlock), sourceOffset, targetOffset, count);
		}

		void Print() {
			nativeMemoryBlock->Print();
		}

		float GetValue(int id) {
			return nativeMemoryBlock->data[id];
		}

		void SetValue(int id, float value) {
			nativeMemoryBlock->data[id] = value;
		}
	};

	public ref class ManagedActivationFunction {
	public:
		ActivationFunction* activationFunction;

		float Activate(float value) { //functor isnt working
			return (*activationFunction)(value);
		}
		float Derivative(float value) {
			return activationFunction->Derivative(value);
		}
	};

	public ref class ManagedLogisticFunction : public ManagedActivationFunction {
	public:
		ManagedLogisticFunction() {
			activationFunction = new LogisticFunction();
		}
		
		~ManagedLogisticFunction() {
			delete activationFunction;
		}
	};

	public ref class ManagedFeedforwardNetwork {
	public:
		FeedforwardNetwork* nativeNetwork;

		ManagedFeedforwardNetwork(int inputUnits, const vector<int>& hiddenLayerUnits, int outputUnits, ManagedActivationFunction^ managedActivationFunction, float learningRate, float momentumRate) {
			nativeNetwork = new FeedforwardNetwork(inputUnits, hiddenLayerUnits, outputUnits, managedActivationFunction->activationFunction, learningRate, momentumRate);
		}

		~ManagedFeedforwardNetwork() {
			delete nativeNetwork;
		}
	
		void Propagate(ManagedMemoryBlock^ input) {
			nativeNetwork->Propagate(*(input->nativeMemoryBlock));
		}

		float GetOutputValue(int id) {
			return nativeNetwork->output.data[id];
		}
	};

	public ref class ManagedSimpleRecurrentNetwork {
	public:
		SimpleRecurrentNetwork* nativeNetwork;
		ManagedSimpleRecurrentNetwork(int inputUnits, int hiddenUnits, int outputUnits, ManagedActivationFunction^ managedActivationFunction) {
			nativeNetwork = new SimpleRecurrentNetwork(inputUnits, hiddenUnits, outputUnits, managedActivationFunction->activationFunction);
		}

		~ManagedSimpleRecurrentNetwork() {
			delete nativeNetwork;
		}
	
		void Propagate(ManagedMemoryBlock^ input) {
			nativeNetwork->Propagate(*(input->nativeMemoryBlock));
		}

		float GetOutputValue(int id) {
			return nativeNetwork->output.data[id];
		}
	};

	public ref class ManagedCWRecurrentNetwork {
	public:
		CWRecurrentNetwork* nativeNetwork;
		ManagedCWRecurrentNetwork(int inputUnits, int hiddenModuleUnits, int outputUnits, const vector<int>& modulesClockRate, ManagedActivationFunction^ managedActivationFunction) {
			nativeNetwork = new CWRecurrentNetwork(inputUnits, hiddenModuleUnits, outputUnits, modulesClockRate, managedActivationFunction->activationFunction);
		}

		~ManagedCWRecurrentNetwork() {
			delete nativeNetwork;
		}
	
		void Propagate(ManagedMemoryBlock^ input) {
			nativeNetwork->Propagate(*(input->nativeMemoryBlock));
		}

		float GetOutputValue(int id) {
			return nativeNetwork->output.data[id];
		}
	};

	public ref class ManagedBackpropagation {
	private:
		Backpropagation* m_nativeLearning;
	public:
		float error;

		ManagedBackpropagation(ManagedFeedforwardNetwork^ network) {
			m_nativeLearning = new Backpropagation(network->nativeNetwork);
		}

		~ManagedBackpropagation() {
			delete m_nativeLearning;
		}

		void Train(ManagedMemoryBlock^ target) {
			m_nativeLearning->Train(*(target->nativeMemoryBlock));
			error = m_nativeLearning->error;
		}
	};

	public ref class ManagedTBPTT {
	private:
		TBPTT* m_nativeLearning;
	public:
		float error;

		ManagedTBPTT(ManagedCWRecurrentNetwork^ network, float learningRate, float momentumRate, int depth) {
			m_nativeLearning = new TBPTT(network->nativeNetwork, learningRate, momentumRate, depth);
		}

		ManagedTBPTT(ManagedSimpleRecurrentNetwork^ network, float learningRate, float momentumRate, int depth) {
			m_nativeLearning = new TBPTT(network->nativeNetwork, learningRate, momentumRate, depth);
		}

		~ManagedTBPTT() {
			delete m_nativeLearning;
		}

		void Train(ManagedMemoryBlock^ target) {
			m_nativeLearning->Train(*(target->nativeMemoryBlock));
			error = m_nativeLearning->error;
		}
	};

	public ref class ManagedRTRL {
	private:
		RTRL* m_nativeLearning;
	public:
		float error;

		ManagedRTRL(ManagedSimpleRecurrentNetwork^ network, float learningRate, float momentumRate) {
			m_nativeLearning = new RTRL(network->nativeNetwork, learningRate, momentumRate);
		}

		~ManagedRTRL() {
			delete m_nativeLearning;
		}

		void Train(ManagedMemoryBlock^ target) {
			m_nativeLearning->Train(*(target->nativeMemoryBlock));
			error = m_nativeLearning->error;
		}
	};
}
