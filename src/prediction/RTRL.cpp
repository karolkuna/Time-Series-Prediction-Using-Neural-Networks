//
//  RTRL.cpp
//  prediction
//
//  Created by Karol Kuna on 20/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "RTRL.h"

#define CURRENT_TIMESTEP 0

RTRL::RTRL(SimpleRecurrentNetwork* network, float learningRate, float momentumRate) {
	if (network == NULL) {
		throw std::logic_error("No network provided!");
	}
	
	m_network = network;
	m_learningRate = learningRate;
	m_momentumRate = momentumRate;
	
	m_firstHiddenUnit = 1 + network->inputUnits;
	m_firstOutputUnit = 1 + network->inputUnits + network->hiddenUnits;
	m_totalUnits = 1 + network->inputUnits + network->hiddenUnits + network->outputUnits;
	m_lastInputUnit = m_firstHiddenUnit - 1;
	m_lastHiddenUnit = m_firstOutputUnit - 1;
	m_lastOutputUnit = m_totalUnits - 1;
	
	rtrlDerivatives = MemoryBlock(m_totalUnits * network->weights);
	rtrlPastDerivatives = MemoryBlock(rtrlDerivatives.size);
	rtrlFutureDerivatives = MemoryBlock(rtrlDerivatives.size);
	
	previousOutput = MemoryBlock(network->output.size);
	network->output.CopyTo(previousOutput);
	
	rtrlDerivatives.Fill(0);
	rtrlPastDerivatives.Fill(0);
	rtrlFutureDerivatives.Fill(0);
}

RTRL::~RTRL() {

}

void RTRL::Train(MemoryBlock& target) {
	//shift weight derivatives in time
	rtrlDerivatives.CopyTo(rtrlPastDerivatives);
	rtrlFutureDerivatives.CopyTo(rtrlDerivatives);
	
	//loop through all hidden layer weights
	for (int to = m_firstHiddenUnit; to <= m_lastHiddenUnit; to++) {
		for (int from = 0; from <= m_lastHiddenUnit; from++) {
			CalculateDerivativesForWeight(from, to); //from threshold, input or hidden unit
		}
	}
	//loop through all output layer weights
	for (int to = m_firstOutputUnit; to <= m_lastOutputUnit; to++) {
		CalculateDerivativesForWeight(0, to); //from threshold unit
		for (int from = m_firstHiddenUnit; from <= m_lastHiddenUnit; from++) {
			CalculateDerivativesForWeight(from, to); //from hidden unit
		}
	}
	target.CopyTo(m_network->error);
	m_network->error.Subtract(m_network->output);
	
	UpdateWeights();
	
	m_network->output.CopyTo(previousOutput);
}

void RTRL::CalculateDerivativesForWeight(int weightFrom, int weightTo) {
	
	//loop through all output units
	for (int to = m_firstOutputUnit; to <= m_lastOutputUnit; to++) {
		float derivative = Weight(0, to) * WeightDerivative(weightFrom, weightTo, 0, CURRENT_TIMESTEP);
		//loop through all their input connections
		for (int from = m_firstHiddenUnit; from <= m_lastHiddenUnit; from++) {
			derivative += Weight(from, to) * WeightDerivative(weightFrom, weightTo, from, CURRENT_TIMESTEP);
		}
		
		float newWeightDerivative =
			m_network->outputLayer->activationDerivative.data[to - m_firstOutputUnit]
			* (derivative + (weightTo == to) * Activation(weightFrom, CURRENT_TIMESTEP - (IsRecurrentWeight(weightFrom, weightTo) ? 1 : 0)));
		
		SetWeightDerivative(weightFrom, weightTo, to, CURRENT_TIMESTEP + 1, newWeightDerivative);
	}
	
	//loop through all hidden units
	for (int to = m_firstHiddenUnit; to <= m_lastHiddenUnit; to++) {
		float derivative = 0;
		//loop through all their input connections
		for (int from = 0; from <= m_lastHiddenUnit; from++) {
			derivative += Weight(from, to) * WeightDerivative(weightFrom, weightTo, from, CURRENT_TIMESTEP - (IsRecurrentWeight(from, to) ? 1 : 0));
		}
		
		float newWeightDerivative =
			m_network->hiddenLayer->activationDerivative.data[to - m_firstHiddenUnit]
			* (derivative + (weightTo == to) * Activation(weightFrom, CURRENT_TIMESTEP - (IsRecurrentWeight(weightFrom, weightTo) ? 1 : 0)));
			
		SetWeightDerivative(weightFrom, weightTo, to, CURRENT_TIMESTEP + 1, newWeightDerivative);
	}
	
}

void RTRL::UpdateWeight(int from, int to) {
	float wDelta = 0;
	for (int o = 0; o < m_network->output.size; o++) {
		wDelta += m_network->error.data[o] * WeightDerivative(from, to, m_firstOutputUnit + o, CURRENT_TIMESTEP + 1);
	}
	
	wDelta += m_momentumRate * WeightDelta(from, to);
	SetWeightDelta(from, to, wDelta);
	SetWeight(from, to, Weight(from, to) + m_learningRate * wDelta);
}

void RTRL::UpdateWeights() {
	//loop through all hidden layer weights
	for (int to = m_firstHiddenUnit; to <= m_lastHiddenUnit; to++) {
		for (int from = 0; from <= m_lastHiddenUnit; from++) {
			UpdateWeight(from, to); //from threshold, input or hidden unit
		}
	}
	//loop through all output layer weights
	for (int to = m_firstOutputUnit; to <= m_lastOutputUnit; to++) {
		UpdateWeight(0, to); //from threshold unit
		for (int from = m_firstHiddenUnit; from <= m_lastHiddenUnit; from++) {
			UpdateWeight(from, to); //from hidden unit
		}
	}
}

LayerPointer RTRL::GetWeightPointer(int from, int to) {
	if (to < 1) { //to threshold unit
		throw std::logic_error("No weights to threshold layer!");
		
	} else if (to < m_firstHiddenUnit) { //to input unit
		throw std::logic_error("No weights to input layer!");
		
	} else if (to <= m_lastHiddenUnit) { //to hidden unit
		if (from >= m_firstOutputUnit) {
			throw std::logic_error("No weights from output to hidden layer");
		}
		int weightId = (to - m_firstHiddenUnit) * m_network->hiddenLayer->inputUnits + from;
		return LayerPointer(m_network->hiddenLayer, weightId);
		
	} else if (to <= m_lastOutputUnit) { //to output unit
		int weightId = (to - m_firstOutputUnit) * m_network->outputLayer->inputUnits;
		if (from < 1) { //threshold unit
			//weightId += 0;
		} else if (from >= m_firstHiddenUnit && from <= m_lastHiddenUnit) {
			weightId += from - m_network->inputLayer->units;
		} else {
			throw std::logic_error("Invalid from unit!");
		}
		
		return LayerPointer(m_network->outputLayer, weightId);
		
	} else {
		throw std::logic_error("Invalid to unit!");
	}
}

LayerPointer RTRL::GetUnitPointer(int unitId, int time) {
	if (unitId < 1) {
		return LayerPointer(m_network->thresholdLayer, unitId);
	} else if (unitId <= m_lastInputUnit) {
		return LayerPointer(m_network->inputLayer, unitId - 1);
	} else if (unitId <= m_lastHiddenUnit) {
		if (time == CURRENT_TIMESTEP) {
			return LayerPointer(m_network->hiddenLayer, unitId - m_firstHiddenUnit);
		} else if (time == CURRENT_TIMESTEP - 1) {
			return LayerPointer(m_network->contextLayer, unitId - m_firstHiddenUnit);
		} else {
			throw std::logic_error("Invalid time step!");
		}
	} else {
		return LayerPointer(m_network->outputLayer, unitId - m_firstOutputUnit);
	}
}

float RTRL::WeightDerivative(int from, int to, int unit, int time) {
	int weightId;
	LayerPointer lp = GetWeightPointer(from, to);
	if (lp.layer == m_network->hiddenLayer) {
		weightId = lp.offset;
	} else if (lp.layer == m_network->outputLayer) {
		weightId = m_network->hiddenWeights + lp.offset;
	} else {
		throw std::logic_error("Invalid layer!");
	}
	
	if (time == CURRENT_TIMESTEP - 1) {
		return rtrlPastDerivatives.data[weightId * m_totalUnits + unit];
	} else if (time == CURRENT_TIMESTEP) {
		return rtrlDerivatives.data[weightId * m_totalUnits + unit];
	} else if (time == CURRENT_TIMESTEP + 1) {
		return rtrlFutureDerivatives.data[weightId * m_totalUnits + unit];
	} else {
		throw std::logic_error("Invalid time step!");
	}
}

void RTRL::SetWeightDerivative(int from, int to, int unit, int time, float value) {
	int weightId;
	LayerPointer lp = GetWeightPointer(from, to);
	if (lp.layer == m_network->hiddenLayer) {
		weightId = lp.offset;
	} else if (lp.layer == m_network->outputLayer) {
		weightId = m_network->hiddenWeights + lp.offset;
	} else {
		throw std::logic_error("Invalid layer!");
	}
	if (time == CURRENT_TIMESTEP - 1) {
		rtrlPastDerivatives.data[weightId * m_totalUnits + unit] = value;
	} else if (time == CURRENT_TIMESTEP) {
		rtrlDerivatives.data[weightId * m_totalUnits + unit] = value;
	} else if (time == CURRENT_TIMESTEP + 1) {
		rtrlFutureDerivatives.data[weightId * m_totalUnits + unit] = value;
	} else {
		throw std::logic_error("Invalid time step!");
	}
}

float RTRL::Weight(int from, int to) {
	LayerPointer lp = GetWeightPointer(from, to);
	return lp.layer->weights.data[lp.offset];
}

float RTRL::WeightDelta(int from, int to) {
	LayerPointer lp = GetWeightPointer(from, to);
	return lp.layer->weightsDelta.data[lp.offset];
}


void RTRL::SetWeight(int from, int to, float value) {
	LayerPointer lp = GetWeightPointer(from, to);
	lp.layer->weights.data[lp.offset] = value;
}

void RTRL::SetWeightDelta(int from, int to, float value) {
	LayerPointer lp = GetWeightPointer(from, to);
	lp.layer->weightsDelta.data[lp.offset] = value;
}

float RTRL::Activation(int unitId, int time) {
	LayerPointer lp = GetUnitPointer(unitId, time);
	
	if (lp.layer == m_network->outputLayer && time == (CURRENT_TIMESTEP - 1)) {
		return previousOutput.data[lp.offset];
	}
	
	return lp.layer->activation.data[lp.offset];
}

void RTRL::SetActivation(int unitId, int time, float value) {
	LayerPointer lp = GetUnitPointer(unitId, time);
	lp.layer->activation.data[lp.offset] = value;
}

bool RTRL::IsRecurrentWeight(int from, int to) {
	if (from >= m_firstHiddenUnit && from <= m_lastHiddenUnit
		&& to >= m_firstHiddenUnit && to <= m_lastHiddenUnit) {
		return true;
	} else {
		return false;
	}
}
