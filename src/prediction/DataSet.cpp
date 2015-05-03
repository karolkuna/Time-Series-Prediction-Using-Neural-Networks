//
//  DataSet.cpp
//  prediction
//
//  Created by Karol Kuna on 02/05/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#include "DataSet.h"

DataSet::DataSet(const string& fileName, MemoryBlock* input, MemoryBlock* output) {
	m_input = input;
	m_output = output;
	
	m_file.open(fileName);
}

DataSet::DataSet(const string& fileName, MemoryBlock* output) {
	m_input = NULL;
	m_output = output;
	
	m_file.open(fileName);
}

DataSet::~DataSet() {
	m_file.close();
}

bool DataSet::Read() {
	if (!m_file) {
		return false;
	}
	
	if (m_input != NULL) {
		for (int i = 0; i < m_input->size; i++) {
			m_file >> m_input->data[i];
		}
	}
	for (int i = 0; i < m_output->size; i++) {
		m_file >> m_output->data[i];
	}
	
	return true;
}
