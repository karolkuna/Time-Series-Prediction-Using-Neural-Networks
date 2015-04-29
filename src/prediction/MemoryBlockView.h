//
//  MemoryBlockView.h
//  prediction
//
//  Created by Karol Kuna on 29/04/15.
//  Copyright (c) 2015 Karol Kuna. All rights reserved.
//

#ifndef __prediction__MemoryBlockView__
#define __prediction__MemoryBlockView__

#include <stdio.h>
#include "MemoryBlock.h"

class MemoryBlockView : public MemoryBlock {
private:
	int m_from;
	int m_size;
public:
	MemoryBlockView(MemoryBlock& block, int from, int size);
	~MemoryBlockView() { data = NULL; };
};

#endif /* defined(__prediction__MemoryBlockView__) */
