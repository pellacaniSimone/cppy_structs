/**
 * @file tipo.cpp
 * @author Simone Pellacani (pellacanisimone2017@gmail.com)
 * @brief Tipo Class to manage node data type in generic mode
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "tipo.h"

Tipo::Tipo(){;}
Tipo::~Tipo(){;}

int  Tipo::compare(TipoInf s1, TipoInf s2) { return s1 - s2; }
int  Tipo::init() { return 0; }
void Tipo::print(const TipoInf& inf) { cout << inf; }



