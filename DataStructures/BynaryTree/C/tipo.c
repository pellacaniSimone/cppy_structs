/**
 * @file tipo.c
 * @author Simone Pellacani (pellacanisimone2017@gmail.com)
 * @brief Tipo Class to manage node data type in generic mode
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdio.h> 

#include "tipo.h"


int compare( tipo_inf  s1, tipo_inf   s2) { return s1- s2; }
//void copy(tipo_inf * dest,  tipo_inf * source) { dest= source; }
void print(const tipo_inf *inf) { printf("%d", *inf); }

//tipo_inf initTipo(){ 
//  tipo_inf a;
//  return a;
//}