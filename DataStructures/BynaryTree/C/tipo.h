/**
 * @file tipo.h
 * @author Simone Pellacani (pellacanisimone2017@gmail.com)
 * @brief Tipo Class to manage node data type in generic mode
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef TIPO_INF_DT
#define TIPO_INF_DT

#include <stdio.h> 

typedef int tipo_inf;


int compare(tipo_inf  , tipo_inf   );
//void copy(tipo_inf& , const tipo_inf& );
void print(const tipo_inf *);


/*Extra*/
tipo_inf initTipo();

#endif