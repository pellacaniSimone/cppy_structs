/**
 * @file tipo.h
 * @author Simone Pellacani (pellacanisimone2017@gmail.com)
 * @brief  Tipo Class to manage node data type in generic mode
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef TIPO_HPP
#define TIPO_HPP

using namespace std;

#include <iostream>

using TipoInf = int;  // alias C++

class Tipo {
public:
    Tipo();
    ~Tipo();
    static int compare(TipoInf , TipoInf ) ;
    static int init();
    static void print(const TipoInf& ) ;
};

#endif
