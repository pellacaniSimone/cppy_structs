#/**
# * @file Makefile
# * @author Simone Pellacani (pellacanisimone2017@gmail.com)
# * @brief Cpp makefile
# * @version 0.1
# * @date 2024-11-08
# * 
# * @copyright Copyright (c) 2024
# * 
# */


CXX = g++

CXXFLAGS =  -Wall -Werror -Wextra -pedantic -std=c++17 -g -fsanitize=address -DSTRICT_WARNINGS

LDFLAGS =  -fsanitize=address -DSTRICT_WARNINGS

OBJDIR = oggetto/

SRC = btest.cpp tipo.cpp  bnode.cpp  btree.cpp

OBJ = $(SRC:.c=.o)

EXEC = btest.out

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) $(CXXFLAGS)


clean:
	rm -rf $(OBJ) $(EXEC)