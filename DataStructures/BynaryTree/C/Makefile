#/**
# * @file Makefile
# * @author Simone Pellacani (pellacanisimone2017@gmail.com)
# * @brief C makefile
# * @version 0.1
# * @date 2024-11-08
# * 
# * @copyright Copyright (c) 2024
# * 
# */

CXX = gcc

# -c x # force C

CXXFLAGS =   -Wall -Werror -Wextra -pedantic -std=c17 -g -fsanitize=address -DSTRICT_WARNINGS

LDFLAGS = -fsanitize=address -DSTRICT_WARNINGS


SRC = btest.c tipo.c  bnode.c  btree.c deque.c

OBJ = $(SRC:.c=.o)

EXEC = btest.out

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) $(CXXFLAGS)

clean:
	rm -rf $(OBJ) $(EXEC)

