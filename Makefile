PLATFORM = linux
CC       = g++
CFLAGS   = -Wall -std=c++11
OSLIBS   =
LDFLAGS  = -ltbb -lm


OBJS = canny
all: $(OBJS)

canny: canny.cpp
	$(CC) canny.cpp $(CFLAGS) $(LDFLAGS) -o canny

# Maintenance and stuff ------------------------------------------------------
clean:
	rm -f $(OBJS) *.o core



# PLATFORM = linux
# CC       = gcc
# CFLAGS   = -Wall
# OSLIBS   =
# LDFLAGS  = -lm
#
#
# OBJS = canny
# all: $(OBJS)
#
# canny: canny.c
# 	$(CC) canny.c $(CFLAGS) $(LDFLAGS) -o canny
#
# # Maintenance and stuff ------------------------------------------------------
# clean:
# 	rm -f $(OBJS) *.o core
