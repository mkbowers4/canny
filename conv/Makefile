PLATFORM = linux
CC       = gcc
CFLAGS   = -Wall
OSLIBS   =
LDFLAGS  = -lm


OBJS = canny
all: $(OBJS)

canny: canny.c
	$(CC) canny.c $(CFLAGS) $(LDFLAGS) -o canny


# Maintenance and stuff ------------------------------------------------------
clean:
	rm -f $(OBJS) *.o core
