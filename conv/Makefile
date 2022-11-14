# Compiler/linker setup ------------------------------------------------------

# Mac OS X-specific flags.  Comment these out if using Linux.
#PLATFORM = osx
#CC       = gcc
#CFLAGS   = -fast -Wall
#OSLIBS   = -Wl,-framework -Wl,IOKit
#LDFLAGS  =

# Linux-specific flags.  Comment these out if using Mac OS X.
PLATFORM = linux
CC       = gcc
CFLAGS   = -O3 -Wall
OSLIBS   =
LDFLAGS  = -lm

#-std=c++11: special modifier to allow for lambda expressions in TBB

# make all: all OBJS are compiled and executable available
# make <individual>
# make clean
# '-ltbb' has to be included at the end of the command (can't be in the CFLAGS)
# do not use spaces (tabs are allowed)
# Example programs -----------------------------------------------------------

OBJS = conv
all: $(OBJS)

conv: conv.c
	$(CC) $(CFLAGS) conv $(LDFLAGS) -o conv


# Maintenance and stuff ------------------------------------------------------
clean:
	rm -f $(OBJS) *.o core
