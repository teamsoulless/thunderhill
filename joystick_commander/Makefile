##########################################################
# Makefile for Joystick Commander
##########################################################


# source PS environment if not already done
PSYNC_HOME ?= /usr/local/polysync

# target
TARGET  := bin/polysync-joystick-commander

# source files
SRCS    :=  src/joystick.c \
	    src/messages.c \
	    src/commander.c \
	    src/node.c

# dep files
OBJS    := $(SRCS:.c=.o)
DEPS    := $(SRCS:.c=.dep)
XDEPS   := $(wildcard $(DEPS))

# get standard PolySync build resources
include $(PSYNC_HOME)/build_res.mk

# compiler
CC = gcc

# cflags
CCFLAGS += `pkg-config --cflags sdl2`

# include directories
INCLUDE += -Iinclude -I$(PSYNC_HOME)/dynamic_driver/include

# add node template library, must be first
# add util library last, let path be found
LIBS := -L$(PSYNC_HOME)/lib -lpolysync_node $(LIBS) -lpolysync_data_model `pkg-config --libs sdl2`

# all builds directories and target
all: dirs $(TARGET)

# check deps
ifneq ($(XDEPS),)
include $(XDEPS)
endif

# target
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

# object files
$(OBJS): %.o: %.c %.dep
	$(CC) $(CCFLAGS) $(INCLUDE) -o $@ -c $<

# dep files
$(DEPS): %.dep: %.c Makefile
	$(CC) $(CCFLAGS) $(INCLUDE) -MM $< > $@

# directories
dirs::
	mkdir -p bin
	mkdir -p doc/html

# clean
clean::
	rm -f src/*.o
	rm -f src/*.dep
	rm -f $(TARGET)
	rm -rf bin
	rm -rf doc/html
	rm -rf doc/latex
	rm -rf doc/pdf
	rm -f ospl-*.log
