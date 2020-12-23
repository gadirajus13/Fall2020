# Makefile for UD CISC user-level thread library

CC = gcc
CFLAGS = -g

LIBOBJS = t_lib.o 

TSTOBJS = test00.o philosophers.o test03.o rendezvous.o test10.o test03.o 3test.o partialorder.o

# specify the executable 

EXECS = test00 philosophers test03 rendezvous test10 test03 3test partialorder

# specify the source files

LIBSRCS = t_lib.c 

TSTSRCS = test00.c philosophers.c test03.c rendezvous.c test10.c test03.c 3test.c partialorder.c

# ar creates the static thread library

t_lib.a: ${LIBOBJS} Makefile
	ar rcs t_lib.a ${LIBOBJS}

# here, we specify how each file should be compiled, what
# files they depend on, etc.

t_lib.o: t_lib.c t_lib.h Makefile
	${CC} ${CFLAGS} -c t_lib.c

test00.o: test00.c ud_thread.h Makefile
	${CC} ${CFLAGS} -c test00.c

test00: test00.o t_lib.a Makefile
	${CC} ${CFLAGS} test00.o t_lib.a -o test00

test03.o: test03.c ud_thread.h Makefile
	${CC} ${CFLAGS} -c test03.c

test03: test03.o t_lib.a Makefile
	${CC} ${CFLAGS} test03.o t_lib.a -o test03

philosophers.o: philosophers.c ud_thread.h Makefile
	${CC} ${CFLAGS} -c philosophers.c

philosophers: philosophers.o t_lib.a Makefile
	${CC} ${CFLAGS} philosophers.o t_lib.a -o philosophers

rendezvous.o: rendezvous.c ud_thread.h Makefile
	${CC} ${CFLAGS} -c rendezvous.c

rendezvous: rendezvous.o t_lib.a Makefile
	${CC} ${CFLAGS} rendezvous.o t_lib.a -o rendezvous

partialorder.o: partialorder.c ud_thread.h Makefile
	${CC} ${CFLAGS} -c partialorder.c

partialorder: partialorder.o t_lib.a Makefile
	${CC} ${CFLAGS} partialorder.o t_lib.a -o partialorder

3test.o: 3test.c ud_thread.h Makefile
	${CC} ${CFLAGS} -c 3test.c

3test: 3test.o t_lib.a Makefile
	${CC} ${CFLAGS} 3test.o t_lib.a -o 3test

test10.o: test10.c ud_thread.h Makefile
	${CC} ${CFLAGS} -c test10.c

test10: test10.o t_lib.a Makefile
	${CC} ${CFLAGS} test10.o t_lib.a -o test10

clean:
	rm -f t_lib.a ${EXECS} ${LIBOBJS} ${TSTOBJS} 
