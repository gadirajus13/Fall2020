CC=gcc
# CC=gcc -Wall

all: order fork-join rendezvous

order: order.c
	$(CC) order.c -pthread -o order

fork-join: fork-join.c
	$(CC) fork-join.c -pthread -o fork-join

rendezvous: rendezvous.c
	$(CC) rendezvous.c -pthread -o rendezvous

clean:
	rm -rf order fork-join rendezvous