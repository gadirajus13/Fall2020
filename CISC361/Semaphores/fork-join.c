#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>

sem_t s;

void *child(void *arg) {
    printf("child\n");
    sem_wait(&s);
    sleep(1);
    return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t p;
    sem_init(&s, 1, 1);
    printf("parent: begin\n");
    pthread_create(&p, NULL, child, NULL);
    sem_wait(&s);
    printf("parent: end\n");
    return 0;
}

