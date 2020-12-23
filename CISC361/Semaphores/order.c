#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t complete1, complete2, complete4;

void* thread(void *a)
{
    /*thread function*/
    int w = (int)a;
    if(w == 2){
        printf("I am worker %d\n", w);
        sem_post(&complete2);
    }
    else if(w != 3){
        sem_wait(&complete2);
        printf("I am worker %d\n", w);
        sem_post(&complete2);
        if(w==1)
            sem_post(&complete1);
        else
            sem_post(&complete4);
    }
    else{
        sem_wait(&complete1);
        sem_wait(&complete4);
        printf("I am worker %d\n", w);
    }
}

int main()
{
    sem_init(&complete1, 0, 0);
    sem_init(&complete2, 0, 0);
    sem_init(&complete4, 0, 0);

    pthread_t p1, p2, p3, p4;
    pthread_create(&p1, NULL, thread, 1);
    pthread_create(&p2, NULL, thread, 2);
    pthread_create(&p3, NULL, thread, 3);
    pthread_create(&p4, NULL, thread, 4);
    
    pthread_join(p1, NULL);
    pthread_join(p2, NULL);
    pthread_join(p3, NULL);
    pthread_join(p4, NULL);
  
    return 0;
}