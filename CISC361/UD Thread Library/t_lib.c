#include <signal.h>
#include "t_lib.h"

tcb *running;
tcb *ready;

// Inter-Thread Communications (Phase 4)

/* Create a new mailbox with a new mailbox pointed to by mb */
int mbox_create(mbox **mb) 
{
  *mb = malloc(sizeof(mbox));
  (*mb)->msg = NULL;
  sem_init(&((*mb)->mbox_sem), 0);
  return 0;
}

/* Destroy all states of mailbox pointed to by mb */
void mbox_destroy(mbox **mb) {
  free((*mb)->msg);
  free((*mb)->mbox_sem);
  free(mb);
}

/* Add message to end of mailbox*/
void mbox_deposit(mbox *mb, char *msg, int len){
  mb->msg->message = malloc(sizeof(msg));
  mb->msg->len = len;

  struct messageNode *addMsg = malloc(sizeof(struct messageNode));
  addMsg->message = malloc(sizeof(msg));
  addMsg->message = msg;
  addMsg->next = NULL;
  addMsg->len = len;

  sem_wait(mb->mbox_sem);
  struct messageNode *tmp = mb->msg;
  if (tmp == NULL) {
    mb->msg = addMsg;
  } 
  else {
    while (tmp->next != NULL) {
      tmp = tmp->next;
    }

    tmp->next = addMsg;  
  }

  sem_signal(mb->mbox_sem);
}

void mbox_withdraw(mbox *mb, char *msg, int *len){
  messageNode *start = mb->msg;
  if(start == NULL){
    *msg = "";
    *len = 0;
  }
  else{
    strncpy(msg, start->message);
    *len = start->len;

    mb->msg = start->next;
    free(start);
  }
}

void send(int tid, char *msg, int len){
  if(running->thread_id == tid){
    mbox_deposit(running->mailbox, msg, len);
  }
  else{
    tcb *curr;
    if(ready!=NULL){
      curr = ready;
      while(curr->next!=NULL){
	if(curr->thread_id==tid){
	  mbox_deposit(curr->mailbox, msg, len);
	}
	
        curr = curr->next;
      }
    }
  }
}

void recieve(int *tid, char *msg, int *len){
  mbox *r = running->mailbox;
  if(tid==0){
    mbox_withdraw(r, msg, len);
  }
  else{
    messageNode *curr = mb->msg;
    if(curr!=NULL){
      while(curr->next!=NULL){
	if(curr->sender == *tid){
	  msg = curr->message;
	  len = curr->len;
	}
	curr = curr->next;
      }
    }
  }
}


// Semaphore Functions (Phase 3)

/* Create a new semaphore pointed to by sp with a count value of sem_count. */
int sem_init(sem_t **sp, int sem_count)
{
  sighold(SIGALRM);
  *sp = (sem_t*) malloc(sizeof(sem_t));
  (*sp)->count = sem_count;
  (*sp)->q = NULL;
  sigrelse(SIGALRM);
} 

/* Current thread does a wait (P) on the specified semaphore. */
void sem_wait(sem_t *sp)
{
  sighold(SIGALRM);
  sp->count--;
  if (sp->count < 0) { // to wait on semaphore
    tcb* tmp;
    tmp = running;
    tmp->next = NULL;
    running = ready; // move the 1st TCB in ready to running 
	
    if (ready != NULL)
      ready = ready->next;

    if (sp->q == NULL) 
      sp->q = tmp; // move running to sem's q
    else { 
      tcb* iter;
      iter = sp->q;
      while (iter->next != NULL) // loop to end of sem's q
        iter = iter->next; 
      iter->next = tmp; // add old running to end of sem's q
    }
    sigrelse(SIGALRM);

    swapcontext(tmp->thread_context, running->thread_context); 
  }
}

/* Current thread does a signal (V) on the specified semaphore. Follow the Mesa
 semantics where the thread that signals continues, and the first waiting (blocked)
 thread (if there is any) becomes ready. */

void sem_signal(sem_t *sp)
{
  sighold(SIGALRM);	
  sp->count++; //increase count

  if ((sp->q != NULL)) { // if q isn't empty, wake the 1st TCB and move it to the end of ready
    tcb* tmp;
    tmp = sp->q;
    sp->q = sp->q->next; // update head of sem's q
    if (ready == NULL)
      ready = tmp;
    else {
      tcb* iter;
      iter = ready;
      while (iter->next != NULL) // loop through ready queue
        iter = iter->next; // add awoken TCB to end of ready
      iter->next = tmp;
    }
    tmp->next = NULL; 	
  }
  sigrelse(SIGALRM);
}

/* Destroy (free) any state related to specified semaphore. */
void sem_destroy(sem_t **sp)
{
  sighold(SIGALRM);
  if ((*sp)->q != NULL) { // if sem's q isn't empty
    if (ready == NULL)
      ready = (*sp)->q; // move to ready 
    else {
      tcb* iter;
      iter = ready;
      while (iter->next!= NULL) // loop to end of ready q
	    iter = iter-> next;
      iter->next = (*sp)->q; // add to end of ready q
    }
    (*sp)->q = NULL;
  }

  free(*sp); // free the semaphore
  sigrelse(SIGALRM);
}


// Phase 1

void t_yield()
{
  sighold(SIGALRM);
  tcb* tmp;
  tmp =  running; //store the currently running htread in tmp
  tmp->next = NULL;
  
  if (ready != NULL) { //only yield if there is something in ready queue
	running = ready; //update running to first thread in ready queue
   	ready = ready->next; //update ready to next thread
  	running->next = NULL;
  	tcb* iter;
	iter = ready;
	if (iter == NULL) //if there is nothing else in ready queue
		ready = tmp;
	else { 
  		while (iter->next != NULL) //loop till end of queue
			iter = iter->next;
  		iter->next = tmp; //add tmp to end of queue
	}
	sigrelse(SIGALRM);
	swapcontext(tmp->thread_context, running->thread_context);
  }

}
 /* Initialize the thread library by setting up the "running" 
and the "ready" queues, creating TCB of the "main" thread, and inserting it into the running queue. */
void t_init()
{
	tcb *tmp = (tcb*) malloc(sizeof(tcb));
	tmp->thread_context = (ucontext_t *) malloc(sizeof(ucontext_t));
	getcontext(tmp->thread_context);
	tmp->next = NULL;
	tmp->thread_id = 0;
	running = tmp;
	ready = NULL; 
}
/* Create a thread with priority pri, start function func with argument thr_id 
as the thread id. Function func should be of type void func(int). TCB of the newly
 created thread is added to the end of the "ready" queue; the parent thread calling
t_create() continues its execution upon returning from t_create(). */

int t_create(void (*fct)(int), int id, int pri)
{ 
  sighold(SIGALRM);
  size_t sz = 0x10000;
  tcb* tmp = (tcb*) malloc(sizeof(tcb));
  tmp->thread_context = (ucontext_t *) malloc(sizeof(ucontext_t));

  getcontext(tmp->thread_context);
/***
  uc->uc_stack.ss_sp = mmap(0, sz,
       PROT_READ | PROT_WRITE | PROT_EXEC,
       MAP_PRIVATE | MAP_ANON, -1, 0);
***/
  tmp->thread_context->uc_stack.ss_sp = malloc(sz);  /* new statement */
  tmp->thread_context->uc_stack.ss_size = sz;
  tmp->thread_context->uc_stack.ss_flags = 0;
  // tmp->thread_context->uc_link = running->thread_context; 
  makecontext(tmp->thread_context, (void (*)(void)) fct, 1, id);
  tmp->thread_id = id;
  tmp->thread_priority = pri;
  tmp->next = NULL;

  if (ready == NULL)
	ready = tmp;
  else {
	tcb* t = ready;
	while(t->next!=NULL) {
		t = t->next;
	}
	t->next = tmp;
  }
  sigrelse(SIGALRM);
}

/* Terminate the calling thread by removing (and freeing) its TCB from the
"running" queue, and resuming execution of the thread in the head of the 
"ready" queue via setcontext(). */
void t_terminate()
{
	sighold(SIGALRM);
	tcb* tmp;
	tmp = running;
	running = ready;
	if (ready!=NULL)
		ready = ready->next;
	free(tmp->thread_context->uc_stack.ss_sp);
	free(tmp->thread_context);
	free(tmp);
	sigrelse(SIGALRM);
	setcontext(running->thread_context);
}

void t_shutdown()
{
	sighold(SIGALRM);
	if (ready!=NULL) {
		tcb* tmp;
		tmp = ready;
		while(tmp != NULL) {
			ready = ready->next;
			free(tmp->thread_context->uc_stack.ss_sp);
			free(tmp->thread_context);
			free(tmp);
			tmp = ready;
		}
	}
	free(running->thread_context);
	free(running);
	sigrelse(SIGALRM);	
}
