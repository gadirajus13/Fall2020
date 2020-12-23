#include "get_path.h"

int pid;
char *which(char *command, struct pathelement *pathlist);
void where(char *command, struct pathelement *p);
int parseLine(char *commandline, char** args);
void sig_intHandler(int sig);
void sig_stpHandler(int sig);
void sig_chldHandler(int sig);
void list(char *dir);
void printenv(char **envp);
void addUser(char *user);
void removeUser(char *user);
static void *watchUser(void *arbitrary);
void pipeFunc(char **args, int arguments);
void source(int pfd[], char **cmd, char *symbol);
void destination(int pfd[], char **cmd, char *symbol);
char *isPipe(char **args, int arguments);

#define PROMPTMAX 64
#define MAXARGS   16
#define MAXLINE   128

typedef struct Users{
  char *name;
  struct Users *next;
  struct Users *prev;
} Users_t;
