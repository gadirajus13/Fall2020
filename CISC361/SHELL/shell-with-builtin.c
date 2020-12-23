//Addison Kuykendall and Sohan Gadiraju
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include "sh.h"
#include <limits.h>
#include <unistd.h>
#include <pwd.h>
#include <glob.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include <utmpx.h>
#include <pthread.h>
extern char **environ;


#define BUFFERSIZE 2048
Users_t *userHead;
pthread_mutex_t userLock;

int main(int argc, char **argv, char **envp)
{
	char	buf[MAXLINE];
	char    *arg[MAXARGS];  // an array of tokens
	char 	**args;
	char    *ptr;
	char    *pch;
	char    *cwd = malloc(sizeof(char)*1024);
	cwd = getcwd(NULL, 0);
	char    *pwd = malloc(sizeof(char)*1024);
	pwd = cwd;
	pid_t	pid;
	int	status, i, arg_no, noclobber, userThread;
	char *cmdline = calloc(MAX_CANON, sizeof(char));
	char   *prompt = malloc(sizeof(char)*1024);
	prompt = "";
	glob_t globbuf;
	int s = 1;
	
	int nmbrArgs = 0;
	struct utmpx *ut;
	struct utmpx *up;
	

	printf("%s [%s]>> ", prompt,  getcwd(NULL,0));	/* print prompt (printf requires %% to print %) */
	signal(SIGINT, sig_intHandler);  // Check and ignore Ctrl C
	signal(SIGTSTP, sig_stpHandler); //Check and ignore Ctrl Z
	
	if(isPipe(&arg[0], nmbrArgs)){ //Check if pipe is found on command line
			printf("Pipe detected\n");
			pipeFunc(&args[0], nmbrArgs);
    	}

	while (1) {
	        nmbrArgs = 0;
		s = fgets(buf, MAXLINE, stdin);
		
		if (s == 0){ // Checking for Ctrl-D
			printf("Use exit to leave this shell");
			break;
		}
	
		if (strlen(buf) == 1 && buf[strlen(buf) - 1] == '\n')
		  goto nextprompt;  // "empty" command line
		
		if (buf[strlen(buf) - 1] == '\n')
			buf[strlen(buf) - 1] = 0; /* replace newline with null */
		// parse command line into tokens (stored in buf)
		arg_no = 0;
                pch = strtok(buf, " ");
                while (pch != NULL && arg_no < MAXARGS)
                {
		  arg[arg_no] = pch;
		  arg_no++;
		  nmbrArgs++;
                  pch = strtok (NULL, " ");
                }
		arg[arg_no] = (char *) NULL;

		if (arg[0] == NULL)  // "blank" command line
		  goto nextprompt;
		
		args = calloc(MAXARGS, sizeof(char*));

		struct sigaction sa;
		sa.sa_handler = &sig_chldHandler;
		sigemptyset(&sa.sa_mask);
		sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
		if (sigaction(SIGCHLD, &sa, 0) == -1) {
			perror(0);
			exit(1);
  		}

		signal(SIGTSTP, SIG_IGN); //handles ctrl+z signal
    	signal(SIGINT, SIG_IGN); //handles ctrl+c signal;

		free(args);
		/* print tokens
		for (i = 0; i < arg_no; i++)
		  printf("arg[%d] = %s\n", i, arg[i]);
                */
		signal(SIGINT, sig_intHandler);  // Check and ignore Ctrl C
		signal(SIGTSTP, sig_stpHandler); //Check and ignore Ctrl Z
		
        if (strcmp(arg[0], "pwd") == 0) { // built-in command pwd 
		  printf("Executing built-in [pwd]\n");
	          ptr = getcwd(NULL, 0);
                  printf("%s\n", ptr);
                  free(ptr);
	        }
	
		else if(strcmp(arg[0], "exit")==0){ //built-in commad exit
		    exit(0);
		}

		else if(strcmp(arg[0], "prompt")==0){ //prompt command line
		  //If there are no arguments, ask what the promtp should be
		  if(arg[1]==NULL){
		    printf("input prompt prefix: \n");
		    char buf2[MAXLINE];
		    fgets(buf2, MAXLINE, stdin);
		    prompt = buf2;
		    prompt[strlen(prompt)-1] = '\0';
		  }
		  //Otherwise make the prompt equal the argument
		  else{
		    prompt = arg[1];
		  }
		}

		else if (strcmp(arg[0], "kill") == 0) // kill command line
        {    	           
			printf("Executing built-in [kill]\n");
			if (arg[1] == NULL) 
				printf("kill: usage: kill [-signum] pid\n");	// no args, error
			else if (arg[1] != NULL && arg[2]==NULL) 
			{									// only one argument, use SIGTERM
				const int pid = atoi(arg[0]);
				if (kill(pid, SIGTERM) == -1)
					printf("kill: (%i) - No such process\n", pid);	// failure
			} 
			else {
					int signum = atoi(arg[0]+1); // ignore - infront of number
					if (signum > 31) signum = 0;
					char **args_p = arg+1;
					for (i = 1; *args_p; i++) {			// loop through each pid
						const int pid = atoi(arg[i]);
						if (kill(pid, signum) == -1)
							printf("kill: (%i) - No such process\n", pid);	// failure
						args_p++;
				}
			}
		}

		else if (strcmp(arg[0], "list") == 0) // list command line
		{                  
		    printf("Executing built-in list:\n");
			if(arg[1]==NULL){
				list(cwd);
				
			}
			else{
				int temp = arg_no;
				while(temp >1){
				list(arg[temp-1]);
				temp--;
				}
				
			}
		}

		else if (strcmp(arg[0], "pid") == 0) // pid command line
        		{                   
        		    printf("Executing built-in %s\n", arg[0]);
        		    printf("%i\n", getpid());   
       		} 
		
		else if(strcmp(arg[0], "setenv")==0){ //setenc command line
		  //If there are no arguments, print the enviroments
		  if(arg[1] == NULL){
		    char **tmp;
		    for(tmp = environ; *tmp != 0; *tmp++){
		      printf("%s\n", *tmp);
		    }
		    //free(tmp);
		    }
		  //If there is one argument, set it as an empty enviroment variable
		    else if(arg[2]==NULL){
		      
		      //Special case for PATH
		      if(strcmp(arg[1], "PATH")==0){
					continue;
		       }
		       else{
			setenv(arg[1], "", 1);
	       	      }
		    }
		    //If there are 2 arguments, make the second one the value of the first
		    else{
				setenv(arg[1], arg[2], 1);
		    }
		}
		
		else if(strcmp(arg[0], "printenv")==0){ //Built-in Printenv vommand
		      //Print the whole enviroments if there are no arguments
		      if(arg[1]==NULL){
		        char **tmp;
		        for(tmp=environ; *tmp!=0; *tmp++){
		  	  printf("%s\n", *tmp);
		        }
		      }
		      //call getenv on the argument if there is 1
		      else if(arg[2]==NULL){
				printf("%s\n", getenv(arg[1]));
		      }
		      //If there are more arguments, print error
		      else{
				printf("Too many arguments\n");
		      }
		}
		
		else if(strcmp(arg[0], "cd")==0){ //built-in command cd
			cwd = getcwd(NULL, 0);
			char *tmp = calloc(strlen(cwd)+1, sizeof(char));
			strcpy(tmp, cwd);
			printf("Executing built-in cd\n");
			
			//If they only type "cd", go home
			if(arg[1] == NULL){
				printf("%s\n", getenv("HOME"));
				if(chdir(getenv("HOME"))!=-1){
					strcpy(cwd, getenv("HOME"));
					strcpy(pwd, tmp);
				}
				else{
				printf("That directory does not exist\n");
				}
			}
			//Go back 1 directory
				else if(strcmp(arg[1], "-") == 0){
				if(chdir((pwd))!=-1){
					printf("%s\n", cwd);
					printf("%s\n", pwd);
					strcpy(cwd, pwd);
					strcpy(pwd, tmp);
				}
				else{
				  printf("That directory does not exist\n");
				  }
				}
			//Go to named directory
				else{
				  int i, j;
				  int wildcardFlag = 0;
				  if(chdir(arg[1])==0){
				    strcpy(cwd, getcwd(NULL, 0));
				    strcpy(pwd, tmp);
				  }
				  else{
				    printf("That directory does not exist\n");
				  }
				}
			free(tmp);
		}

		else if(strcmp(arg[0], "where") == 0){ // built-in command where
			printf("Executing built-in [where]\n");
			if(arg[1] == NULL){ //"empty" where
				printf("where: Too few arguments.\n");
				goto nextprompt;
			}
			struct pathelement *p, *tmp;
			char *cmd;

			p = get_path();
			tmp = p;
			while(tmp){
				where(arg[1], p);
				tmp = tmp->next;
			} 
		}

		else if (strcmp(arg[0], "which") == 0) { // built-in command which
		  	struct pathelement *p, *tmp;
        	char *cmd;
                    
		  	printf("Executing built-in [which]\n");

			if (arg[1] == NULL) {  // "empty" which
				printf("which: Too few arguments.\n");
				goto nextprompt;
					}

			p = get_path();
			
			tmp = p;
		  	while (tmp) {      // print list of paths
		    printf("path [%s]\n", tmp->element);
		    tmp = tmp->next;
			}
          
		      	cmd = which(arg[1], p);
                      if (cmd) {
		    	printf("%s\n", cmd);
                        free(cmd);
		      }
		      else               // argument not found
			printf("%s: Command not found\n", arg[1]);

		    while (p) {   // free list of path values
				tmp = p;
				p = p->next;
				free(tmp->element);
				free(tmp);
				}
	        }
	
		else if(strcmp(arg[0], "noclobber") == 0){ // Built-in Noclobber command
			if(arg[1] == NULL){
				if(noclobber == 0){ //turns noclobber on or off
					noclobber = 1;
					printf("Executing built-in [%s]\n%d\n", arg[0], noclobber);
				}
				else{
					noclobber = 0;
					printf("Executing built-in [%s]\n%d\n", arg[0], noclobber);
				}
			}
			else{
				fprintf(stderr, "Too many arguments\n");
			}
		}

		else if(strcmp(arg[0], "watchuser") == 0){ // Built-in watchuser command
			if(arg[1]== NULL){
				Users_t *temp = userHead;
				while(temp != NULL){ 
					printf("%s\n", temp->name);
					temp = temp->next;
				}
			}
			else if((arg[1] != NULL) && (arg[2]==NULL)){
				printf("Executing built-in [%s]\n", arg[0]);
				if(userThread == 0){ //creates a thread of users
				  //	pthread_create(&ut, NULL, watchUser, "List Of Users");
					userThread = 1; //only one user thread should be created
				}
				pthread_mutex_lock(&userLock); //locks thread
				addUser(arg[1]); //adds user to the watch users list
				pthread_mutex_unlock(&userLock); //unlocks thread
			}
			else if(arg[2] != NULL && strcmp(arg[2], "off") == 0){ // Turn off watch user
				printf("Turning off [watchuser]\n");
				pthread_mutex_lock(&userLock); //locks thread
				removeUser(arg[1]);
				pthread_mutex_unlock(&userLock); //unlocks thread
			}
		}
		else {
		  char *path = (char*) malloc(strlen(arg[0]));
		  struct pathelement *p = get_path();
		  //get the command
		  path = which(arg[0], p);
		  struct stat pstat;
		  stat(path, &pstat);
		  //Check that the command exists
		  int check = access(path, F_OK | X_OK);
		  if(check == 0 && S_ISREG(pstat.st_mode)){
		    if(path!=NULL){
		      //Create the child process
		      pid_t child_pid = fork();
		      if(child_pid == 0){
			int redirectLocation = -1;
			//find the location of the redirect if one exists
			for(int i = 0; i<nmbrArgs; i++){
			  if(strcmp(arg[i], ">") ==0 ||strcmp(arg[i], ">>")==0||strcmp(arg[i], ">&")==0||strcmp(arg[i], ">>&")==0||strcmp(arg[i], "<")==0){
			    redirectLocation = i;
			  }
			}
			//Execute the redirect if there is one
	         	if(redirectLocation!=-1){
			  char* redirect = malloc(sizeof(char)*1024);
			  redirect = arg[redirectLocation];
			  char* loc = malloc(sizeof(char)*1024);
			  loc = arg[redirectLocation+1];
			  int exists = 0;
			  struct stat s;
			  if(stat(loc, &s)==0){
			    exists = 1;
			  }
			  //Check noclobber
			  if(noclobber==0){
			    if(exists==1){
			      if(strcmp(redirect, ">")==0){
				int f = open(loc, O_WRONLY|O_CREAT|O_TRUNC, 0666);
				close(STDOUT_FILENO);
				dup(f);
				close(f);
			      }
			      if(strcmp(redirect, ">&")==0){
				int f = open(loc, O_WRONLY|O_CREAT|O_TRUNC, 0666);
				close(STDOUT_FILENO);
				dup(f);
				close(STDERR_FILENO);
				dup(f);
				close(f);
			      }
			      if(strcmp(redirect, ">>")==0){
				int f = open(loc, O_WRONLY|O_CREAT|O_APPEND,0666);
			          close(STDOUT_FILENO);
				  dup(f);
				  close(f);
			      }
			      if(strcmp(redirect, ">>&")==0){
				int f = open(loc, O_WRONLY|O_CREAT|O_APPEND,0666);
				close(STDOUT_FILENO);
				dup(f);
				close(STDERR_FILENO);
				dup(f);
				close(f);
			      }
			      if(strcmp(redirect, "<")==0){
				int f = open(loc, O_RDONLY);
				close(STDIN_FILENO);
				dup(f);
				close(f);
			      }
			    }
			    else{
			      printf("File does not exist\n");
			    }   
			  }
			  else{
			    printf("Noclobber is on\n");
			  }
			  //get rid of the file name and the redirect so it does not write it
			  for(int j = redirectLocation; j<nmbrArgs; j++){
			    arg[j] = NULL;
			  }
			  
			    printf("command %s", path);
			    execve(path, arg, (char *) 0);
			  int childstat;
			}	  	
		       else{
			 execlp(path, arg, (char *)0);
			}
		      free(path);
			}
		      }
		  }
		}
        nextprompt:
		printf("%s %s>> ", prompt, getcwd(NULL, 0));
	}
	exit(0);
}


void list (char *dir)
{
    DIR *adir = opendir(dir);
  	struct dirent *file;

  	if(adir != NULL){
		file=readdir(adir);
		printf("%s:\n", dir);
		while(file){
			printf("\t%s\n",file->d_name);
			file=readdir(adir);
    	}
  	}
  else{
    printf("list: cannot access %s: no such file or directory\n", dir);
  }
  free(adir);
}

void where(char *command, struct pathelement *p){
	if(p == NULL){
		return;
	}
	else{
		struct pathelement *tmp = p;
		char *ch;
		while(tmp!=NULL){
		ch = calloc(strlen(command)+strlen(tmp->element)+2, sizeof(char));
		strcpy(ch, tmp->element);
		strcat(ch, "/");
		strcat(ch, command);
		if(access(ch, F_OK)==0){
			printf("%s\n", ch);
		}
		tmp = tmp->next;
		free(ch);
		}
	}
}

//Ctrl C handler to ignore
void sig_intHandler(int button) {
	signal(SIGINT, sig_intHandler);
	fflush(stdout);
}

// Ctrl Z handler
void sig_stpHandler(int sig) {
  signal(SIGTSTP, sig_stpHandler);
  fflush(stdout);
}

//Kill child process when Ctrl C is pressed
void sig_chldHandler(int sig) {
  int saved_errno = errno;
  while (waitpid((pid_t)(-1), 0, WNOHANG) > 0) {}
  errno = saved_errno;
}

// Adding user in order to make Watch User
void addUser(char *user){
  Users_t *newUser = malloc(sizeof(struct Users)); //malloc space for the new user
  Users_t *temp;
  newUser->name = malloc(strlen(user) + 1); //malloc space for user's name
  strcpy(newUser->name, user); //copies user into the new user's name field
  newUser->next = NULL;
  newUser->prev = NULL;
  if(userHead == NULL){ //first user in the list
    userHead = newUser;
    userHead->prev = NULL;
    userHead->next = NULL;
  }
  else{
    temp = userHead;
    while(temp->next != NULL){ //loops through list to add new users at the end
      temp = temp->next;
    }
    temp->next = newUser;
    newUser->prev = temp;
  }
}

// Removing a user from logins
void removeUser(char *user){
	Users_t *temp;
	if(userHead == NULL){
		printf("There's no user to remove\n");
	}
	else{
		temp = userHead; //starts at head
		while(temp != NULL){ //loops through list
			if(strcmp(temp->name, user) == 0){ //Name is on list
				if(temp == userHead){ 
					userHead = userHead->next;
					free(temp->name);
					free(temp);
				}
				else if(temp->next != NULL && temp->prev != NULL){ //Name is in middle of list
					temp->next->prev = temp->prev;
					temp->prev->next = temp->next;
					free(temp->name);
					free(temp);
				}
				else if(temp->prev != NULL && temp->next == NULL){ //Name is at end of list
					temp->prev->next = NULL;
					free(temp->name);
					free(temp);
				}
			}
			else{ //iterate through list
				temp = temp->next;
			}
		}
	}
}

// Watch User function to record which user is online and who has logged on
static void *watchUser(void *arbitrary){
	struct utmpx *up;
	Users_t *temp = userHead;
	while(1){ //while running
    setutxent();
    while(up == getutxent()){
      if(up->ut_type == USER_PROCESS){
        pthread_mutex_lock(&userLock); //lock thread
        while(temp != NULL){
          if(strcmp(temp->name, up->ut_user) == 0){
            printf("%s had logged on %s from %s\n", up->ut_user, up->ut_line, up->ut_host);
          }
          temp = temp->next;
        }
        pthread_mutex_unlock(&userLock); //unlock thread
      }
      temp = userHead;
    }
    sleep(20); //sleep for 20
  }
}

// Pipes output of first function into input of the second
void pipeFunc(char **args, int arg){
	char **src = calloc(MAXARGS, sizeof(char *));
	char **dst = calloc(MAXARGS, sizeof(char *));
	char *isPipe = calloc(MAX_CANON, sizeof(char *));
	int i, pid, status, fid[2];
	int pipeCheck = 0;
	int j = 0;
	for(i = 0; i < arg; i++){  //checks all arguments
		if ((strcmp(args[i], "|") == 0) || (strcmp(args[i], "|&") == 0)){
			isPipe = args[i]; //isPipe becomes the pipe character
			break;
		}
		src[i] = args[i];
	}
	while(args[i] != NULL){ //iterates to the end of the list sarting at i, need j to start at 0
		dst[j] = args[i];
		i++;
		j++;
	}
	pipe(fid);
	source(fid, src, isPipe);
	destination(fid, dst, isPipe); 
	close(fid[0]);
	close(fid[1]);
}

//  Creates child process for pipe function to store output 
void source(int pfd[], char **cmdline, char *symbol){
	int pid = fork();
	if(pid == 0){ //child
		if(strcmp(symbol, "|&") == 0){
			close(2);
		}
		close(1);
		dup(pfd[1]);
		close(pfd[0]);
		execvp(cmdline[0], cmdline);
		perror(cmdline[0]);
		kill(pid, SIGTERM);
	}
	else if(pid == -1){
	perror("Failed to create child process\n");
	exit(EXIT_FAILURE);
  }
}

// Sends output of first command to second
void destination(int pfd[], char **cmdline, char *symbol){
	int pid = fork();
	if(pid == 0){  // Checking to see if process is a child
		if(strcmp(symbol, "|&") == 0){
			close(2);
		}
		close(0);
		dup(pfd[0]);
		close(pfd[1]);
		execvp(cmdline[0], cmdline);
		perror(cmdline[0]);
		kill(pid, SIGTERM); //Kill child process to avoid zombie
	}
	else if(pid == -1){
		perror("Failed to create child process");
		exit(EXIT_FAILURE);
	}
}

char *isPipe(char **args, int arg){
	char *currArg = args[0];
	char *finalArg = args[0];
	for(int i = 0; i < arg; i++){ 
		if ((strcmp(currArg, "|") == 0) || (strcmp(currArg, "|&") == 0)){
			finalArg = args[i - 1];
			return finalArg; 
		}
		currArg = args[i];
	}
	return NULL; 
}


