// Programming Assignment 1
// Sohan Gadiraju and Addison Kuykendall
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mp3.h"
#define  BUFFERSIZE 128

mp3_t *head;

void insert(char *name,char *title, int sec);
void delete(mp3_t** head_ref, mp3_t* mp3);
void print();
void freeList();

int main()
{
  int i, sec, len;
  struct mp3 *m;
  char buffer[BUFFERSIZE], c1;
  char title[BUFFERSIZE];
  char del[BUFFERSIZE];
  head = NULL;

  while (1) {
    printf("\nList Operations\n");
    printf("===============\n");
    printf("(1) Add MP3\n");
    printf("(2) Delete MP3\n");
    printf("(3) Display MP3\n");
    printf("(4) Display MP3 (Reversed)\n");
    printf("(5) Exit\n");
    printf("Enter your choice : ");
    if (scanf("%d%c", &i, &c1) <= 0) {          // use c to capture \n
        printf("Enter only an integer...\n");
        exit(0);
    } else {
        switch(i)
        {
        case 1: printf("Enter the name of the artist: ");
                if (fgets(buffer, BUFFERSIZE , stdin) != NULL) {
                        len = strlen(buffer);
                        buffer[len - 1] = '\0';   // override \n to become \0
                    } else {
                        printf("wrong name...");
                        exit(-1);
                    }
                printf("Enter the title of the track: ");
                if (fgets(title, BUFFERSIZE , stdin) != NULL) {
                    len = strlen(title);
                    title[len - 1] = '\0';   // override \n to become \0
                    } else {
                        printf("wrong name...");
                        exit(-1);
                    }
                printf("Enter the duration (in seconds) of the track: ");    
                scanf("%d%c", &sec, &c1);  // use c to capture \n
                printf("[%s] [%s] [%d]\n", buffer, title,sec);
                insert(buffer,title, sec);
                break;
        case 2: if (head == NULL)
                  printf("List is Empty\n");
                else{
                    printf("Enter the name of the artist you wish to delete: ");
                    if (fgets(del, BUFFERSIZE , stdin) != NULL) {
                            len = strlen(del);
                            del[len - 1] = '\0';   // override \n to become \0
                        } else {
                            exit(-1);
                        }
                        printf("%s", del);
                        
                        mp3_t *temp;
                        temp = head;
                        while (temp != NULL) {
                            if( strcmp(temp->name, del) == 0){
                              delete(&head,temp);
                            }
                            temp = temp->next;    
                        }
                }   
                break;        
        case 3: if (head == NULL)
                  printf("List is Empty\n");
                else
                  print();
                break;
        case 4: if(head == NULL)
		  printf("List is Empty\n");
		else
		  printReverse();
                break;
        case 5: freeList();
                return 0;
        default: printf("Invalid option\n");
        }
    }
  }
  return 0;
}
