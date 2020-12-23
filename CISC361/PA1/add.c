// Programming Assignment 1
// Sohan Gadiraju and Addison Kuykendall
#include "mp3.h"

extern mp3_t *head;

void insert(char *name, char *title, int sec)
{
  mp3_t *temp, *mp3;

  mp3 = (mp3_t *) malloc(sizeof(mp3_t));        // malloc space for MP3
  mp3->name = (char *) malloc(strlen(name) + 1);  // malloc space for name
  mp3->title = (char *) malloc(strlen(title) + 1);
  strcpy(mp3->name, name);                        // "assign" name via copy
  strcpy(mp3->title, title);
  mp3->duration = sec;
  mp3->next = NULL;

  if (head == NULL)
  {
    head = mp3;               // add the first MP3
  }
  else
  {
    temp = head;
    while (temp->next != NULL)
      temp = temp->next;
    temp->next = mp3;         // append to the tail/end
    mp3->prev = temp;
  }
}
void delete(mp3_t** head_ref, mp3_t* mp3)  
{
    if (*head_ref == NULL || mp3 == NULL)  
        return;  

    if (*head_ref == mp3)
        *head_ref = mp3->next;

    if (mp3->next != NULL){  
        mp3->next->prev = mp3->prev;
    }
  
    if (mp3->prev != NULL){  
        mp3->prev->next = mp3->next;
    }

    free(mp3->name);
    free(mp3->title);
    free(mp3);  
    return;  
}  
