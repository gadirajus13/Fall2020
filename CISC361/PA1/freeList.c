// Programming Assignment 1
// Sohan Gadiraju
#include "mp3.h"

extern mp3_t *head;

void freeList()
{
  mp3_t *temp;
  int  i = 0;

  while (head != NULL) {
    temp = head;
    head = head->next; // point to next MP3 record
    free(temp->name);  // first free name inside MP3 record
    free(temp->title);
    free(temp);        // then free MP3 record
    i++;
  }
  printf("free %d MP3 records...\n", i);
}
