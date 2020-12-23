// Programming Assignment 1
// Sohan Gadiraju and Addison Kuykendall
#include "mp3.h"

extern mp3_t *head;

void print()
{
  mp3_t *temp;
  int  i = 0;

  temp = head;

  while (temp != NULL) {
    printf("(%d)--%s--%s---%d\n", ++i, temp->name, temp->title,temp->duration);
    temp = temp->next;
  }
}
void printReverse()
{
  mp3_t *temp;
  int i =0;
  temp = head;
  while(temp->next!=NULL){
    temp=temp->next;
  }
  while(temp!=NULL){
    printf("(%d)--%s--%s---%d\n", ++i, temp->name, temp->title, temp->duration);
    temp = temp->prev;
  }
}
