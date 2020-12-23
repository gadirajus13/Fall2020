// Programming Assignment 1
// Sohan Gadiraju and Addison Kuykendall
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
typedef struct mp3
{
  char *name;
  char *title;
  int duration;    
  struct mp3 *next;
  struct mp3 *prev;
} mp3_t; 
