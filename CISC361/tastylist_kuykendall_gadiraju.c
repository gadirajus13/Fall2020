// Addison Kuykendall and Sohan Gadiraju
#include<stdio.h>
#include<stdlib.h>

struct node
{
    int data;
    struct node *next;
}*head;

struct node;

void append(int num)
{
	struct node *temp,*right;
	temp = (struct node *)malloc(sizeof(struct node));
	temp->data=num;
	right = head;
	while(right->next != NULL)
		right=right->next;
	right->next = temp;
}

void add(int num)
{
	struct node *temp;
	temp=(struct node *)malloc(sizeof(struct node));
	temp->data=num;
	temp->next = head;
	head = temp;
}
void addafter(int num, int loc)
{
	int i;
	struct node *temp,*left,*right;
	right=head;
	for(i=1;i<loc;i++)
	{
		left=right;
		right=right->next;
	}
	temp=(struct node *)malloc(sizeof(struct node));
	temp->data=num;
	left->next=temp;
	left=temp;
	left->next=right;
	return;
}


int delete(int num)
{
    struct node **indirect;
    struct node *tmp;
    indirect = &head;

    while (*indirect) {
        if ((*indirect)->data == num) {
            tmp = *indirect;
            *indirect = (*indirect)->next;
            free(tmp);
            return 1;
        }
        indirect = &(*indirect)->next;
    }
    
    return 0;
}

void  display(struct node *r)
{
	r=head;
	while(r!=NULL)
	{
		printf("%d ",r->data);
		r=r->next;
	}
	printf("\n");
}


int count()
{
	struct node *n;
	int c=0;
	n=head;
	while(n!=NULL)
	{
		n=n->next;
		c++;
	}
	return c;
}

void insert(int num)
{
	int c=0;
	struct node *temp;
	temp=head;
	while(temp!=NULL)
	{
		if(temp->data<num){
			c++;
			temp=temp->next;
		}else{
			break;
		}
	}
	if(c==0)
		add(num);
	else
		addafter(num,++c);
}

void freeNodes(struct node* head){
  struct node* temp;
  while(head!=NULL){
    temp = head;
    head = head->next;
    delete(temp->data);
  }
}

int  main()
{
	int i,num;
	struct node *n;
	head=NULL;
	while(1)
	{
		printf("\nList Operations\n");
		printf("===============\n");
		printf("1.Insert\n");
		printf("2.Display\n");
		printf("3.Size\n");
		printf("4.Delete\n");
		printf("5.Exit\n");
		printf("Enter your choice : ");
		if(scanf("%d",&i)<=0){
			printf("Enter only an Integer\n");
			exit(0);
		} else {
			switch(i)
			{
			case 1:      printf("Enter the number to insert : ");
			scanf("%d",&num);
			insert(num);
			break;
			case 2:     if(head==NULL)
			{
				printf("List is Empty\n");
			}
			else
			{
				printf("Element(s) in the list are : ");
			}
			display(n);
			break;
			case 3:     printf("Size of the list is %d\n",count());
			break;
			case 4:     if(head==NULL)
				printf("List is Empty\n");
			else{
				printf("Enter the number to delete : ");
				scanf("%d",&num);
				if(delete(num))
					printf("%d deleted successfully\n",num);
				else
					printf("%d not found in the list\n",num);
			}
			break;
			case 5:
			  freeNodes(head);
			  return 0;
			default:    printf("Invalid option\n");
			}
		}
	}
	return 0;
}


 

