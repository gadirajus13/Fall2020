#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
char buf[16];
int value;
int fd1[2];
pipe(fd1);
if (!fork()) {
int fd2[2];
pipe(fd2);
if (!fork()) {
close(1);
dup(fd2[1]);
close(fd2[0]);
write(fd2[1], "37", 3);
exit(0);
} else {
close(0);
dup(fd2[0]);
close(fd2[1]);
close(1);
dup(fd1[1]);
close(fd1[0]);
read(fd2[0], buf, 16);
value = atoi(buf) + 3;
sprintf(buf, "%d", value);
write(fd1[1], buf, sizeof(buf));
}
} else {
close(0);
dup(fd1[0]);
close(fd1[1]);
read(fd1[0], buf, 16);
printf("[%s]\n", buf);
}
return 0;
}