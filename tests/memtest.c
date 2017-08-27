#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

int main(){
	int*d = (int*)malloc(sizeof(char)*1024*1024*1024*12);
	memset(d, 0, sizeof(char)*1024*1024*1024*12);
	int i = 0;
	unsigned long range = 3221225472;
	while(1){
		srand(time(NULL));   // should only be called once
		unsigned long i = rand()*range;	
		d[i] = 1;
		sleep(1);
	}
}
