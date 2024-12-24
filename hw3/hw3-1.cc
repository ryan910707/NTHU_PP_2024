#include <pthread.h>
#include <stdio.h>
#include <algorithm>
#include <string.h>
#include <omp.h>
using namespace std;

int V,E;
int *matrix;



void read_input(char* input_file){
    FILE* file = fopen(input_file, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    matrix = (int*)malloc(sizeof(int)*V*V);

    for(int i=0;i<V;i++){
        for(int j=0;j<V;j++){
            if(i==j){
                matrix[i*V+j]=0;
            }
            else {
                matrix[i*V+j]=1073741823;
            }
        }
    }

    for(int i=0;i<E;i++){
        int tmp[3];
        fread(tmp, sizeof(int), 3, file);
        matrix[tmp[0]*V+tmp[1]]=tmp[2];
    }
    fclose(file);
}

void floyd(){

    for(int k=0;k<V;k++){
        #pragma omp parallel for schedule(dynamic) 
        for(int i=0;i<V;i++){
            #pragma omp simd
            for(int j=0;j<V;j++){
                matrix[i*V+j]=  min(matrix[i*V+j],matrix[i*V+k]+matrix[k*V+j]);
            }
        }
    }

}
//
void output(char* output_file){
    FILE* file = fopen(output_file, "w");
    for(int i=0;i<V;i++){
        fwrite(matrix+(i*V), sizeof(int),  V, file);
    }
	fclose(file);
}

int main(int argc, char** argv){
    char* input_file = argv[1];
    char* output_file = argv[2];
    read_input(input_file);
    floyd();
    output(output_file);
}