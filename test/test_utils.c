#include "stdio.h"
#include "../include/constants.h"
#include "../src/utils.h"

int main(){

    printf("Test is_boundary():\n");

    for(int k = 0; k < DEPTH; k++){
        for(int j = 0; j < HEIGHT; j++){
            for(int i = 0; i < WIDTH; i++){
                if(!is_boundary(i,j,k)){
                    printf("Index %zu is not at boundary (%d,%d,%d)\n", rowmaj_idx(i,j,k), i,j,k);
                }
            }
        }
    }

    return 0;
}








