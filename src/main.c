#include <time.h>
#include "neural_network.h"

const int   N_FEATURES = 2;
const int   N_NEURONS  = 2;
const int   N_OUTPUT   = 1;
const int   ITERATIONS = 5000;
const float ETA        = 0.1;
const float ALPHA      = 0.5;

int main(int argc, const char *argv[]) {
    srand(time(0));

    matrix *features = read_matrix_from_file("matrices/features.mat");
    printf("features="); print_matrix(features);

    matrix *targets = read_matrix_from_file("matrices/targets.mat");
    printf("targets="); print_matrix(targets);
    
    /* TRAIN */
    parameters learned_parameters = fit(features, targets, N_FEATURES, N_NEURONS, N_OUTPUT, ITERATIONS, ETA, ALPHA);
    
    printf("\n\n----------------------------------------------\n\n");

    /* TEST */
    matrix *input;
    matrix *result;
    for (int col = 0; col < features->n_cols; col++) {
        input = get_col(features, col);
        result = predict(input, learned_parameters);
        printf("input="); print_matrix(input);
        printf("result="); print_matrix(result);
        printf("----------------------------------------------\n\n");
    }

    free_matrix(input);
    free_matrix(result);
    free_matrix(learned_parameters.W2);
    free_matrix(learned_parameters.b1);
    free_matrix(learned_parameters.W1);
    free_matrix(learned_parameters.b2);
    free_matrix(targets);
    free_matrix(features);

    return 0;
}