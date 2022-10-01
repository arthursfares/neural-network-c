#include <time.h>
#include "neural_network.h"

#define N_LAYERS 3

const int    ITERATIONS = 3000;
const float  ETA        = 0.5;

int main(int argc, const char *argv[]) {
    srand(time(0));

    matrix *features = read_matrix_from_file("matrices/features.mat");
    printf("features="); print_matrix(features);

    matrix *targets = read_matrix_from_file("matrices/targets.mat");
    printf("targets="); print_matrix(targets);

    // [ N_FEATURES | N_HIDDEN_1 | N_OUTPUT ]
    int network_data[N_LAYERS] = {2, 2, 1};
    
    /* TRAIN */
    layer_parameters *learned_parameters = multilayer_fit(features, targets, N_LAYERS, network_data, ITERATIONS, ETA);
    
    printf("\n\n----------------------------------------------\n\n");

    /* TEST */
    matrix *input;
    matrix *output, *result;
    for (int col = 0; col < features->n_cols; col++) {
        input = get_col(features, col);
        output = multilayer_predict(input, N_LAYERS, learned_parameters);
        result = create_matrix(output->n_rows, output->n_cols);
        for (int row = 0; row < result->n_rows; row++) {
            for (int col = 0; col < result->n_cols; col++) {
                result->data[row][col] = (output->data[row][col] >= 0.5) ? 1.0 : 0.0;
            }
        }
        printf("input="); print_matrix(input);
        printf("result="); print_matrix(result);
        printf("----------------------------------------------\n\n");
        blast_matrix(result);
    }


    for (size_t param_index = 0; param_index < N_LAYERS-1; param_index++){
        blast_matrix(learned_parameters[param_index].weights);
        blast_matrix(learned_parameters[param_index].bias);
    }
    free(learned_parameters);
    blast_matrix(input);
    blast_matrix(output);
    blast_matrix(targets);
    blast_matrix(features);

    return 0;
}