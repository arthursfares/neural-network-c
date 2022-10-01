#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <math.h>

#include "eldritch_arrays.h"

typedef struct parameters_s { matrix *W1, *b1, *W2, *b2; } parameters;

typedef struct layer_parameters_s { matrix *weights, *bias; } layer_parameters;

double linear_scaling_normalization(double x, double x_max, double x_min);
void scale_row_linearly(matrix *mat, unsigned int row, double max, double min);
parameters init_parameters(int n_features, int n_neurons, int n_output);
layer_parameters init_layer_parameters(unsigned int W_rows, unsigned int W_cols, unsigned int b_rows);
matrix *linear_function(matrix *W, matrix *X, matrix *b);
double sigmoid_function_d(double n);
matrix *sigmoid_function(matrix* Z);
float cost_function(matrix *S, matrix *y);
parameters fit(matrix *features, matrix* targets, int n_features, int n_neurons, int n_outputs, int n_iterations, float eta, float alpha);
matrix *predict(matrix *input, parameters learned_parameters);
layer_parameters *multilayer_fit(matrix *features, matrix* targets, size_t network_data_size, int *network_data, int n_iterations, float eta);
matrix *multilayer_predict(matrix *input, size_t network_data_size, layer_parameters *learned_parameters);

double linear_scaling_normalization(double x, double x_max, double x_min) {
    return (x - x_min) / (x_max - x_min);
}

void scale_row_linearly(matrix *mat, unsigned int row, double max, double min) {
    for (size_t col = 0; col < mat->n_cols; col++) {
        mat->data[row][col] = linear_scaling_normalization(mat->data[row][col], max, min);
    }
}

parameters init_parameters(int n_features, int n_neurons, int n_output) {
    matrix *W1 = create_random_matrix(n_neurons, n_features, 0, 1);
    matrix *b1 = create_random_matrix(n_neurons,          1, 0, 1);
    matrix *W2 = create_random_matrix(n_output,   n_neurons, 0, 1);
    matrix *b2 = create_random_matrix(n_output,           1, 0, 1);
    parameters initial_parameters = {W1, b1, W2, b2};
    return initial_parameters;
}

layer_parameters init_layer_parameters(unsigned int W_rows, unsigned int W_cols, unsigned int b_rows) {
    matrix *W = create_random_matrix(W_rows, W_cols, 0, 1);
    matrix *b = create_random_matrix(b_rows, 1,      0, 1);
    layer_parameters layer_params = {W, b};
    return layer_params;
}

matrix *linear_function(matrix *W, matrix *X, matrix *b) {
    return add_matrices(multiply_matrices(W, X), b);
}

double sigmoid_function_d(double n) {
    return (1 / (1 + exp(-n)));
}

matrix *sigmoid_function(matrix* Z) {
    matrix *S = create_matrix(Z->n_rows, Z->n_cols);
    for (int row = 0; row < S->n_rows; row++) {
        for (int col = 0; col < S->n_cols; col++) {
            S->data[row][col] = sigmoid_function_d(Z->data[row][col]);
        }
    }
    return S;
}

float cost_function(matrix *S, matrix *y) {
    float error, squared_error, sum_squared_error=0;
    for (int row = 0; row < S->n_rows; row++) {
        for (int col = 0; col < S->n_cols; col++) {
            error = S->data[row][col] - y->data[row][col];
            squared_error = error * error;
            sum_squared_error += squared_error;
        }
    }
    return 0.5f * sum_squared_error;
}

parameters fit(matrix *features, matrix* targets, int n_features, int n_neurons, int n_outputs, int n_iterations, float eta, float alpha) {
    matrix *X, *y, *Z1, *S1, *Z2, *S2;
    matrix *ones_S2, *ones_S1;
    matrix *delta2, *W2_gradients, *delta1, *W1_gradients; 
    matrix *new_W2, *new_W1;
    matrix *W2_variation, *W1_variation;
    parameters params = init_parameters(n_features, n_neurons, n_outputs);
    float errors[n_iterations];
    for (int iteration = 0; iteration < n_iterations; iteration++) {
        errors[iteration] = 0;
        for (int target_index = 0; target_index < targets->n_cols; target_index++) {
            X = get_col(features, target_index);
            y = get_col(targets, target_index);
            /* FEEDFORWARD */
            Z1 = linear_function(params.W1, X, params.b1);
            S1 = sigmoid_function(Z1);
            Z2 = linear_function(params.W2, S1, params.b2);
            S2 = sigmoid_function(Z2);
            /* COST CALCULATION */
            errors[iteration] += cost_function(S2, y);
            /* BACKPROPAGATION */
            // calculate output delta
            ones_S2 = create_ones_matrix(S2->n_rows, S2->n_cols);
            delta2 = hadamard_product(subtract_matrices(S2, y), hadamard_product(S2, subtract_matrices(ones_S2, S2)));
            //calculate hidden delta
            ones_S1 = create_ones_matrix(S1->n_rows, S1->n_cols);
            delta1 = hadamard_product(multiply_matrices(transpose(params.W2), delta2), hadamard_product(S1, subtract_matrices(ones_S1, S1)));
            // update output weights
            W2_gradients = multiply_matrices(delta2, transpose(S1));
            new_W2 = subtract_matrices(params.W2, multiply_by_scaler(W2_gradients, eta));
            W2_variation = subtract_matrices(new_W2, params.W2);
            params.W2 = add_matrices(new_W2, multiply_by_scaler(W2_variation, alpha));
            // update output bias
            params.b2 = subtract_matrices(params.b2, multiply_by_scaler(delta2, eta));
            // update hidden weights
            W1_gradients = multiply_matrices(delta1, transpose(X));
            new_W1 = subtract_matrices(params.W1, multiply_by_scaler(W1_gradients, eta));
            W1_variation = subtract_matrices(new_W1, params.W1);
            params.W1 = add_matrices(new_W1, multiply_by_scaler(W1_variation, alpha));
            // update hidden bias
            params.b1 = subtract_matrices(params.b1, multiply_by_scaler(delta1, eta));
            // free allocated memory !!!
            blast_matrix(X); blast_matrix(y);
            blast_matrix(Z1); blast_matrix(S1); blast_matrix(Z2); blast_matrix(S2);
            blast_matrix(ones_S2); blast_matrix(ones_S1);
            blast_matrix(delta2); blast_matrix(W2_gradients); blast_matrix(delta1); blast_matrix(W1_gradients);
            blast_matrix(new_W2); blast_matrix(new_W1);
            blast_matrix(W2_variation); blast_matrix(W1_variation);
        } // END of targets iterations
        printf("iteration %d -> error = %.4f\n", iteration, errors[iteration]);
    } // END of epochs iterations
    // return learned parameters
    return params;
}

matrix *predict(matrix *input, parameters learned_parameters) {
    matrix *Z1, *S1, *Z2, *S2, *result;
    Z1 = linear_function(learned_parameters.W1, input, learned_parameters.b1);
    S1 = sigmoid_function(Z1);
    Z2 = linear_function(learned_parameters.W2, S1, learned_parameters.b2);
    S2 = sigmoid_function(Z2);
    blast_matrix(Z1); blast_matrix(S1); blast_matrix(Z2); 
    return S2;
}

layer_parameters *multilayer_fit(matrix *features, matrix* targets, size_t network_data_size, int *network_data, int n_iterations, float eta) {
    size_t learned_parameters_size = network_data_size - 1;
    layer_parameters *learned_parameters = (layer_parameters*) malloc(learned_parameters_size * sizeof(layer_parameters));

    for (size_t param_index = 1; param_index < network_data_size; param_index++) { 
        learned_parameters[param_index-1] = init_layer_parameters(network_data[param_index], network_data[param_index-1], network_data[param_index]);
    }

    float errors[n_iterations];

    // ---
    for (int iteration = 0; iteration < n_iterations; iteration++) {
        errors[iteration] = 0;
        for (int target_index = 0; target_index < targets->n_cols; target_index++) {
            matrix *X = get_col(features, target_index);
            matrix *y = get_col(targets, target_index);

            size_t activations_size = network_data_size - 1;
            matrix **activations = (matrix**) malloc(activations_size * sizeof(matrix));

            /* FEEDFORWARD */
            for (size_t layer_index = 1; layer_index < network_data_size; layer_index++) {                
                matrix *Z;
                if (layer_index == 1) {
                    Z = linear_function(learned_parameters[layer_index-1].weights, X, learned_parameters[layer_index-1].bias);
                } else {
                    Z = linear_function(learned_parameters[layer_index-1].weights, activations[layer_index-2], learned_parameters[layer_index-1].bias);
                }
                activations[layer_index-1] = sigmoid_function(Z);
                blast_matrix(Z);
            }

            /* COST CALCULATION */
            errors[iteration] += cost_function(activations[activations_size-1], y);

            /* BACKPROPAGATION */
            
            /* calculate output delta */
            matrix *ones_output_activation = create_ones_matrix(activations[network_data_size-2]->n_rows, activations[network_data_size-2]->n_cols);
            matrix *output_delta = hadamard_product(subtract_matrices(activations[network_data_size-2], y), hadamard_product(activations[network_data_size-2], subtract_matrices(ones_output_activation, activations[network_data_size-2])));
            blast_matrix(ones_output_activation);

            /* calculate hidden deltas */
            size_t hidden_deltas_size = network_data_size-2;
            matrix **hidden_deltas = (matrix**) malloc(hidden_deltas_size * sizeof(matrix));
            size_t hidden_index = network_data_size - 3;
            do { 
                matrix *ones_hidden_activation = create_ones_matrix(activations[hidden_index]->n_rows, activations[hidden_index]->n_cols);
                if (hidden_index == network_data_size-3) {
                    hidden_deltas[hidden_index] = hadamard_product(multiply_matrices(transpose(learned_parameters[hidden_index+1].weights), output_delta), hadamard_product(activations[hidden_index], subtract_matrices(ones_hidden_activation, activations[hidden_index])));
                } else {
                    hidden_deltas[hidden_index] = hadamard_product(multiply_matrices(transpose(learned_parameters[hidden_index+1].weights), hidden_deltas[hidden_index+1]), hadamard_product(activations[hidden_index], subtract_matrices(ones_hidden_activation, activations[hidden_index])));
                }
                blast_matrix(ones_hidden_activation);
            } while (hidden_index--);

            /* update output weights */
            matrix *output_weights_gradient = multiply_matrices(output_delta, transpose(activations[network_data_size-3]));
            learned_parameters[network_data_size-2].weights = subtract_matrices(learned_parameters[network_data_size-2].weights, multiply_by_scaler(output_weights_gradient, eta));
            blast_matrix(output_weights_gradient);
            /* update output bias */
            learned_parameters[network_data_size-2].bias = subtract_matrices(learned_parameters[network_data_size-2].bias, multiply_by_scaler(output_delta, eta));
            
            /* update hidden parameters */
            hidden_index = network_data_size - 3;
            do {
                /* update hidden weights */
                matrix *hidden_weights_gradient;
                if (hidden_index == 0) {
                    hidden_weights_gradient = multiply_matrices(hidden_deltas[hidden_index], transpose(X));
                } else {
                    hidden_weights_gradient = multiply_matrices(hidden_deltas[hidden_index], transpose(activations[(network_data_size-2)-hidden_index-1]));
                }
                learned_parameters[hidden_index].weights = subtract_matrices(learned_parameters[hidden_index].weights, multiply_by_scaler(hidden_weights_gradient, eta));
                blast_matrix(hidden_weights_gradient);
                /* update hidden bias */
                learned_parameters[hidden_index].bias = subtract_matrices(learned_parameters[hidden_index].bias, multiply_by_scaler(hidden_deltas[hidden_index], eta));
            } while (hidden_index--);

            // free allocated memory
            blast_matrix(X); blast_matrix(y);
            for (size_t actv_index = 0; actv_index < network_data_size-1; actv_index++) {
                blast_matrix(activations[actv_index]);
            }
            free(activations);
            blast_matrix(output_delta);
            for (size_t hidden_idx = 0; hidden_idx < network_data_size-2; hidden_idx++) {
                blast_matrix(hidden_deltas[hidden_idx]);
            }    
            free(hidden_deltas);
        }
        printf("iteration %d -> error = %.4f\n", iteration, errors[iteration]);
    }
    // ---
    return learned_parameters; 
}

matrix *multilayer_predict(matrix *input, size_t network_data_size, layer_parameters *learned_parameters) {    
    size_t activations_size = network_data_size - 1;
    matrix **activations = (matrix**) malloc(activations_size * sizeof(matrix));
    for (size_t layer_index = 1; layer_index < network_data_size; layer_index++) {                
        matrix *Z;
        if (layer_index == 1) {
            Z = linear_function(learned_parameters[layer_index-1].weights, input, learned_parameters[layer_index-1].bias);
        } else {
            Z = linear_function(learned_parameters[layer_index-1].weights, activations[layer_index-2], learned_parameters[layer_index-1].bias);
        }
        activations[layer_index-1] = sigmoid_function(Z);
        blast_matrix(Z);
    }
    for (size_t actv_index = 0; actv_index < activations_size-1; actv_index++) {
        blast_matrix(activations[actv_index]);
    }
    return activations[activations_size-1];
}

#endif