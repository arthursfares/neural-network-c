#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <math.h>

#include "eldritch_arrays.h"

typedef struct parameters_s { matrix *W1, *b1, *W2, *b2; } parameters;

parameters init_parameters(int n_features, int n_neurons, int n_output);
matrix *linear_function(matrix *W, matrix *X, matrix *b);
double sigmoid_function_d(double n);
matrix *sigmoid_function(matrix* Z);
float cost_function(matrix *S, matrix *y);
parameters fit(matrix *features, matrix* targets, int n_features, int n_neurons, int n_outputs, int n_iterations, float eta, float alpha);
matrix *predict(matrix *input, parameters learned_parameters);


parameters init_parameters(int n_features, int n_neurons, int n_output) {
    matrix *W1 = create_random_matrix(n_neurons, n_features, 0, 1);
    matrix *b1 = create_random_matrix(n_neurons,          1, 0, 1);
    matrix *W2 = create_random_matrix(n_output,   n_neurons, 0, 1);
    matrix *b2 = create_random_matrix(n_output,           1, 0, 1);
    parameters initial_parameters = {W1, b1, W2, b2};
    return initial_parameters;
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
            // update output weights
            ones_S2 = create_ones_matrix(S2->n_rows, S2->n_cols);
            delta2 = hadamard_product(subtract_matrices(S2, y), hadamard_product(S2, subtract_matrices(ones_S2, S2)));
            W2_gradients = multiply_matrices(delta2, transpose(S1));
            new_W2 = subtract_matrices(params.W2, multiply_by_scaler(W2_gradients, eta));
            W2_variation = subtract_matrices(new_W2, params.W2);
            params.W2 = add_matrices(new_W2, multiply_by_scaler(W2_variation, alpha));
            // update output bias
            params.b2 = subtract_matrices(params.b2, multiply_by_scaler(delta2, eta));
            // update hidden weights
            ones_S1 = create_ones_matrix(S1->n_rows, S1->n_cols);
            delta1 = hadamard_product(multiply_matrices(transpose(params.W2), delta2), hadamard_product(S1, subtract_matrices(ones_S1, S1)));
            W1_gradients = multiply_matrices(delta1, transpose(X));
            new_W1 = subtract_matrices(params.W1, multiply_by_scaler(W1_gradients, eta));
            W1_variation = subtract_matrices(new_W1, params.W1);
            params.W1 = add_matrices(new_W1, multiply_by_scaler(W1_variation, alpha));
            // update hidden bias
            params.b1 = subtract_matrices(params.b1, multiply_by_scaler(delta1, eta));
            // free allocated memory !!!
            free_matrix(X); free_matrix(y);
            free_matrix(Z1); free_matrix(S1); free_matrix(Z2); free_matrix(S2);
            free_matrix(ones_S2); free_matrix(ones_S1);
            free_matrix(delta2); free_matrix(W2_gradients); free_matrix(delta1); free_matrix(W1_gradients);
            free_matrix(new_W2); free_matrix(new_W1);
            free_matrix(W2_variation); free_matrix(W1_variation);
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
    result = create_matrix(S2->n_rows, S2->n_cols);
    for (int row = 0; row < result->n_rows; row++) {
        for (int col = 0; col < result->n_cols; col++) {
            result->data[row][col] = (S2->data[row][col] >= 0.5) ? 1.0 : 0.0;
        }
    }
    free_matrix(Z1); free_matrix(S1); free_matrix(Z2); free_matrix(S2);
    return result;
}

#endif