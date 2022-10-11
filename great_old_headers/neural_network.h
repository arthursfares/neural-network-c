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
parameters fit(matrix *features, matrix* targets, int n_features, int n_neurons, int n_outputs, int n_iterations, float eta);
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
    matrix *W_times_X = multiply_matrices(W, X);
    matrix *result = add_matrices(W_times_X, b);
    blast_matrix(W_times_X);
    return result;
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



#pragma region FIT_HELPER_FUNCTIONS

matrix *calculate_output_delta(matrix *S2, matrix *y) {
    matrix *ones_S2 = create_ones_matrix(S2->n_rows, S2->n_cols);
    matrix *ones_S2_minus_S2 = subtract_matrices(ones_S2, S2);
    matrix *sigmoid_derivative = hadamard_product(S2, ones_S2_minus_S2);
    matrix *S2_minus_y = subtract_matrices(S2, y);
    matrix *output_delta = hadamard_product(S2_minus_y, sigmoid_derivative);
    blast_matrix(ones_S2); blast_matrix(ones_S2_minus_S2); blast_matrix(sigmoid_derivative); blast_matrix(S2_minus_y);
    return output_delta;
}

matrix *calculate_hidden_delta(matrix *S1, matrix *W2, matrix *delta2) {
    matrix *ones_S1 = create_ones_matrix(S1->n_rows, S1->n_cols);
    matrix *ones_S1_minus_S1 = subtract_matrices(ones_S1, S1);
    matrix *sigmoid_derivative = hadamard_product(S1, ones_S1_minus_S1);
    matrix *W2_transpose = transpose(W2);
    matrix *W2_transpose_times_delta2 = multiply_matrices(W2_transpose, delta2);
    matrix *hidden_delta = hadamard_product(W2_transpose_times_delta2, sigmoid_derivative);
    blast_matrix(ones_S1); blast_matrix(ones_S1_minus_S1); blast_matrix(sigmoid_derivative);
    blast_matrix(W2_transpose); blast_matrix(W2_transpose_times_delta2);
    return hidden_delta;
}

#pragma endregion FIT_HELPER_FUNCTIONS

parameters fit(matrix *features, matrix* targets, int n_features, int n_neurons, int n_outputs, int n_iterations, float eta) {
    matrix *X, *y, *Z1, *S1, *Z2, *S2;
    matrix *delta2, *W2_gradients, *delta1, *W1_gradients; 
    matrix *S1_transpose, *X_transpose;
    matrix *W2_gradients_scaled, *delta2_scaled, *W1_gradients_scaled, *delta1_scaled;
    matrix *old_W2, *old_b2, *old_W1, *old_b1;
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
            delta2 = calculate_output_delta(S2, y);
            //calculate hidden delta
            delta1 = calculate_hidden_delta(S1, params.W2, delta2);
            // update output weights
            S1_transpose = transpose(S1);
            W2_gradients = multiply_matrices(delta2, S1_transpose);
            W2_gradients_scaled = multiply_by_scaler(W2_gradients, eta);
            old_W2 = params.W2;
            params.W2 = subtract_matrices(old_W2, W2_gradients_scaled);
            // update output bias
            delta2_scaled = multiply_by_scaler(delta2, eta);
            old_b2 = params.b2;
            params.b2 = subtract_matrices(old_b2, delta2_scaled);
            // update hidden weights
            X_transpose = transpose(X);
            W1_gradients = multiply_matrices(delta1, X_transpose);
            W1_gradients_scaled = multiply_by_scaler(W1_gradients, eta);
            old_W1 = params.W1;
            params.W1 = subtract_matrices(old_W1, W1_gradients_scaled);
            // update hidden bias
            delta1_scaled = multiply_by_scaler(delta1, eta);
            old_b1 = params.b1;
            params.b1 = subtract_matrices(old_b1, delta1_scaled);
            // free allocated memory !!!
            blast_matrix(old_W2); blast_matrix(old_b2); blast_matrix(old_W1); blast_matrix(old_b1);
            blast_matrix(X); blast_matrix(y);
            blast_matrix(Z1); blast_matrix(S1); blast_matrix(Z2); blast_matrix(S2);
            blast_matrix(delta2); 
            blast_matrix(delta1);
            blast_matrix(W2_gradients); blast_matrix(W1_gradients);
            blast_matrix(S1_transpose); blast_matrix(X_transpose);
            blast_matrix(W2_gradients_scaled), blast_matrix(delta2_scaled), blast_matrix(W1_gradients_scaled), blast_matrix(delta1_scaled);
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
            matrix *output_delta = calculate_output_delta(activations[network_data_size-2], y);

            /* calculate hidden deltas */
            size_t hidden_deltas_size = network_data_size-2;
            matrix **hidden_deltas = (matrix**) malloc(hidden_deltas_size * sizeof(matrix));
            size_t hidden_index = network_data_size - 3;
            do { 
                if (hidden_index == network_data_size-3) {
                    hidden_deltas[hidden_index] = calculate_hidden_delta(activations[hidden_index], learned_parameters[hidden_index+1].weights, output_delta);
                } else {
                    hidden_deltas[hidden_index] = calculate_hidden_delta(activations[hidden_index], learned_parameters[hidden_index+1].weights, hidden_deltas[hidden_index+1]);
                }
            } while (hidden_index--);

            /* update output weights */
            matrix *prev_activation_transpose = transpose(activations[network_data_size-3]);
            matrix *output_weights_gradient = multiply_matrices(output_delta, prev_activation_transpose);
            matrix *output_weights_gradient_scaled = multiply_by_scaler(output_weights_gradient, eta);
            matrix *old_output_weights = learned_parameters[network_data_size-2].weights;
            learned_parameters[network_data_size-2].weights = subtract_matrices(old_output_weights, output_weights_gradient_scaled);
            blast_matrix(old_output_weights); blast_matrix(output_weights_gradient_scaled); blast_matrix(output_weights_gradient); blast_matrix(prev_activation_transpose);
            /* update output bias */
            matrix *output_delta_scaled = multiply_by_scaler(output_delta, eta);
            matrix *old_output_bias = learned_parameters[network_data_size-2].bias;
            learned_parameters[network_data_size-2].bias = subtract_matrices(old_output_bias, output_delta_scaled);
            blast_matrix(old_output_bias); blast_matrix(output_delta_scaled);
            
            /* update hidden parameters */
            hidden_index = network_data_size - 3;
            do {
                /* update hidden weights */
                matrix *input_transpose, *hidden_weights_gradient, *hidden_weights_gradient_scaled, *old_hidden_weights;
                if (hidden_index == 0) {
                    input_transpose = transpose(X);
                    hidden_weights_gradient = multiply_matrices(hidden_deltas[hidden_index], input_transpose);
                } else {
                    input_transpose = transpose(activations[(network_data_size-2)-hidden_index-1]);
                    hidden_weights_gradient = multiply_matrices(hidden_deltas[hidden_index], input_transpose);
                }
                hidden_weights_gradient_scaled = multiply_by_scaler(hidden_weights_gradient, eta);
                old_hidden_weights = learned_parameters[hidden_index].weights;
                learned_parameters[hidden_index].weights = subtract_matrices(old_hidden_weights, hidden_weights_gradient_scaled);
                blast_matrix(old_hidden_weights); blast_matrix(hidden_weights_gradient_scaled); blast_matrix(hidden_weights_gradient); blast_matrix(input_transpose);
                /* update hidden bias */
                matrix *hidden_deltas_scaled = multiply_by_scaler(hidden_deltas[hidden_index], eta);
                matrix *old_hidden_bias = learned_parameters[hidden_index].bias;
                learned_parameters[hidden_index].bias = subtract_matrices(old_hidden_bias, hidden_deltas_scaled);
                blast_matrix(old_hidden_bias); blast_matrix(hidden_deltas_scaled);
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
    matrix *output = copy_matrix(activations[activations_size-1]);
    for (size_t actv_index = 0; actv_index < activations_size; actv_index++) {
        blast_matrix(activations[actv_index]);
    }
    free(activations);
    return output;
}

#endif