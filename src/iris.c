#include <time.h>
#include <stdbool.h>
#include <string.h>
#include "neural_network.h"

const int   N_FEATURES = 4;
const int   N_NEURONS  = 4;
const int   N_OUTPUT   = 1;
const int   ITERATIONS = 3000;
const float ETA        = 0.1;
const float ALPHA      = 0.1;

bool is_file_open(FILE *fp) {
    if (fp == NULL) {
        printf("[!] Could not open file. Terminating program.");
        return false;
    }
    return true;
}

int main(int argc, const char *argv[]) {
    srand(time(0));

    /* load and prepare the dataset */
    // open de data file for reading
    FILE *iris_fp = fopen("data/iris.data", "r");
    if (!is_file_open(iris_fp)) return -1;
    // we want to store, for 150 flowers, 4 different data, 
    // the petal's and sepal's widths and heights
    matrix *features = create_matrix(4, 150); // each column contains all the data for one flower
    // the flowers can be setosa (0), versicolor (1) or virginica (2)
    matrix *targets = create_matrix(1, 150);
    // a line utility array will be used to go through the lines of the data file
    // and a string pointer will go through each of the lines comma separeted values.
    char line[50];
    char *sp;
    // now, we read the data and store the wanted values
    double current_sepal_lenght, current_sepal_width, current_petal_length, current_petal_width, current_classification;
    char *current_flower_name;
    int current_line_index=0;
    while ( fgets(line, 50, iris_fp) != NULL ) {
        // sepal length
        sp = strtok(line, ","); 
        current_sepal_lenght = atof(sp);
        features->data[0][current_line_index] = current_sepal_lenght;
        // sepal width
        sp = strtok(NULL, ",");
        current_sepal_width = atof(sp);
        features->data[1][current_line_index] = current_sepal_width;
        // petal length
        sp = strtok(NULL, ",");
        current_petal_length = atof(sp);
        features->data[2][current_line_index] = current_petal_length;
        // petal width
        sp = strtok(NULL, ",");
        current_petal_width = atof(sp);
        features->data[3][current_line_index] = current_petal_width;
        // classification
        sp = strtok(NULL, ",");
        current_flower_name = strdup(sp);
        if ( strcmp(current_flower_name, "Iris-setosa\n") == 0 ) current_classification = 0.0;
        else if ( strcmp(current_flower_name, "Iris-versicolor\n") == 0 ) current_classification = 1.0;
        else if ( strcmp(current_flower_name, "Iris-virginica\n") == 0 ) current_classification = 2.0;
        else printf("[!] found no match for flower name\n");
        targets->data[0][current_line_index] = current_classification;
        //update line index
        current_line_index++;
    } // END csv read loop

    // TODO. shffle data.
    // TODO. standardization
    // TODO. split data between train and test data.

    /* ---------------------------- */
    
    /* TRAIN */
    parameters learned_parameters = fit(features, targets, N_FEATURES, N_NEURONS, N_OUTPUT, ITERATIONS, ETA, ALPHA);
    
    printf("\n\n----------------------------------------------\n\n");

    /* TEST */
    matrix *input, *output, *result;
    input = get_col(features, 94);
    output = predict(input, learned_parameters);
    // round the output to be equivalent to the classifications
    result = create_matrix(output->n_rows, output->n_cols);
    for (size_t row = 0; row < output->n_rows; row++) {
        for (size_t col = 0; col < output->n_cols; col++) {
            result->data[row][col] = round(output->data[row][col]);
        }
    }
    // show results
    printf("input="); print_matrix(input);
    printf("result="); print_matrix(result);
    printf("----------------------------------------------\n\n");

    // free allocated memory
    free_matrix(input);
    free_matrix(output);
    free_matrix(result);
    free_matrix(learned_parameters.W2);
    free_matrix(learned_parameters.b1);
    free_matrix(learned_parameters.W1);
    free_matrix(learned_parameters.b2);
    free_matrix(targets);
    free_matrix(features);

    return 0;
}