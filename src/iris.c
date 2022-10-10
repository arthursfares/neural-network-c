#include <time.h>
#include <stdbool.h>
#include <string.h>
#include "neural_network.h"

const int   N_FEATURES = 4;
const int   N_NEURONS  = 4;  // test (N_FEATURES/2)+1 for better results
const int   N_OUTPUTS  = 3;
const int   ITERATIONS = 4000;
const float ETA        = 0.15;

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
    matrix *features = create_matrix(N_FEATURES, 150); // each column contains all the data for one flower
    // the flowers can be setosa (0), versicolor (1) or virginica (2)
    matrix *targets = create_matrix(N_OUTPUTS, 150);
    // a line utility array will be used to go through the lines of the data file
    // and a string pointer will go through each of the lines comma separeted values.
    matrix *iris_data = create_matrix(N_FEATURES+N_OUTPUTS, 150);
    char line[50];
    char *sp;
    // now, we read the data and store the wanted values
    double current_sepal_length, current_sepal_width, current_petal_length, current_petal_width;
    double min_sepal_length=__DBL_MAX__, min_sepal_width=__DBL_MAX__, min_petal_length=__DBL_MAX__, min_petal_width=__DBL_MAX__;
    double max_sepal_length=0, max_sepal_width=0, max_petal_length=0, max_petal_width=0;
    // mins and maxes used for latter normalization
    double setosa, versicolor, virginica;
    char *current_flower_name;
    int current_line_index=0;
    while ( fgets(line, 50, iris_fp) != NULL ) {
        // sepal length
        sp = strtok(line, ","); 
        current_sepal_length = atof(sp);
        iris_data->data[0][current_line_index] = current_sepal_length;
        if (current_sepal_length < min_sepal_length) min_sepal_length = current_sepal_length;
        if (current_sepal_length > max_sepal_length) max_sepal_length = current_sepal_length; 
        // sepal width
        sp = strtok(NULL, ",");
        current_sepal_width = atof(sp);
        iris_data->data[1][current_line_index] = current_sepal_width;
        if (current_sepal_width < min_sepal_width) min_sepal_width = current_sepal_width;
        if (current_sepal_width > max_sepal_width) max_sepal_width = current_sepal_width;
        // petal length
        sp = strtok(NULL, ",");
        current_petal_length = atof(sp);
        iris_data->data[2][current_line_index] = current_petal_length;
        if (current_petal_length < min_petal_length) min_petal_length = current_petal_length;
        if (current_petal_length > max_petal_length) max_petal_length = current_petal_length;
        // petal width
        sp = strtok(NULL, ",");
        current_petal_width = atof(sp);
        iris_data->data[3][current_line_index] = current_petal_width;
        if (current_petal_width < min_petal_width) min_petal_width = current_petal_width;
        if (current_petal_width > max_petal_width) max_petal_width = current_petal_width;
        // classification
        sp = strtok(NULL, ",");
        current_flower_name = strdup(sp);
        setosa = ( strcmp(current_flower_name, "Iris-setosa\n") == 0 ) ? 1.0 : 0.0;
        versicolor = ( strcmp(current_flower_name, "Iris-versicolor\n") == 0 ) ? 1.0 : 0.0;
        virginica = ( strcmp(current_flower_name, "Iris-virginica\n") == 0 ) ? 1.0 : 0.0;
        iris_data->data[4][current_line_index] = setosa;
        iris_data->data[5][current_line_index] = versicolor;
        iris_data->data[6][current_line_index] = virginica;
        //update line index
        current_line_index++;
    } // END csv read loop

    printf("sepal length....\t MIN %.1f\t MAX %.1f\n", min_sepal_length, max_sepal_length);
    printf("sepal width.....\t MIN %.1f\t MAX %.1f\n", min_sepal_width, max_sepal_width);
    printf("petal length....\t MIN %.1f\t MAX %.1f\n", min_petal_length, max_petal_length);
    printf("petal width.....\t MIN %.1f\t MAX %.1f\n", min_petal_width, max_petal_width);

    save_matrix_to_file("matrices/iris/iris_data.mat", iris_data);

    // SHUFFLE iris_data
    // ---
    // matrix *permutation_matrix = create_permutation_matrix(iris_data->n_cols, iris_data->n_rows);
    // // M x prmt = M w/ cols swipped
    matrix *iris_data_shuffled = copy_matrix(iris_data);
    for (size_t col = 0; col < iris_data_shuffled->n_cols; col++) {
        size_t rand_col = col + rand() / (RAND_MAX / (iris_data_shuffled->n_cols - col) + 1);
        for (size_t row = 0; row < iris_data_shuffled->n_rows; row++) {
            double temp = iris_data_shuffled->data[row][rand_col];
            iris_data_shuffled->data[row][rand_col] = iris_data_shuffled->data[row][col];
            iris_data_shuffled->data[row][col] = temp;
        }
    }
    save_matrix_to_file("matrices/iris/iris_data_shuffled.mat", iris_data_shuffled);
    // free_matrix(permutation_matrix);
    blast_matrix(iris_data);

    // GET features AND targets from iris_data_shuffled
    // ---
    features = get_sub_matrix(iris_data_shuffled, 0, N_FEATURES-1, 0, 149);
    save_matrix_to_file("matrices/iris/iris_features.mat", features);
    targets = get_sub_matrix(iris_data_shuffled, N_FEATURES, N_FEATURES+N_OUTPUTS-1, 0 , 149);
    save_matrix_to_file("matrices/iris/iris_targets.mat", targets);

    // NORMALIZATION (linear scaling)
    // x_std = (x - X_MIN) / (X_MAX - X_MIN)
    // ---
    scale_row_linearly(features, 0, max_sepal_length, min_sepal_length);
    scale_row_linearly(features, 1, max_sepal_width, min_sepal_width);
    scale_row_linearly(features, 2, max_petal_length, min_petal_length);
    scale_row_linearly(features, 3, max_petal_width, min_petal_width);
    save_matrix_to_file("matrices/iris/iris_features.mat", features);

    // SPLIT DATA BETWEEN TRAIN AND TEST DATA
    // ---
    matrix *features_train = get_sub_matrix(features, 0, features->n_rows-1, 0, features->n_cols-31);
    matrix *targets_train = get_sub_matrix(targets, 0, targets->n_rows-1, 0, targets->n_cols-31);
    save_matrix_to_file("matrices/iris/train/features_train.mat", features_train);
    save_matrix_to_file("matrices/iris/train/targets_train.mat", targets_train);
    matrix *features_test = get_sub_matrix(features, 0, features->n_rows-1, features->n_cols-30, features->n_cols-1);
    matrix *targets_test = get_sub_matrix(targets, 0, targets->n_rows-1, targets->n_cols-30, targets->n_cols-1);
    save_matrix_to_file("matrices/iris/test/features_test.mat", features_test);
    save_matrix_to_file("matrices/iris/test/targets_test.mat", targets_test);

    // ---
    
    /* TRAIN */
    parameters learned_parameters = fit(features_train, targets_train, N_FEATURES, N_NEURONS, N_OUTPUTS, ITERATIONS, ETA);
    blast_matrix(features_train);
    blast_matrix(targets_train);

    printf("\n\n----------------------------------------------\n\n");

    /* TEST */
    // for every test case, make a prediction
    // print how many wrong predictions were made
    int mistakes = 0;
    double loss = 0;
    matrix *input, *output, *result, *target_col;
    for (size_t col = 0; col < features_test->n_cols; col++) {
        printf("test %d --------- \n\n", col);
        target_col = get_col(targets_test, col);
        input = get_col(features_test, col);
        output = predict(input, learned_parameters);
        result = create_matrix(output->n_rows, output->n_cols);

        // get the index of the biggest result and ground truth index 
        int index_of_highest = 0;
        int index_of_ground_truth = 0;
        double highest_value = output->data[0][0];
        for (size_t i = 1; i < 3; i++) {
            if (output->data[i][0] > highest_value) {
                index_of_highest = i;
                highest_value = output->data[i][0];
            }
            if (target_col->data[i][0] == 1.0) index_of_ground_truth = i;
        }
        for (size_t i = 0; i < 3; i++) {
            result->data[i][0] = (output->data[i][0] == highest_value) ? 1.0 : 0.0;
        }

        // calculate mistakes for accuracy
        if (!are_matrices_equal(result, target_col)) {
            mistakes++;
            printf("\nmistake ate test %d\n", col);
        }

        // add up to loss
        double current_loss_entry = output->data[index_of_ground_truth][0];
        loss += log(current_loss_entry);
        
        // printf("output="); print_matrix(output);
        printf("result="); print_matrix(result);
        printf("target="); print_matrix(target_col);

        blast_matrix(result);
        blast_matrix(target_col);
        blast_matrix(input);
        blast_matrix(output);
    }
    blast_matrix(features_test);
    blast_matrix(targets_test);
    
    printf("\n\n----------------------------------------------\n\n");
    printf("\n\naccuracy... %d\n\n", (30-mistakes));
    printf("\n\naccuracy %%... %.3f%%\n\n", 100.0*((30.0-(double)mistakes)/30.0));
    printf("loss... %.3f\n\n", -loss);

    // free allocated memory
    blast_matrix(learned_parameters.W2);
    blast_matrix(learned_parameters.b1);
    blast_matrix(learned_parameters.W1);
    blast_matrix(learned_parameters.b2);
    blast_matrix(iris_data_shuffled);
    blast_matrix(targets);
    blast_matrix(features);

    return 0;
}