#ifndef ELDRITCH_ARRAYS_H
#define ELDRITCH_ARRAYS_H


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>


typedef struct matrix_s {
    unsigned int n_rows;
    unsigned int n_cols;
    double **data;
} matrix;


matrix *create_matrix(unsigned int n_rows, unsigned int n_cols);
void free_matrix(matrix *mat);
matrix *copy_matrix(matrix *mat);
double get_random_number_from_range(double min, double max);
matrix *create_random_matrix(unsigned int n_rows, unsigned int n_cols, double min, double max);
matrix *read_matrix_from_file(const char *file_path);
matrix *_read_matrix_from_file(FILE *file);
matrix *create_ones_matrix(unsigned int n_rows, unsigned int n_cols);
matrix *create_identity_matrix(unsigned int n_rows, unsigned int n_cols);
void print_matrix_formatted(matrix* mat, const char *format);
void print_matrix(matrix *mat);
matrix *transpose(matrix *mat);
matrix *get_col(matrix *original_mat, unsigned int col);
matrix *get_row(matrix *original_mat, unsigned int row);
matrix *multiply_by_scaler(matrix* mat, double scaler);
void _multiply_by_scaler(matrix* mat, double scaler);
bool do_matrices_have_same_dimensions(matrix *A, matrix *B);
matrix *add_matrices(matrix* A, matrix *B);
matrix *subtract_matrices(matrix* A, matrix *B);
bool _add_matrices(matrix* A, matrix *B);
matrix *multiply_matrices(matrix *A, matrix *B);
matrix *hadamard_product(matrix *A, matrix* B);
bool _swipe_columns(matrix *mat, unsigned int col1, unsigned int col2);
matrix *swipe_columns(matrix *mat, unsigned int col1, unsigned int col2);
bool _swipe_rows(matrix *mat, unsigned int row1, unsigned int row2);
matrix *swipe_rows(matrix *mat, unsigned int row1, unsigned int row2);


matrix *create_matrix(unsigned int n_rows, unsigned int n_cols) {
    if (n_rows <= 0) { printf("[!] can not have negative number of rows\n"); return NULL; }
    if (n_cols <= 0) { printf("[!] can not have negative number of columns\n"); return NULL; }
    matrix *mat = (matrix*) calloc(1, sizeof(*mat));
    if (mat == NULL) { printf("[!] error during matrix STRUCT memory allocation\n"); return NULL; }
    mat->n_rows = n_rows;
    mat->n_cols = n_cols;
    mat->data = (double**) calloc(mat->n_rows, sizeof(*mat->data));
    if (mat->data == NULL) { printf("[!] error during matrix ROWS DATA memory allocation\n"); return NULL; }
    for (int i = 0; i < mat->n_rows; i++) {
        mat->data[i] = (double*) calloc(mat->n_cols, sizeof(**mat->data));
        if (mat->data[i] == NULL) { printf("[!] error during matrix COLUMNS DATA memory allocation\n"); return NULL; }
    }
    return mat;
}

void free_matrix(matrix *mat) {
    for (int i = 0; i < mat->n_rows; i++) { free(mat->data[i]); }
    free(mat->data);
    free(mat);
}

matrix *copy_matrix(matrix *mat) {
    matrix *copy = create_matrix(mat->n_rows, mat->n_cols);
    for (int row = 0; row < copy->n_rows; row++) {
        for (int col = 0; col < copy->n_cols; col++) {
            copy->data[row][col] = mat->data[row][col];
        }
    }
    return copy;
}

double get_random_number_from_range(double min, double max) {
    double number;
    // value between 0 and 1
    number = (double) rand() / ((double) RAND_MAX + 1); 
    // value within range
    return (min + number * (max - min));
}

matrix *create_random_matrix(unsigned int n_rows, unsigned int n_cols, double min, double max) {
    matrix *mat = create_matrix(n_rows, n_cols);
    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < n_cols; col++) {
            mat->data[row][col] = get_random_number_from_range(min, max);
        }
    }
    return mat;
}

matrix *read_matrix_from_file(const char *file_path) {
    FILE *mat_file = fopen(file_path, "r");
    if (mat_file == NULL) { printf("[!] could not open matrix data file\n"); return NULL; }
    matrix *mat = _read_matrix_from_file(mat_file);
    fclose(mat_file);
    return mat;
}

matrix *_read_matrix_from_file(FILE *file) {
    // the file must be structured like this:
    // - first line with number of rows and number of columns
    // - matrix with rows on each line and columns separeted by tabs
    int n_rows=0, n_cols=0;
    fscanf(file, "%d %d", &n_rows, &n_cols);
    matrix *mat = create_matrix(n_rows, n_cols);
    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < n_cols; col++) {
            fscanf(file, "%lf\t", &mat->data[row][col]);
        }
    }
    return mat;
}

matrix *create_ones_matrix(unsigned int n_rows, unsigned int n_cols) {
    matrix *result = create_matrix(n_rows, n_cols);
    for (int row = 0; row < result->n_rows; row++) {
        for (int col = 0; col < result->n_cols; col++) {
            result->data[row][col] = 1.0;
        }
    }
    return result;
}

matrix *create_identity_matrix(unsigned int n_rows, unsigned int n_cols) {
    matrix *mat = create_matrix(n_rows, n_cols);
    for (int row = 0; row < mat->n_rows; row++) {
        for (int col = 0; col < mat->n_cols; col++) {
            mat->data[row][col] = (row == col) ? 1.0 : 0.0;
        }
    }
    return mat;
}

void print_matrix_formatted(matrix* mat, const char *format) {
    fprintf(stdout, "\n");
    for (int row = 0; row < mat->n_rows; row++) {
        for (int col = 0; col < mat->n_cols; col++) {
            fprintf(stdout, format, mat->data[row][col]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}

void print_matrix(matrix *mat) {
    print_matrix_formatted(mat, "%lf\t\t");
}

matrix *transpose(matrix *mat) {
    matrix *mat_t = create_matrix(mat->n_cols, mat->n_rows);
    for (int row = 0; row < mat->n_rows; row++) {
        for (int col = 0; col < mat->n_cols; col++) { 
            mat_t->data[col][row] = mat->data[row][col];
        }
    } 
    return mat_t;
}

matrix *get_col(matrix *original_mat, unsigned int target_col) {
    if (target_col >= original_mat->n_cols) { printf("[!] matrix does not have this many columns\n"); return NULL; }
    matrix *mat = create_matrix(original_mat->n_rows, 1);
    for (int row = 0; row < mat->n_rows; row++) {
        mat->data[row][0] = original_mat->data[row][target_col];
    }
    return mat;
}

matrix *get_row(matrix *original_mat, unsigned int target_row) {
    if (target_row >= original_mat->n_rows) { printf("[!] matrix does not have this many rows\n"); return NULL; }
    matrix *mat = create_matrix(1, original_mat->n_cols);
    // memory per row is contiguous, no loops needed
    memcpy(mat->data[0], original_mat->data[target_row], original_mat->n_cols * sizeof(*mat->data[0]));
    return mat;
}

matrix *multiply_by_scaler(matrix* mat, double scaler) {
    matrix *scaled_mat = copy_matrix(mat);
    _multiply_by_scaler(scaled_mat, scaler);
    return scaled_mat;
}

void _multiply_by_scaler(matrix* mat, double scaler) {
    for (int row = 0; row < mat->n_rows; row++) {
        for (int col = 0; col < mat->n_cols; col++) {
            mat->data[row][col] *= scaler;
        }
    }
}

bool do_matrices_have_same_dimensions(matrix *A, matrix *B) {
    bool dimension_check = false;
    if (A->n_rows == B->n_rows && A->n_cols == B->n_cols) { dimension_check = true; }
    return dimension_check;
}

matrix *add_matrices(matrix* A, matrix *B) {
    matrix *result = copy_matrix(A);
    if (!_add_matrices(result, B)) { free_matrix(result); return NULL; }
    return result;
}

matrix *subtract_matrices(matrix* A, matrix *B) {
    matrix *result = copy_matrix(A);
    B = multiply_by_scaler(B, -1.0);
    if (!_add_matrices(result, B)) { free_matrix(result); return NULL; }
    return result;
}

bool _add_matrices(matrix* A, matrix *B) {
    if (!do_matrices_have_same_dimensions(A, B)) {
        printf("[!] could not add matrices, diferent dimensions\n");
        return false;
    }
    for (int row = 0; row < A->n_rows; row++) {
        for (int col = 0; col < B->n_cols; col++) {
            A->data[row][col] += B->data[row][col];
        }
    }
    return true;
}

matrix *multiply_matrices(matrix *A, matrix *B) {
    if ( A->n_cols != B->n_rows ) { printf("[!] could not multiply matrices\n"); return NULL; }
    matrix *result = create_matrix(A->n_rows, B->n_cols);
    for (int row = 0; row < A->n_rows; row++) {
        for (int col = 0; col < B->n_cols; col++) { 
            for (int sum_iter = 0; sum_iter < A->n_cols; sum_iter++) {
                result->data[row][col] += A->data[row][sum_iter] * B->data[sum_iter][col];
            }
        }
    }
    return result;
}

matrix *hadamard_product(matrix *A, matrix* B) {
    if ((A->n_rows != B->n_rows) || (A->n_cols != B->n_cols)) { printf("[!] could not apply hadamard product\n"); return NULL; }
    matrix *result = create_matrix(A->n_rows, B->n_cols);
    for (int row = 0; row < result->n_rows; row++) {
        for (int col = 0; col < result->n_cols; col++) { 
            result->data[row][col] = A->data[row][col] * B->data[row][col];
        }
    }
    return result;
}

bool _swipe_columns(matrix *mat, unsigned int col1, unsigned int col2) {
    if (col1 >= mat->n_cols || col2 >= mat->n_cols) {
        printf("[!] invalid column index\n");
        return false;
    }
    double temp;
    for (size_t row = 0; row < mat->n_rows; row++) {
        temp = mat->data[row][col1];
        mat->data[row][col1] = mat->data[row][col2];
        mat->data[row][col2] = temp;
    }
    return true;
}

matrix *swipe_columns(matrix *mat, unsigned int col1, unsigned int col2) {
    matrix *result = copy_matrix(mat);
    if (!_swipe_columns(result, col1, col2)) {
        free_matrix(result);
        return NULL;
    }
    return result;
}

bool _swipe_rows(matrix *mat, unsigned int row1, unsigned int row2) {
    if (row1 >= mat->n_rows || row2 >= mat->n_rows) {
        printf("[!] invalid row index\n");
        return false;
    }
    double *temp = mat->data[row2];
    mat->data[row2] = mat->data[row1];
    mat->data[row1] = temp;
    return true;
}

matrix *swipe_rows(matrix *mat, unsigned int row1, unsigned int row2) {
    matrix *result = copy_matrix(mat);
    if (!_swipe_rows(result, row1, row2)) {
        free_matrix(result);
        return NULL;
    }
    return result;
}



#endif