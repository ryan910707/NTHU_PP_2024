#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <png.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <stdlib.h>
#include <sched.h>

int num_threads;
int* image;
int width;
int height;
int iters;
double left;
double right;
double lower;
double upper;
double y_scale, x_scale;
int chunk_size; // Parameter for column chunk size

// Constants for SIMD operations
__m512d v_2;
__m512d v_4;
__m256i v_1;

// Synchronization variables
int* row_ready;
int* row_chunk_counts;
pthread_mutex_t* row_mutexes;
pthread_cond_t* row_conds;

// Task structure
typedef struct {
    int row;
    int col_start;
    int col_end;
} task_t;

// Task queue implementation
typedef struct task_node {
    task_t task;
    struct task_node* next;
} task_node_t;

typedef struct {
    task_node_t* front;
    task_node_t* rear;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} task_queue_t;

void task_queue_init(task_queue_t* queue) {
    queue->front = queue->rear = NULL;
    pthread_mutex_init(&queue->mutex, NULL);
    pthread_cond_init(&queue->cond, NULL);
}

void task_queue_push(task_queue_t* queue, task_t task) {
    task_node_t* node = (task_node_t*)malloc(sizeof(task_node_t));
    node->task = task;
    node->next = NULL;

    pthread_mutex_lock(&queue->mutex);
    if (queue->rear == NULL) {
        queue->front = queue->rear = node;
    } else {
        queue->rear->next = node;
        queue->rear = node;
    }
    pthread_cond_signal(&queue->cond);
    pthread_mutex_unlock(&queue->mutex);
}

// void task_queue_destroy(task_queue_t* queue) {
//     pthread_mutex_destroy(&queue->mutex);
//     pthread_cond_destroy(&queue->cond);
// }

int task_queue_pop(task_queue_t* queue, task_t* task) {
    pthread_mutex_lock(&queue->mutex);
    while (queue->front == NULL) {
        pthread_cond_wait(&queue->cond, &queue->mutex);
    }
    task_node_t* node = queue->front;
    *task = node->task;
    queue->front = node->next;
    if (queue->front == NULL) {
        queue->rear = NULL;
    }
    pthread_mutex_unlock(&queue->mutex);
    return 1;
}

__mmask8 mask = 0xFF;
void* worker_function(void* arg) {
    task_queue_t* queue = (task_queue_t*)arg;
    task_t task;
    while (1) {
        task_queue_pop(queue, &task);
        if (task.row == -1) {
            // Exit task received
            break;
        }
        int row = task.row;
        int col_start = task.col_start;
        int col_end = task.col_end;
        int offset = row*width;

        double y0 = row * y_scale + lower;
        __m512d v_y0 = _mm512_set1_pd(y0);

        for (int i = col_start; i < col_end; i += 8) {
            int remaining = col_end - i;
            if (remaining >= 8) {
                __m512d v_i = _mm512_set_pd(i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i + 0);
                __m512d v_x0 = _mm512_fmadd_pd(v_i, _mm512_set1_pd(x_scale), _mm512_set1_pd(left));

                __m512d v_x = _mm512_setzero_pd();
                __m512d v_y = _mm512_setzero_pd();
                __m512d v_length_squared = _mm512_setzero_pd();
                __m256i v_repeat = _mm256_setzero_si256();
                __m512d v_x_sq = _mm512_setzero_pd();
                __m512d v_y_sq = _mm512_setzero_pd();
                for(int k=0;k<iters;k++) {
                    __mmask8 cmp_mask = _mm512_cmp_pd_mask(v_length_squared, v_4, _CMP_LT_OS);
                    if (cmp_mask == 0)
                        break;

                    v_repeat = _mm256_mask_add_epi32(v_repeat, cmp_mask, v_repeat, v_1);

                    __m512d temp = _mm512_add_pd(_mm512_sub_pd(v_x_sq, v_y_sq), v_x0);
                    v_y = _mm512_fmadd_pd(_mm512_mul_pd(v_x, v_y), v_2, v_y0);
                    v_x = temp;
                    v_x_sq = _mm512_mul_pd(v_x, v_x);
                    v_y_sq = _mm512_mul_pd(v_y, v_y);
                    v_length_squared = _mm512_fmadd_pd(v_x, v_x, v_y_sq);
                }
                _mm256_mask_storeu_epi32((__m256i*)(image + offset + i), mask, v_repeat);
            } else {
                for (int k = i; k < col_end; ++k) {
                    double x0 = k * x_scale + left;
                    double x = 0;
                    double y = 0;
                    double x_2 = 0;
                    double y_2 = 0;
                    double length_squared = 0;
                    int repeats = 0;
                    for(int k =0;k<iters;k++) {
                        if(length_squared>4)
                            break;
                        double temp = x_2 - y_2 + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        x_2 = x*x;
                        y_2 = y*y;
                        length_squared = x_2 + y_2;
                        repeats++;
                    }
                    image[offset + k] = repeats;
                }
            }
        }

        // Decrement the row chunk count
        pthread_mutex_lock(&row_mutexes[row]);
        row_chunk_counts[row]--;
        if (row_chunk_counts[row] == 0) {
            // All chunks for this row are done
            pthread_cond_signal(&row_conds[row]);
        }
        pthread_mutex_unlock(&row_mutexes[row]);
    }
    // pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);

    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    chunk_size = 2000;

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));

    /* Initialize constants for SIMD */
    v_2 = _mm512_set1_pd(2.0);
    v_4 = _mm512_set1_pd(4.0);
    v_1 = _mm256_set1_epi32(1);

    y_scale = ((upper - lower) / height);
    x_scale = ((right - left) / width);

    /* Initialize task queue */
    task_queue_t task_queue;
    task_queue_init(&task_queue);

    /* Initialize synchronization variables */
    row_ready = (int*)malloc(height*sizeof(int));
    row_chunk_counts = (int*)malloc(height * sizeof(int));
    row_mutexes = (pthread_mutex_t*)malloc(height * sizeof(pthread_mutex_t));
    row_conds = (pthread_cond_t*)malloc(height * sizeof(pthread_cond_t));

    /* Calculate number of chunks per row */
    int chunks_per_row = (width + chunk_size - 1) / chunk_size;
    for (int i = 0; i < height; i++) {
        row_chunk_counts[i] = chunks_per_row;
        pthread_mutex_init(&row_mutexes[i], NULL);
        pthread_cond_init(&row_conds[i], NULL);
    }

    /* Create worker threads */
    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_function, &task_queue);
    }

    /* Fill task queue with tasks (chunks of columns per row) */
    for (int row = height-1; row >=0; row--) {
        for (int col = 0; col < width; col += chunk_size) {
            task_t task;
            task.row = row;
            task.col_start = col;
            task.col_end = (col + chunk_size < width) ? (col + chunk_size) : width;
            task_queue_push(&task_queue, task);
        }
    }

    /* Initialize PNG writing */
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);


    /* Push exit tasks for each worker */
    for (int i = 0; i < num_threads; i++) {
        task_t exit_task;
        exit_task.row = -1;
        exit_task.col_start = 0;
        exit_task.col_end = 0;
        task_queue_push(&task_queue, exit_task);
    }
    for (int y = height-1; y >=0; y--) {
        pthread_mutex_lock(&row_mutexes[y]);
        while (row_chunk_counts[y] != 0) {
            pthread_cond_wait(&row_conds[y], &row_mutexes[y]);
        }
        pthread_mutex_unlock(&row_mutexes[y]);

        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = image[y * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                int p_shift = (p & 0xF)<<4;
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p_shift;
                } else {
                    color[0] = p_shift;
                }
            }
        }
        png_write_row(png_ptr, row);
    }

    /* Wait for all threads to finish */
    // for (int i = 0; i < num_threads; i++) {
    //     pthread_join(threads[i], NULL);
    // }

    /* Destroy task queue */
    // task_queue_destroy(&task_queue);

    /* Clean up */
    png_write_end(png_ptr, NULL);
    // png_destroy_write_struct(&png_ptr, &info_ptr);
    // fclose(fp);
    // free(row);
    // free(image);
    // free(row_ready);
    // free(row_chunk_counts);

    // for (int i = 0; i < height; i++) {
    //     pthread_mutex_destroy(&row_mutexes[i]);
    //     pthread_cond_destroy(&row_conds[i]);
    // }
    // free(row_mutexes);
    // free(row_conds);

}
