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
int64_t* image;
int width;
int height;
int iters;
double left;
double right;
double lower;
double upper;
double y_scale, x_scale;
int chunk_size; // New parameter for column chunk size

// Constants for SIMD operations
__m512d v_2;
__m512d v_4;
__m512i v_1;
const __m512d v_increment = _mm512_set1_pd(8.0);

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

void task_queue_destroy(task_queue_t* queue) {
    pthread_mutex_destroy(&queue->mutex);
    pthread_cond_destroy(&queue->cond);
}

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
    free(node);
    pthread_mutex_unlock(&queue->mutex);
    return 1;
}

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
                __m512i v_repeat = _mm512_setzero_si512();

                for(int k=0;k<iters;k++) {
                    __mmask8 cmp_mask = _mm512_cmp_pd_mask(v_length_squared, v_4, _CMP_LT_OS);
                    if (cmp_mask == 0)
                        break;

                    v_repeat = _mm512_mask_add_epi64(v_repeat, cmp_mask, v_repeat, v_1);

                    __m512d v_x_sq = _mm512_mul_pd(v_x, v_x);
                    __m512d v_y_sq = _mm512_mul_pd(v_y, v_y);
                    __m512d temp = _mm512_add_pd(_mm512_sub_pd(v_x_sq, v_y_sq), v_x0);
                    v_y = _mm512_fmadd_pd(_mm512_mul_pd(v_x, v_y), v_2, v_y0);
                    v_x = temp;
                    v_length_squared = _mm512_fmadd_pd(v_x, v_x, _mm512_mul_pd(v_y, v_y));
                }
                _mm512_storeu_si512((__m512i*)(image + row * width + i), v_repeat);
            } else {
                for (int k = i; k < col_end; ++k) {
                    double x0 = k * x_scale + left;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[row * width + k] = repeats;
                }
            }
        }
    }
    pthread_exit(NULL);
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
    chunk_size = 1800;

    /* allocate memory for image */
    image = (int64_t*)malloc(width * height * sizeof(int64_t));

    /* Initialize constants for SIMD */
    v_2 = _mm512_set1_pd(2.0);
    v_4 = _mm512_set1_pd(4.0);
    v_1 = _mm512_set1_epi64(1);

    y_scale = ((upper - lower) / height);
    x_scale = ((right - left) / width);

    /* Initialize task queue */
    task_queue_t task_queue;
    task_queue_init(&task_queue);

    /* Create worker threads */
    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_function, &task_queue);
    }

    /* Fill task queue with tasks (chunks of columns per row) */
    
    for (int col = 0; col < width; col += chunk_size) {
        for (int row = 0; row < height; row++) {
            task_t task;
            task.row = row;
            task.col_start = col;
            task.col_end = (col + chunk_size < width) ? (col + chunk_size) : width;
            task_queue_push(&task_queue, task);
        }
    }

    /* Push exit tasks for each worker */
    for (int i = 0; i < num_threads; i++) {
        task_t exit_task;
        exit_task.row = -1;
        exit_task.col_start = 0;
        exit_task.col_end = 0;
        task_queue_push(&task_queue, exit_task);
    }
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    /* Wait for all threads to finish */
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Destroy task queue */
    // task_queue_destroy(&task_queue);

    /* Write the image to a PNG file */
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int64_t p = image[(height - 1 - y) * width + x];
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

    png_write_end(png_ptr, NULL);
    return 0;
}
