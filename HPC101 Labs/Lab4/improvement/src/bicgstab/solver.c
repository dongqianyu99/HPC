#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <mpi.h>

void gemv(double* y, double* A, double* x, int N, int size, int rank, int start, int end) {
    // int chunk_size = N / size;
    // int start = rank * chunk_size;
    // int end = (rank == size - 1) ? N : start + chunk_size;
    
    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        __m512d y_reg = _mm512_setzero_pd();

        int j;
        for (j = 0; j <= N - 8; j += 8) {
            __m512d A_reg = _mm512_loadu_pd(A + i * N + j);  // 内存未对齐
            __m512d x_reg = _mm512_loadu_pd(x + j);          
            __m512d mul_reg = _mm512_mul_pd(A_reg, x_reg);
            y_reg = _mm512_add_pd(y_reg, mul_reg);              
        }

        y[i] = _mm512_reduce_add_pd(y_reg);

        // 剩余边界
        for (; j < N; j++) {
            y[i] += A[i * N + j] * x[j];
        }
    }
}

// void gemv_mpi(double* y, double* A, double* x, int N, int rank, int size) {
//     // int chunk_size = N / size;
//     // int start = rank * chunk_size;
//     // int end = (rank == size - 1) ? N : start + chunk_size;

//     // 尽量均匀分配相关任务
//     int chunk = N / size;
//     int remainder = N % size;
//     int start, end;

//     if (rank < remainder) {
//         start = rank * (chunk + 1);
//         end = start + chunk;
//     } else {
//         start = rank * chunk + remainder;
//         end = start + chunk - 1;
//     }

//     for (int i = start; i < end; i++) 
//     {
//         y[i] = 0.0;
//         int j;
//         __m256d y_vec = _mm256_setzero_pd();
//         for (j = 0; j <= N - 4; j += 4) {
//             __m256d x_vec = _mm256_loadu_pd(&x[j]);
//             __m256d A_vec = _mm256_loadu_pd(&A[i * N + j]);
//             y_vec = _mm256_fmadd_pd(x_vec, A_vec, y_vec);
//         }
//         double temp[4];
//         _mm256_storeu_pd(temp, y_vec);
//         y[i] = temp[0] + temp[1] + temp[2] + temp[3];
//         for (; j < N; j++)
//         y[i] += x[j] * A[i * N + j];
//     }
// }

// double dot_product(double* x, double* y, int N) {
//     // dot product of x and y
//     double result = 0.0;
//     for (int i = 0; i < N; i++) {
//         result += x[i] * y[i];
//     }
//     return result;
// }

double dot_product(double* x, double* y, int N) {
    // dot product of x and y
    int i = 0;
    double result = 0.0;
    __m512d res_reg = _mm512_setzero_pd();
    for (; i <= N - 8; i+= 8) {
        __m512d x_reg = _mm512_loadu_pd(x + i);
        __m512d y_reg = _mm512_loadu_pd(y + i);
        __m512d mul_reg = _mm512_mul_pd(x_reg, y_reg);
        res_reg = _mm512_add_pd(mul_reg, res_reg);
    }

    result = _mm512_reduce_add_pd(res_reg);

    for (; i < N; i++) {
        result += x[i] * y[i];
    }

    return result;
}


// void precondition(double* A, double* K2_inv, int N) {
//     // K2_inv = 1 / diag(A)
//     for (int i = 0; i < N; i++) {
//         K2_inv[i] = 1.0 / A[i * N + i];
//     }
// }


void precondition(double* A, double* K2_inv, int N) {
    int i = 0;
    for (; i <= N - 8; i += 8) {
        // SIMD 寄存器从高位到地位存储 Index 7 -> Index 0
        __m512d diag = _mm512_set_pd(
            A[(i+7) * N + (i+7)],
            A[(i+6) * N + (i+6)],
            A[(i+5) * N + (i+5)],
            A[(i+4) * N + (i+4)],
            A[(i+3) * N + (i+3)],
            A[(i+2) * N + (i+2)],
            A[(i+1) * N + (i+1)],
            A[i * N + i]
        );
        
        // Perform 1.0 / diag for each element
        __m512d inv_diag = _mm512_div_pd(_mm512_set1_pd(1.0), diag);
        
        // Store the result back to K2_inv
        _mm512_storeu_pd(K2_inv + i, inv_diag);
    }
    
    // 处理剩余的元素
    for (; i < N; i++) {
        K2_inv[i] = 1.0 / A[i * N + i];
    }
}



// void precondition_apply(double* z, double* K2_inv, double* r, int N) {
//     // z = K2_inv * r
//     for (int i = 0; i < N; i++) {
//         z[i] = K2_inv[i] * r[i];
//     }
// }

void precondition_apply(double* z, double* K2_inv, double* r, int N) {
    int i = 0;
    for (; i <= N - 8; i += 8) {
        __m512d k2_inv_reg = _mm512_loadu_pd(K2_inv + i);
        __m512d r_reg = _mm512_loadu_pd(r + i);
        __m512d z_reg = _mm512_mul_pd(k2_inv_reg, r_reg);

        // 将结果存储到 z 数组中
        _mm512_storeu_pd(z + i, z_reg);
    }

    // 处理剩余未能被 8 整除的元素
    for (; i < N; i++) {
        z[i] = K2_inv[i] * r[i];
    }
}



int bicgstab(int N, double* A, double* b, double* x, int max_iter, double tol, MPI_Comm comm) {
    int rank, size;
    // 获取当前进程的相关信息
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 计算每个进程处理的数据量
    int *recvcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));

    int base_size = N / size;
    // 初始化 recvcounts 和 displs
    int offset = 0;
    int i = 0;
    for (; i < size - 1; i++) {
        recvcounts[i] = base_size;
        displs[i] = offset;
        offset += recvcounts[i];               
    }
    recvcounts[i] = N - base_size * (size - 1);
    displs[i] = offset; 

    int start = displs[rank];
    int end = start + recvcounts[rank];

    /**
     * Algorithm: BICGSTAB
     *  r: residual
     *  r_hat: modified residual
     *  p: search direction
     *  K2_inv: preconditioner (We only store the diagonal of K2_inv)
     * Reference: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
     */
    double* r      = (double*)calloc(N, sizeof(double));
    double* r_hat  = (double*)calloc(N, sizeof(double));
    double* p      = (double*)calloc(N, sizeof(double));
    double* v      = (double*)calloc(N, sizeof(double));
    double* s      = (double*)calloc(N, sizeof(double));
    double* h      = (double*)calloc(N, sizeof(double));
    double* t      = (double*)calloc(N, sizeof(double));
    double* y      = (double*)calloc(N, sizeof(double));
    double* z      = (double*)calloc(N, sizeof(double));
    double* K2_inv = (double*)calloc(N, sizeof(double));

    double rho_old = 1, alpha = 1, omega = 1;
    double rho = 1, beta = 1;
    double tol_squared = tol * tol;

    // Take M_inv as the preconditioner
    // Note that we only use K2_inv (in wikipedia)
    precondition(A, K2_inv, N);

    // 1. r0 = b - A * x0
    gemv(r, A, x, N, rank, size, start, end);
    MPI_Barrier(comm); 
    if(rank == 0)
        MPI_Gatherv(MPI_IN_PLACE, 0 , MPI_DOUBLE, r, recvcounts, displs, MPI_DOUBLE, 0, comm);
    else
        MPI_Gatherv(r + start, end - start , MPI_DOUBLE, NULL , recvcounts, displs, MPI_DOUBLE, 0, comm);
    MPI_Barrier(comm);
    MPI_Bcast(r , N , MPI_DOUBLE , 0 ,comm);

    for (int i = 0; i < N; i++) {
        r[i] = b[i] - r[i];
    }

    // 2. Choose an arbitary vector r_hat that is not orthogonal to r
    // We just take r_hat = r, please do not change this initial value
    memmove(r_hat, r, N * sizeof(double));  // memmove is safer memcpy :)

    // 3. rho_0 = (r_hat, r)
    rho = dot_product(r_hat, r, N);

    // 4. p_0 = r_0
    memmove(p, r, N * sizeof(double));

    int iter;
    for (iter = 1; iter <= max_iter; iter++) {
        if (iter % 1000 == 0) {
            printf("Iteration %d, residul = %e\n", iter, sqrt(dot_product(r, r, N)));
        }

        // 1. y = K2_inv * p (apply preconditioner)
        precondition_apply(y, K2_inv, p, N);

        // 2. v = Ay
        // gemv(v, A, y, N);
        gemv(v, A, x, N, rank, size, start, end);
        MPI_Barrier(comm); 
        if(rank == 0)
            MPI_Gatherv(MPI_IN_PLACE, 0 , MPI_DOUBLE, v, recvcounts, displs, MPI_DOUBLE, 0, comm);
        else
            MPI_Gatherv(v + start, end - start , MPI_DOUBLE, NULL , recvcounts, displs, MPI_DOUBLE, 0, comm);
        MPI_Barrier(comm);
        MPI_Bcast(v , N , MPI_DOUBLE , 0 ,comm);

        // 3. alpha = rho / (r_hat, v)
        alpha = rho / dot_product(r_hat, v, N);

        // 4. h = x_{i-1} + alpha * y
        for (int i = 0; i < N; i++) {
            h[i] = x[i] + alpha * y[i];
        }

        // 5. s = r_{i-1} - alpha * v
        for (int i = 0; i < N; i++) {
            s[i] = r[i] - alpha * v[i];
        }

        // 6. Is h is accurate enough, then x_i = h and quit
        if (dot_product(s, s, N) < tol_squared) {
            memmove(x, h, N * sizeof(double));
            break;
        }

        // 7. z = K2_inv * s
        precondition_apply(z, K2_inv, s, N);

        // 8. t = Az
        // gemv(t, A, z, N);
        gemv(t, A, x, N, rank, size, start, end);
        MPI_Barrier(comm); 
        if(rank == 0)
            MPI_Gatherv(MPI_IN_PLACE, 0 , MPI_DOUBLE, t, recvcounts, displs, MPI_DOUBLE, 0, comm);
        else
            MPI_Gatherv(t + start, end - start , MPI_DOUBLE, NULL , recvcounts, displs, MPI_DOUBLE, 0, comm);
        MPI_Barrier(comm);
        MPI_Bcast(t , N , MPI_DOUBLE , 0 ,comm);

        // 9. omega = (t, s) / (t, t)
        omega = dot_product(t, s, N) / dot_product(t, t, N);

        // 10. x_i = h + omega * z
        for (int i = 0; i < N; i++) {
            x[i] = h[i] + omega * z[i];
        }

        // 11. r_i = s - omega * t
        for (int i = 0; i < N; i++) {
            r[i] = s[i] - omega * t[i];
        }

        // 12. If x_i is accurate enough, then quit
        if (dot_product(r, r, N) < tol_squared) break;

        rho_old = rho;
        // 13. rho_i = (r_hat, r)
        rho = dot_product(r_hat, r, N);

        // 14. beta = (rho_i / rho_{i-1}) * (alpha / omega)
        beta = (rho / rho_old) * (alpha / omega);

        // 15. p_i = r_i + beta * (p_{i-1} - omega * v)
        for (int i = 0; i < N; i++) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
    }

    free(r);
    free(r_hat);
    free(p);
    free(v);
    free(s);
    free(h);
    free(t);
    free(y);
    free(z);
    free(K2_inv);
    free(recvcounts);
    free(displs);

    if (iter >= max_iter)
        return -1;
    else
        return iter;
}
