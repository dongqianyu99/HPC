#include <cstdlib>
#include <iostream>
#include <mpi.h>

#include "judger.h"

extern "C" int bicgstab(int N, double* A, double* b, double* x, int max_iter, double tol, MPI_Comm comm);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // When using MPI, please remember to initialize here

    if (argc != 2) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_data>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
    // Read data from file
    std::string filename = argv[1];

    // N: size of matrix A (N x N)
    // A: matrix A
    // b: vector b
    // x: initial guess of solution
    int N;
    double *A = nullptr, *b = nullptr, *x = nullptr;

    // Read data from file
    if (world_rank == 0) {
        read_data(filename, &N, &A, &b, &x);
    }

    // 对各个进程广播 N 的值
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 只有主进程读取数据，其他进程的数据由主进程分配
    if (world_rank != 0) {
        A = (double*)malloc(N * N * sizeof(double));
        b = (double*)malloc(N * sizeof(double));
        x = (double*)malloc(N * sizeof(double));
    }

    // 向各个进程广播 A，b，x 的值
    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Call BiCGSTAB function
    auto start = std::chrono::high_resolution_clock::now();
    int iter   = bicgstab(N, A, b, x, MAX_ITER, TOL, MPI_COMM_WORLD);
    auto end   = std::chrono::high_resolution_clock::now();

    // Check the result
    if (world_rank == 0) {
        auto duration = end - start;
        judge(iter, duration, N, A, b, x);
    }

    // Free allocated memory
    free(A);
    free(b);
    free(x);

    // When using MPI, please remember to finalize here
    MPI_Finalize();
    return 0;
}
