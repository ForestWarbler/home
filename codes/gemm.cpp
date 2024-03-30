
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <random>
#include <omp.h>
#include <immintrin.h>
#include <mpi.h>

using namespace std;
typedef chrono::high_resolution_clock Clock;

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, double**& C,
    string outfile)
{
    int i, j;
    ofstream fout;
    fout.open(outfile);
    if (!fout.is_open())
    {
        cout << "Error opening file for result" << endl;
        exit(1);
    }

    fout << setprecision(10);
    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            fout << C[i][j] << " ";
        }
        fout << endl;
    }
}

/* Read the case file */
static void read_case(string casefile, int& ni, int& nk, int& nj,
    double& alpha, double& beta,
    double**& A, double**& B, double**& C)
{
    ifstream fin;
    fin.open(casefile);
    if (!fin.is_open())
    {
        cout << "Error opening file for case" << endl;
        exit(1);
    }
    fin >> ni >> nk >> nj;
    fin >> alpha >> beta;
    A = (double**)malloc(ni * sizeof(double*));
    for (int i = 0; i < ni; i++)
    {
        A[i] = (double*)malloc(nk * sizeof(double));
    }
    B = (double**)malloc(nk * sizeof(double*));
    for (int i = 0; i < nk; i++)
    {
        B[i] = (double*)malloc(nj * sizeof(double));
    }
    C = (double**)malloc(ni * sizeof(double*));
    for (int i = 0; i < ni; i++)
    {
        C[i] = (double*)malloc(nj * sizeof(double));
    }

    for (int i = 0; i < ni; i++)
    {
        for (int j = 0; j < nk; j++)
        {
            fin >> A[i][j];
        }
    }
    for (int i = 0; i < nk; i++)
    {
        for (int j = 0; j < nj; j++)
        {
            fin >> B[i][j];
        }
    }
    for (int i = 0; i < ni; i++)
    {
        for (int j = 0; j < nj; j++)
        {
            fin >> C[i][j];
        }
    }

    fin.close();
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gemm(int ni, int nk, int nj,
    double alpha,
    double beta,
    double**& C,
    double**& A,
    double**& B)
{
  int i, j, k;
  int step = 8;
  int rank, size;
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int tn = omp_get_max_threads();
  omp_set_num_threads(tn);

  double *C_tmp = new double[(ni / 2) * nj];
  double *C_flat = new double[(ni / 2) * nj];

  if (rank == 0) {
#pragma omp parallel for
    for (i = 0; i < ni / 2; i++) {
      for (j = 0; j < nj - step; j += step) {
        C[i][j] = C[i][j] * beta;
        C[i][j + 1] += C[i][j + 1] * beta;
        C[i][j + 2] += C[i][j + 2] * beta;
        C[i][j + 3] += C[i][j + 3] * beta;
        C[i][j + 4] += C[i][j + 4] * beta;
        C[i][j + 5] += C[i][j + 5] * beta;
        C[i][j + 6] += C[i][j + 6] * beta;
        C[i][j + 7] += C[i][j + 7] * beta;
      }
      for (; j < nj; j++) {
        C[i][j] = C[i][j] * beta;
      }
    }

    double sum = 0;
#pragma omp parallel for private(j, k)
    for (i = 0; i < (ni / 2); i++) {
      for (k = 0; k < nk; k++) {
        __m512d a_vec = _mm512_set1_pd(A[i][k] * alpha);
        for (j = 0; j <= nj - step; j += step) {
          __m512d b_vec = _mm512_loadu_pd(&B[k][j]);
          __m512d c_vec = _mm512_loadu_pd(&C[i][j]);
          __m512d result_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
          _mm512_storeu_pd(&C[i][j], result_vec);
        }
        for (; j < (nj / 2); j++) {
          C[i][j] += alpha * A[i][k] * B[k][j];
        }
      }
    }
  } else {
#pragma omp parallel for
    for (i = ni / 2; i < ni; i++) {
      for (j = 0; j < nj - step; j += step) {
        C[i][j] = C[i][j] * beta;
        C[i][j + 1] += C[i][j + 1] * beta;
        C[i][j + 2] += C[i][j + 2] * beta;
        C[i][j + 3] += C[i][j + 3] * beta;
        C[i][j + 4] += C[i][j + 4] * beta;
        C[i][j + 5] += C[i][j + 5] * beta;
        C[i][j + 6] += C[i][j + 6] * beta;
        C[i][j + 7] += C[i][j + 7] * beta;
      }
      for (; j < nj; j++) {
        C[i][j] = C[i][j] * beta;
      }
    }

    double sum = 0;
#pragma omp parallel for private(j, k)
    for (i = ni / 2; i < ni; i++) {
      for (k = 0; k < nk; k++) {
        __m512d a_vec = _mm512_set1_pd(A[i][k] * alpha);
        for (j = 0; j <= nj - step; j += step) {
          __m512d b_vec = _mm512_loadu_pd(&B[k][j]);
          __m512d c_vec = _mm512_loadu_pd(&C[i][j]);
          __m512d result_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
          _mm512_storeu_pd(&C[i][j], result_vec);
        }
        for (; j < (nj / 2); j++) {
          C[i][j] += alpha * A[i][k] * B[k][j];
        }
      }
    }
#pragma omp parallel for
    for (i = ni / 2; i < ni; i++) {
      for (j = 0; j < nj; j++) {
        *(C_tmp + (i - (ni / 2)) * nj + j) = C[i][j];
      }
    }

    MPI_Send(C_tmp, (ni / 2) * nj, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    MPI_Recv(C_flat, (ni / 2) * nj, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
#pragma omp parallel for
    for (i = ni / 2; i < ni; i++) {
      for (j = 0; j < nj; j++) {
        C[i][j] = *(C_flat + (i - (ni / 2)) * nj + j);
      }
    }
  }

  if (rank == 0) {
    delete[] C_flat;
  } else {
    delete[] C_tmp;
  }
}

int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int ni = 0;
    int nj = 0;
    int nk = 0;

    /* Variable declaration/allocation. */
    double alpha;
    double beta;

    /* Initialize array(s). */
    double** A = NULL;
    double** B = NULL;
    double** C = NULL;

    /* Read the option */
    int opt = 0;
    int opt_num = 0;
    string casefile = "case1.dat";
    string outfile = "result.dat";
    while ((opt = getopt(argc, argv, "hc:o:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            printf("Usage: ./mygemm -c <casefile> -o <outfile>\n");
            return 0;
        case 'c':
            casefile = optarg;
            opt_num += 1;
            break;
        case 'o':
            outfile = optarg;
            opt_num += 1;
            break;
        }
    }
    if (opt_num != 2)
    {
        printf("Usage: ./mygemm -c <casefile> -o <outfile>\nError: Check your input\n");
        return 0;
    }
    read_case(casefile, ni, nk, nj, alpha, beta, A, B, C);
    printf("INFO: ni = %d, nj = %d, nk = %d, alpha = %f, beta = %f casefile = %s\n", ni, nj, nk, alpha, beta, casefile.c_str());

    /***************************
     * Timing starts here
     * ************************/
    cout << "Start computing..." << endl;
    auto startTime = Clock::now();
    /* Run kernel. */
    MPI_Init(&argc, &argv);
    int step = 8;
    int rank, size;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    kernel_gemm(ni, nk, nj, alpha, beta, C, A, B);

    auto endTime = Clock::now();
    /***************************
     * main computation is here
     * ************************/
    auto compTime =
        chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "End computing..." << endl;
    cout << "Compute time=  " << compTime.count() << " microseconds" << endl;
    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    if (rank == 0) {
      print_array(ni, nj, C, outfile);
    }
    MPI_Finalize();
    /* Be clean. */
    for (int i = 0; i < ni; i++)
    {
        free(A[i]);
    }
    free(A);
    for (int i = 0; i < nk; i++)
    {
        free(B[i]);
    }
    free(B);
    for (int i = 0; i < ni; i++)
    {
        free(C[i]);
    }
    free(C);

    return 0;
}
