#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"
#include <x86intrin.h>
#include <omp.h>

#define NI 4096
#define NJ 4096
#define NK 4096

/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_and_valid_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  float golden_sum = 27789682688.000000;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

  if ( abs(sum-golden_sum)/golden_sum > 0.00001 ) // more than 0.001% error rate
    printf("Incorrect sum of C array. Expected sum: %f, your sum: %f\n", golden_sum, sum);
  else
    printf("Correct result. Sum of C array = %f\n", sum);
}


/* Main computational kernel: baseline. The whole function will be timed,
   including the call and return. DO NOT change the baseline.*/
static
void gemm_base(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
	C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}


static
void gemm_tile(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  const unsigned int TILE_SIZE = 16;
  for(unsigned int i = 0; i < NI; i += TILE_SIZE) // iterate through matrix in tiles
  {
    for(unsigned int j = 0; j < NJ; j+= TILE_SIZE)
    {
      for(unsigned int k = 0; k < NK; k += TILE_SIZE)
      {
        for(unsigned int ii = i; (ii < i + TILE_SIZE) && (ii < NI); ii++) // iterate though tiles
        {
          for(unsigned int jj = j; (jj < j + TILE_SIZE) && (jj < NJ); jj++)
          {
            if(k == 0)
            {
              C[ii*NJ+jj] *= beta;
            }

            for(unsigned int kk = k; (kk < k + TILE_SIZE) && (kk < NK); kk++)
            {
              C[ii*NJ+jj] += alpha * A[ii*NK+kk] * B[kk*NJ+jj];
            }
          }
        }
      }
    }
  }
}

/* Main computational kernel: with tiling and simd optimizations. */
// => Form C := alpha*A*B + beta*C,
    // A is NIxNK
    // B is NKxNJ
    // C is NIxNJ
static
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{

    const unsigned int TILE_DIM_1 = 32;
    const unsigned int TILE_DIM_2 = 512;
    const unsigned int TILE_DIM_3 = 16;

    for (unsigned int ii = 0; ii < NI; ii++) {
        for (unsigned int jj= 0; jj < NJ; jj+=8) {
            __m256 vecC = _mm256_loadu_ps(&C[ii * NJ + jj]);
            __m256 vecBeta = _mm256_set1_ps(beta);
            vecC = _mm256_mul_ps(vecC, vecBeta);
            _mm256_storeu_ps(&C[ii * NJ + jj], vecC);
        }
    }

    //loop tiling with TILE_DIM_1, TILE_DIM_2, TILE_DIM_3
    for (unsigned int i = 0; i < NI; i += TILE_DIM_1) 
    {
        for (unsigned int k = 0; k < NK; k += TILE_DIM_3) 
        {
            for (unsigned int j = 0; j < NJ; j += TILE_DIM_2) 
            { 
                for (unsigned int ii = i; ii < i + TILE_DIM_1 && ii < NI; ii++) 
                {
                    for (unsigned int kk= k; kk< k + TILE_DIM_3 && kk< NK; kk++) 
                    {
                        __m256 vecA = _mm256_broadcast_ss(&A[ii * NK + kk]);
                        for (unsigned int jj= j; jj< j + TILE_DIM_2 && jj< NJ; jj+= 8) 
                        { 
                            __m256 vecB = _mm256_loadu_ps(&B[kk* NJ + jj]);
                            __m256 vecC = _mm256_loadu_ps(&C[ii * NJ + jj]);
                            __m256 tempMult = _mm256_mul_ps(vecA, vecB);
                            vecC = _mm256_add_ps(vecC, _mm256_mul_ps(tempMult, _mm256_set1_ps(alpha)));
                            _mm256_storeu_ps(&C[ii * NJ + jj], vecC);
                        }
                    }
                }
            }
        }
    }
}

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
static
void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  omp_set_num_threads(20);
  const unsigned int TILE_DIM_1 = 32;
  const unsigned int TILE_DIM_2 = 512;
  const unsigned int TILE_DIM_3 = 16;

  for (unsigned int ii = 0; ii < NI; ii++) {
      for (unsigned int jj= 0; jj < NJ; jj+=8) {
          __m256 vecC = _mm256_loadu_ps(&C[ii * NJ + jj]);
          __m256 vecBeta = _mm256_set1_ps(beta);
          vecC = _mm256_mul_ps(vecC, vecBeta);
          _mm256_storeu_ps(&C[ii * NJ + jj], vecC);
      }
  }

  //loop tiling with TILE_DIM_1, TILE_DIM_2, TILE_DIM_3
  #pragma omp parallel for 
  for (unsigned int i = 0; i < NI; i += TILE_DIM_1) 
  {
      for (unsigned int k = 0; k < NK; k += TILE_DIM_3) 
      {
          for (unsigned int j = 0; j < NJ; j += TILE_DIM_2) 
          { 
              for (unsigned int ii = i; ii < i + TILE_DIM_1 && ii < NI; ii++) 
              {
                  for (unsigned int kk= k; kk< k + TILE_DIM_3 && kk< NK; kk++) 
                  {
                      __m256 vecA = _mm256_broadcast_ss(&A[ii * NK + kk]);
                      for (unsigned int jj= j; jj< j + TILE_DIM_2 && jj< NJ; jj+= 32) 
                      { 
                          __m256 vecB1 = _mm256_loadu_ps(&B[kk* NJ + jj]);
                          __m256 vecC1 = _mm256_loadu_ps(&C[ii * NJ + jj]);
                          __m256 vecB2 = _mm256_loadu_ps(&B[kk* NJ + jj + 8]);
                          __m256 vecC2 = _mm256_loadu_ps(&C[ii * NJ + jj + 8]);
                          
                          __m256 tempMult1 = _mm256_mul_ps(vecA, vecB1);
                          __m256 tempMult2 = _mm256_mul_ps(vecA, vecB2);
                          
                          vecC1 = _mm256_add_ps(vecC1, _mm256_mul_ps(tempMult1, _mm256_set1_ps(alpha)));
                          vecC2 = _mm256_add_ps(vecC2, _mm256_mul_ps(tempMult2, _mm256_set1_ps(alpha)));
                          
                          _mm256_storeu_ps(&C[ii * NJ + jj], vecC1);
                          _mm256_storeu_ps(&C[ii * NJ + jj + 8], vecC2);
                          
                          __m256 vecB3 = _mm256_loadu_ps(&B[kk* NJ + jj+ 16]);
                          __m256 vecC3 = _mm256_loadu_ps(&C[ii * NJ + jj+ 16]);
                          __m256 vecB4 = _mm256_loadu_ps(&B[kk* NJ + jj + 24]);
                          __m256 vecC4 = _mm256_loadu_ps(&C[ii * NJ + jj + 24]);

                          __m256 tempMult3 = _mm256_mul_ps(vecA, vecB3);
                          __m256 tempMult4 = _mm256_mul_ps(vecA, vecB4);

                          vecC3 = _mm256_add_ps(vecC3, _mm256_mul_ps(tempMult3, _mm256_set1_ps(alpha)));
                          vecC4 = _mm256_add_ps(vecC4, _mm256_mul_ps(tempMult4, _mm256_set1_ps(alpha)));
                          
                          _mm256_storeu_ps(&C[ii * NJ + jj + 16], vecC3);
                          _mm256_storeu_ps(&C[ii * NJ + jj + 24], vecC4);
                      }
                  }
              }
          }
      }
  }
  
 
}

int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* opt selects which gemm version to run */
  int opt = 0;
  if(argc == 2) {
    opt = atoi(argv[1]);
  }
  //printf("option: %d\n", opt);
  
  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  switch(opt) {
  case 0: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
    break;
  case 1: // tiling
    /* Run kernel. */
    gemm_tile (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling time");
    break;
  case 2: // tiling and simd
    /* Run kernel. */
    gemm_tile_simd (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd time");
    break;
  case 3: // tiling, simd, and parallelization
    /* Run kernel. */
    gemm_tile_simd_par (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd-par time");
    break;
  default: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
  }
  /* Print results. */
  print_and_valid_array_sum(C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}