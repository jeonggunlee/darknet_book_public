# GEMM 이란?

참고 자료 : [https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

- `General Matrix to Matrix Multiplication`
- 1979년에 만들어진 BLAS 라이브러리의 일부 입니다.
- 두개의 입력 행렬을 곱해서 출력을 얻는 방법 입니다.

딥러닝에서 대부분의 연산은 `output = input * weight + bias`로 표현이 됩니다. 여기서 `input`, `output`, `weight`를 행렬로 표현해서 GEMM을 사용해 연산할 수 있습니다.



![gemm1](/figure/gemm1.PNG)



## Fully Connected Layer



![gemm2](/figure/gemm2.PNG)



`fully connected layer`는 위와 같이 표현할 수 있습니다.


## Convolutional Layer



![gemm3](/figure/gemm3.PNG)



- `im2col` : 3차원 이미지 배열을 2차원 배열로 변환합니다.

`convolutional layer`는 위와 같이 표현할 수 있습니다.
위 그림의 경우는 `stride`가 `kernel size`와 같은 경우를 의미합니다.

---
# gemm.c

## gemm

gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);

```c
void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu(TA,  TB,  M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}
```

- `TA`    : A 행렬의 전치 (0 : 사용안함 | 1 : 사용)
- `TB`    : B 행렬의 전치 (0 : 사용안함 | 1 : 사용)
- `M`     : filter 개수
- `N`     : output feature map 크기
- `K`     : filter 크기
- `ALPHA` : scale factor
- `A`     : filter
- `lda`   : A를 위한 포인터
- `B`     : input feature maps
- `ldb`   : B를 위한 포인터
- `BETA`  : initial value
- `C`     : output feature maps
- `ldc`   : C를 위한 포인터


## gemm_cpu

```
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A,lda, B, ldb, C, ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA, A,lda, B, ldb, C, ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA, A,lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A,lda, B, ldb, C, ldc);
}
```

- `BETA`로 `C(output feature maps)`를 초기화 합니다.
- GEMM 연산을 진행합니다.




![gemm4](/figure/gemm4.PNG)



최종적으로 GEMM 연산은 위와 같이 표현이 될 수 있습니다.

## gemm_nn

```
void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}
```

- `A` : 전치행렬 사용 안함
- `B` : 전치행렬 사용 안함

## gemm_nt

```
void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}
```

- `A` : 전치행렬 사용 안함
- `B` : 전치행렬 사용

## gemm_tn

```
void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}
```

- `A` : 전치행렬 사용
- `B` : 전치행렬 사용 안함

## gemm_tt

```
void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}
```

- `A` : 전치행렬 사용
- `B` : 전치행렬 사용
