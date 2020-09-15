# blas 란?

벡터의 덧셈, 내적, 선형 조합, 행렬 곱셈과 같은 일반적인 선형 대수 연산을 수행하기 위한 역할을 합니다.

- Basic Linear Algebra Subprograms의 약자입니다.
- 크게 3개의 level(`vector-vector`, `matrix-vector`, `matrix-matrix`)로 구분되어 집니다.

---
# blas.c

## copy_cpu

```c
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}
```

- X -> Y 로 값을 복사하는 함수입니다.

## mean_cpu

```c
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}
```

- 각 필터의 평균을 구하는 함수 입니다.

$$mean = \frac{1}{m} \sum_{i=0}^{m} X[i]$$

각 필터마다 위와 같은 연산을 합니다.

## variance_cpu

```c
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}
```

- 각 필터의 분산을 구하는 함수 입니다.

$$variance = \frac{1}{m - 1} \sum_{i=0}^{m} (X[i] - mean)^2 $$

각 필터마다 위와 같은 연산을 합니다.

## normalize_cpu

```c
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}
```

- 정규화 함수 입니다.


$$X[i] = \frac{X[i] - mean}{\sqrt{variance}}$$


## axpy_cpu

```c
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}
```

- $$Y = Y + alpha * X$$

## scal_cpu

```c
void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}
```

- $$X = X * alpha$$

## fill_cpu

```c
void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}
```

- X에 alpha로 값을 초기화 합니다.

## mul_cpu

```c
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}
```

- $$Y = Y * X$$

## pow_cpu

```c
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}
```

- $$Y = X^{\alpha}$$

## deinter_cpu

```c
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            if(X) X[j*NX + i] += OUT[index];
            ++index;
        }
        for(i = 0; i < NY; ++i){
            if(Y) Y[j*NY + i] += OUT[index];
            ++index;
        }
    }
}
```

- 연결된 배열 X와 Y를 분리합니다.

## inter_cpu

```c
void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}
```

- 배열 X와 Y를 연결합니다.

## mult_add_into_cpu

```c
void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
}
```

$$Z = Z + X*Y$$

## smooth_l1_cpu

```c
void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}
```

- `fabs(double num)` : 인자값으로 들어온 double 타입의 num의 절대값을 반환 합니다.

- 실측값과 예측값의 차이가 1보다 작은 경우
  + error : $$(truth - pred)^2$$
  + delta : $$truth - pred$$

- 실측값과 예측값의 차이가 1보다 큰 경우
  + error : $$2*|truth - pred| - 1$$
  + delta : diff가 0보다 작은 경우 1 큰 경우 -1

## l1_cpu

```c
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}
```

- error : $$|truth - pred|$$
- delta :  diff가 0보다 큰 경우 1 작은 경우 -1

## softmax_x_ent_cpu

```c
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}
```

- error : $$log(pred)$$ 만약 0이면 0
- delta : $$truth - pred$$

## logistic_x_ent_cpu

```c
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p) - (1-t)*log(1-p);
        delta[i] = t-p;
    }
}
```

- error : $$-truth * \log(pred) - (1 - truth) * \log(1 - pred)$$
- delta : $$truth - pred$$

## l2_cpu

```c
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}
```

- error = $$(truth - pred)^2$$
- delta = $$truth - pred$$

## dot_cpu

```c
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}
```

- dot product
- $$dot = dot + X * Y$$

## softmax

```c
void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}
```

- softmax 연산함수

- $$\frac{e(x_i - max(x))}{\sum e(x_i - max(x))}$$

## softmax_cpu

```c
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}
```

- softmax 실행함수

## upsample_cpu

```c
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}
```

- upsampling layer

- 가장 근접한 값으로 값을 채워서 upsampling 합니다.

```
if stride : 2

[1, 2, 3, 4]  -->  [1, 1, 2, 2, 3, 3, 4, 4]
```

## reorg_cpu

```c
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}
```

- reorg layer

- 재구성합니다.

```
if | w : 2 | h : 2 | c : 4 | stride : 2 |

[ 1,  2,  3,  4]            [ 1,  5,  2,  6]
[ 5,  6,  7,  8]     --     [ 9, 13, 10, 14]
[ 9, 10, 11, 12]     --     [ 3,  7,  4,  8]
[13, 14, 15, 16]            [11, 15, 12, 16]
```

## flatten

```c
void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){  
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}
```

- 다차원 배열을 1차원 배열로 만드는 함수 입니다.

- 기존부터 1차원 배열이기 때문에 darknet에서 flatten은 재배열 입니다.


![flatten](/figure/flatten.PNG)



## weighted_sum_cpu

```c
void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}
```

- $$c = s * a + (1 - s) * b$$

## weighted_delta_cpu

```c
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}
```

- $$\delta a = \delta a + \delta c * s $$
- $$\delta b = \delta b + \delta c * (1 - s)$$
- $$\delta s = \delta s + \delta c * (a - b)$$

## shortcut_cpu

```c
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] = s1*out[out_index] + s2*add[add_index];
                }
            }
        }
    }
}
```

- shortcut layer

- output feature map에 추가할 feature map을 더합니다.

## softmax_x_ent_cpu

```c
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}
```

- softmax한 pred와 실제 truth의 delta와 error를 구합니다.

- $$L = - \sum_i y_i log p_i $$ 입니다. 위에 error를 전부 더하면 최종 error가 됩니다. 이러한 error를 Cross Entropy라고 합니다.
