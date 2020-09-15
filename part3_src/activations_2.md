# activation

activation function

## linear

```c
static inline float linear_activate(float x){return x;}

static inline float linear_gradient(float x){return 1;}
```

단순한 선형 함수 입니다.

$$Linear(x) = x$$

## logistic

```c
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}

static inline float logistic_gradient(float x){return (1-x)*x;}
```

- `sigmoid`
- 값이 0 ~ 1 사이로 정규화 됩니다.
- 값이 0과 1 주변에 몰려있는 현상이 발생하여 `Gradient Vanishing` 현상이 발생 할 수 있습니다.

$$Sigmoid(x) = \frac{1}{1 + e^{-x}}$$

## loggy

```c
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}

static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
```

- -1 ~ 1 사이로 정규화 됩니다.

$$Loggy(x) = \frac{2}{1 + e^{-x}} - 1$$

## relu

```c
static inline float relu_activate(float x){return x*(x>0);}

static inline float relu_gradient(float x){return (x>0);}
```

- `Rectified linear unit`
- 가장 많이 사용되는 활성화 함수 입니다. 음수면 값을 0으로 만들고 양수면 값을 그대로 통과시킵니다.
- `Gradient Vanishing` 현상을 방지합니다.

- `Gradient Exploding` 현상이 발생할 수 있습니다.
- `Dead relu problem` : 대부분의 값이 0이 되는 경우 업데이트가 되지 않는 문제를 발생시킬 수 있습니다.
- 희소성(`sparsity`)이 매우 높습니다.

$$
RELU(x) = \left\{\begin{matrix}
x && if \quad x > 0\\
0 && if \quad x \leq 0
\end{matrix}\right.
$$

## elu

```c
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}

static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
```

- `Exponential linear unit`
- `Dead relu problem`현상을 방지합니다.

- `Gradient Exploding` 현상이 발생할 수 있습니다.
- `exponential` 연산 때문에 시간이 오래 걸립니다.
- $$\alpha$$ 값이 학습 파라미터가 아닙니다.


$$
ELU(x) = \left\{\begin{matrix}
x && if \quad x \geq 0\\
\alpha (e^x - 1) && if \quad x < 0
\end{matrix}\right.
$$

## selu

```c
static inline float selu_activate(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}

static inline float selu_gradient(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
```

- `Scaled Exponential Linear Unit`
- `Gradient Vanishing` 현상을 방지합니다.
- `Gradient Exploding` 현상을 방지합니다.

- 딥러닝 네트워크를 자체 정규화 합니다.
- `lecun_normal`로 가중치를 초기화 해야합니다.

$$
SELU(x) = \lambda \left\{\begin{matrix}
x && if \quad x \geq 0\\
\alpha(e^x - 1) && if \quad x < 0
\end{matrix}\right.
$$

- $$\alpha$$ : 1.6732
- $$\lambda$$ : 1.0507

#### lecun normal

$$
W ~ N(0, Var(W))
$$

$$
Var(W) = \sqrt{\frac{1}{n_{in}}}
$$

## relie

```c
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}

static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
```

$$
RELIE(x) = \left\{\begin{matrix}
x && if \quad x > 0\\
\alpha x && if \quad x \leq 0
\end{matrix}\right.
$$

- $$\alpha = 0.01$$

## ramp

```c
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}

static inline float ramp_gradient(float x){return (x>0)+.1;}
```

$$
RAMP(x) = \left\{\begin{matrix}
x + 0.1*x && if \quad x > 0\\
0.1*x && if \quad x \leq 0
\end{matrix}\right.
$$

## leaky relu

```c
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}

static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
```

- `Leaky Rectified Linear Unit`
- `Dead relu problem`현상을 방지합니다.

- `Gradient Exploding` 현상이 발생할 수 있습니다.
- $$\alpha$$ 값이 학습 파라미터가 아닙니다.

$$
LRELU(x) = \left\{\begin{matrix}
x && if \quad x > 0\\
\alpha x && if \quad x \leq 0
\end{matrix}\right.
$$

- $$\alpha = 0.1$$

## tanh

```c
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}

static inline float tanh_gradient(float x){return 1-x*x;}
```

- 값이 -1 ~ 1 사이로 정규화 됩니다.
- 값이 0과 1 주변에 몰려있는 현상이 발생하여 `Gradient Vanishing` 현상이 발생 할 수 있습니다.

$$
TANH(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

## plse

```c
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}
```

$$
PLSE(x) = \left\{\begin{matrix}
0.01 * (x + 4) && if \quad x < -4\\
0.01 * (x - 4) + 1 && if \quad x > 4
\end{matrix}\right.
$$

```
    __________________               __________________
                      |             |
                      ---------------
                     -4             4
```

## stair

```c
static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}

static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
```

$$
STAIR(x) = \left\{\begin{matrix}
floor(\frac{x}{2}) && if \quad n \% 2 == 0\\
(x - n) floor(\frac{x}{2}) && else
\end{matrix}\right.
$$

- $$n : floor(x)$$

```
                   __ /
               __ /
           __ /
          /
```

## hardtan

```c
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
```

$$
HARDTAN(x) = \left\{\begin{matrix}
1 && if \quad x > 1\\
-1 && if \quad x < -1 \\
x && if \quad -1 \leq x \leq 1
\end{matrix}\right.
$$

```
                         __________________
                        /  
    __________________/
                     -1  1
```

## lhtan

```c
static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}

static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}
```

$$
LHTAN(x) = \left\{\begin{matrix}
0.001 * (x - 1) + 1 && if \quad x > 1\\
0.001 * x && if \quad x < 0 \\
x && if \quad 0 \leq x \leq 1
\end{matrix}\right.
$$

```
                              __________________
    _________________________/
                            0
```

## Reference
- [https://mlfromscratch.com/activation-functions-explained/#/](https://mlfromscratch.com/activation-functions-explained/#/)
