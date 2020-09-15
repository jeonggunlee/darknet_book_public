# Activation Function 이란?

*퍼셉트론과 딥러닝의 차이점*

딥러닝 네트워크에서 `Layer`사이에 항상 활성화 함수(Activation Function)가 있습니다. 이러한 `activation function`은 어떤 역할을 하는 것 일까요?
활성화 함수란 값을 활성화시키는 비선형 함수 입니다.
비선형 함수가 활성화 함수의 핵심 입니다.
만약 딥러닝 네트워크에 활성화 함수가 없고 선형 함수만 존재하는 경우 Layer를 아무리 추가해도 결국은 하나의 Layer가 있는 네트워크와 같게 됩니다. ( $$a(a(ax + b) + b) + b = ax' + b$$ )
즉, 딥러닝 네트워크는 활성화 함수의 존재로 Layer가 깊어질 수 있고 수많은 연산으로 많은 것을 표현할 수 있게 됩니다.

---

# activations.c

```c
// darknet.h

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;
```

- `enum` : 열거형, 정수형 상수를 선언할 때 일일이 선언하지 어려운 부분을 해결하기 위한 키워드 입니다.

위에 열거형의 경우 이름은 `ACTIVATION`이고 `LOGISTIC : 0 ... SELU : 13` 값이 할당 됩니다.

## get_activation_string

```c
char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}
```

`ACTIVATION`을 입력받아 문자열과 매핑하는 함수입니다.

## get_activation

```c
ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}
```

문자열을 입력받아 `ACTIVATION`과 매핑하는 함수입니다.

## activate

```c
float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}
```

- `ACTIVATION`에 해당하는 함수를 반환하는 함수입니다.

- activation 함수를 적용하기 위한 함수입니다.

## gradient

```c
float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}
```

- `ACTIVATION`에 해당하는 gradient를 반환하는 함수입니다.

- activation gradient 함수를 적용하기 위한 함수입니다.
