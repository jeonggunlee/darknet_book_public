## Network Structure

Network 구조체 입니다.

- `n` : layer의 수
- `batch` : batch 크기
- `seen` :
- `t` :
- `epoch` : epoch 수
- `subdivisions` :
- `layers` :
- `output` :
- `policy` : learning rate policy
- `learning_rate` : 학습률
- `momentum` : learning rate의 운동에너지를 감소시키기 위한 값

```c
# origin

weight = weight + learning rate * dL / dw

# momentum

velocity = momentum * velocity - learning rate * dL / dw
weight = weight + velocity
```

만약 momentum이 0.9이고 2번을 한다고 가정하면, $$velocity = 0.9 * (-\frac{\partial L}{\partial W_1}) - \lambda \frac{\partial L}{\partial W_2}$$

- `decay` : weight decay (L2 regularization)

L2 regularization은 가중치가 크면 클수록 큰 페널티를 줘서 overfitting을 방지하는 방법입니다.

Loss함수에 $$\frac{1}{2} \lambda W W^T$$를 더합니다. 여기서 $$\lambda$$가 `decay`입니다. 클수록 가중치에 큰 페널티를 줍니다.

$$\frac{1}{2}$$가 있는 이유는 미분시 $$W^2$$의 2가 내려와서 값을 1로 만들기 위함입니다.

$$W = W - learning rate * (\frac{\partial L}{\partial W} + \lambda W)$$

- `gamma` : learning rate 감소율
- `scale` :
- `power` :
- `time_steps` :
- `step` : 현재 스탭
- `max_batches` :
- `scales` :
- `steps` :
- `num_steps` :
- `burn_in` :
- `adam` :
- `B1` :
- `B2` :
- `eps` :
- `inputs` :
- `outputs` :
- `truths` :
- `notruth` :
- `h, w, c` :
- `max_crop` :
- `min_crop` :
- `max_ratio` :
- `min_ratio` :
- `center` :
- `angle` :
- `aspect` :
- `exposure` :
- `saturation` :
- `hue` :
- `random` :
- `gpu_index` :
- `hierarchy` :
- `input` :
- `truth` :
- `delta` :
- `workspace` :
- `train` :
- `index` :
- `cost` :
- `clip` :

```c
typedef struct network{
    int n;                            
    int batch;                        
    size_t *seen;
    int *t;
    float epoch;                     
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;
} network;
```
