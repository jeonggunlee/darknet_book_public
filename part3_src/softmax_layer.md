## softmax란?

참조 : [https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/](https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/)

- 입력의 모든 합을 1로 만드는 함수 입니다.

- $$p_i = \frac{exp(x_i)}{\sum^{C}_{c=1} exp{x_c}}$$

softmax는 역전파의 시작점 입니다.

우리는 error를 통해서 softmax input의 gradient를 구해야합니다. (그래야 뒤로 쭉쭉 갈수 있겠죠?)

먼저 연산을 위해 미리 softmax를 미분한다면

$$i = j$$ 일때

- $$\frac{\partial p_i}{\partial x_i} = \frac{\partial \frac{exp(x_i)}{\sum^{C}_{c=1} exp(x_c)}}{\partial x_i}$$

- $$\frac{\partial p_i}{\partial x_i} = \frac{exp(x_i) \sum^{C}_{c=1} exp(x_c) - exp(x_i) exp(x_i)}{(\sum^{C}_{c=1} exp(x_c))^2}$$

- $$ = \frac{exp(x_i) [ \sum^{C}_{c=1} \left \{ \exp(x_c) \right \} - exp(x_i)]}{(\sum^{C}_{c=1} exp(x_c))^2}$$

- $$ = \frac{exp(x_i)}{\sum^{C}_{c=1} exp(x_c)} \frac{\sum^{C}_{c=1} \left \{ exp(x_c) \right \} - exp(x_i) }{(\sum^{C}_{c=1} exp(x_c))}$$

- $$ = \frac{exp(x_i)}{\sum^{C}_{c=1} exp(x_c)} \left ( 1 - \frac{exp(x_i)}{\sum^{C}_{c=1} exp(x_c)} \right ) $$

- $$ = p_i (1 - p_i)$$

$$i \neq j$$ 일때

- $$\frac{\partial p_i}{\partial x_j} = \frac{0 - exp(x_i) exp(x_j)}{(\sum^{C}_{c=1} exp(x_c))^2}$$

- $$ = - \frac{exp(x_i)}{\sum^{C}_{c = 1} exp(x_c)} \frac{exp(x_j)}{\sum^{C}_{c=1} exp(x_c)}$$

- $$ = - p_i p_j$$

역전파

- $$\frac{\partial L}{\partial x_i} = \frac{\partial (- \sum_{j} y_j \log p_j )}{ \partial x_i }$$

- $$ = - \sum_j y_j \frac{\partial \log p_j}{\partial x_i}$$

- $$ = - \sum_j y_j \frac{1}{p_j} \frac{\partial p_j}{\partial x_i}$$

- $$ = - \frac{y_i}{p_i} p_i (1 - p_j) - \sum_{i \neq j} \frac{y_j}{p_j} (- p_i p_j)$$

- $$ = - y_i + y_i p_i + \sum_{i \neq j} y_j p_i$$

- $$ = - y_i + \sum_j y_j p_i$$

- $$ = - y_i + p_i \sum_j y_j$$

- $$p_i - y_i$$

---

# softmax_layer.c

## forward_softmax_layer

```c
void forward_softmax_layer(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
```

`forward`

## backward_softmax_layer

```c
void backward_softmax_layer(const softmax_layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1); // network delta = layer delta
}
```

`backward`

## make_softmax_layer

```c
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;


    return l;
}
```

`update`
