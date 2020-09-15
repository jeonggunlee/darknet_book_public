# logistic.c

## forward_logistic_layer

```c
void forward_logistic_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, LOGISTIC);
    if(net.truth){
        logistic_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}
```

`forward`

## backward_logistic_layer

```c
void backward_logistic_layer(const layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}
```

`backward`

## make_logistic_layer

```c
layer make_logistic_layer(int batch, int inputs)
{
    fprintf(stderr, "logistic x entropy                             %4d\n",  inputs);
    layer l = {0};
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_logistic_layer;
    l.backward = backward_logistic_layer;
    #ifdef GPU
    l.forward_gpu = forward_logistic_layer_gpu;
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    #endif
    return l;
}
```

`make`
