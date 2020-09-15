
# Average Pooling Layer 란?

Feature Map의 평균 값을 계산해 전파시키는 Layer 입니다.

# avgpool_layer.c

```c
//avgpool_layer.h

typedef layer avgpool_layer;
```

## make_avgpool_layer


```c
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;

    return l;
}
```

`Average Pooling Layer`를 만드는 함수입니다.

## forward_avgpool_layer

```c
void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}
```

`forward`



![avgpool](/figure/avgpool.PNG)



- Feature Map의 평균값을 전파 합니다.

## backward_avgpool_layer

```c
void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}
```

`backward`



![avgpool_grad](/figure/avgpool_grad.PNG)



- 국부적 기울기는 $$\frac{1}{h \times w}$$가 되기 때문에 역전파된 기울기 값에 $$\frac{1}{h \times w}$$를 곱해서 역전파합니다.
- 하나의 Feature Map의 모든 기울기 값은 동일합니다.


## resize_avgpool_layer

```c
void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}
```

- avgpooling layer의 크기를 resize합니다.
