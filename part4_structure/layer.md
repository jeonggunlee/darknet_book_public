## Layer Structure

Layer 구조체 입니다.

```c
struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;                              /// input height, width, channel
    int out_h, out_w, out_c;                /// output height, width, channel
    int n;                                  /// number of filter(= output channel)
    int max_boxes;
    int groups;
    int size;                               /// kernel size
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu;

    float * binary_input;

    struct layer *input_layer;                      /// rnn input layer
    struct layer *self_layer;                       /// rnn self layer
    struct layer *output_layer;                     /// rnn output layer

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;
};
```
