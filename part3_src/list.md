# list.c

```c
// darknet.h

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

```

- list 구조체입니다.

## make_list

```c
list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}
```

- list를 동적할당하고 값을 0으로 초기화 합니다.

## list_pop

```c
void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;

    return val;
}
```

- list의 맨 뒤에 node의 값을 반환 후 제거합니다.

## list_insert

```c
void list_insert(list *l, void *val)
{
	node *new = malloc(sizeof(node));
	new->val = val;
	new->next = 0;

	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;
		new->prev = l->back;
	}
	l->back = new;
	++l->size;
}
```

- list에 val을 가지는 node를 입력합니다.

## free_node

```c
void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}
```

- node의 메모리 할당을 해제합니다.

## free_list

```c
void free_list(list *l)
{
	free_node(l->front);
	free(l);
}
```

- list의 메모리 할당을 해제합니다.

## free_list_contents

```c
void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}
```

- list내에 val의 메모리 할당을 해제합니다.

## list_to_array

```c
void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
```

- list를 array로 바꿉니다.
