| Example | Arm | correct + 0 helpers | wrong output | dpct helpers left | said done but wasn't | cost |
|---|---|---|---|---|---|---|
| saxpy | toolkit | 3/3 | 0/3 | 0/3 | 0/3 | $0.10 |
| saxpy | baseline | 1/3 | 0/3 | 2/3 | 2/3 | $0.14 |
| fuzz | toolkit | 3/3 | 0/3 | 0/3 | 0/3 | $0.30 |
| fuzz | baseline | 3/3 | 0/3 | 0/3 | 0/3 | $0.19 |
| fir | toolkit | 3/3 | 0/3 | 0/3 | 0/3 | $0.19 |
| fir | baseline | 2/3 | 0/3 | 1/3 | 1/3 | $0.15 |
| softmax_hecbench | toolkit | 3/3 | 0/3 | 0/3 | 0/3 | $0.31 |
| softmax_hecbench | baseline | 1/3 | 1/3 | 1/3 | 2/3 | $0.19 |
| **all** | toolkit | 12/12 | 0/12 | 0/12 | 0/12 | $0.90 |
| **all** | baseline | 7/12 | 1/12 | 4/12 | 5/12 | $0.67 |
