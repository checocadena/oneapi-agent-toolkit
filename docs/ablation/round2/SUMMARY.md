| Model | Example | Arm | correct + 0 helpers | wrong output | dpct helpers left | said done but wasn't | cost |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5-5 | **all** | toolkit | 20/20 | 0/20 | 0/20 | 0/20 | $1.37 |
| claude-sonnet-5-5 | **all** | baseline | 10/20 | 2/20 | 10/20 | 10/20 | $1.23 |
| claude-sonnet-5-5 | **all** | baseline_shell | 7/20 | 0/20 | 13/20 | 12/20 | $1.20 |
| claude-sonnet-5-5 | saxpy | toolkit | 5/5 | 0/5 | 0/5 | 0/5 | $0.17 |
| claude-sonnet-5-5 | saxpy | baseline | 1/5 | 0/5 | 4/5 | 4/5 | $0.30 |
| claude-sonnet-5-5 | saxpy | baseline_shell | 3/5 | 0/5 | 2/5 | 2/5 | $0.14 |
| claude-sonnet-5-5 | fuzz | toolkit | 5/5 | 0/5 | 0/5 | 0/5 | $0.43 |
| claude-sonnet-5-5 | fuzz | baseline | 4/5 | 0/5 | 1/5 | 1/5 | $0.38 |
| claude-sonnet-5-5 | fuzz | baseline_shell | 1/5 | 0/5 | 4/5 | 4/5 | $0.35 |
| claude-sonnet-5-5 | fir | toolkit | 5/5 | 0/5 | 0/5 | 0/5 | $0.30 |
| claude-sonnet-5-5 | fir | baseline | 3/5 | 0/5 | 2/5 | 2/5 | $0.24 |
| claude-sonnet-5-5 | fir | baseline_shell | 2/5 | 0/5 | 3/5 | 3/5 | $0.29 |
| claude-sonnet-5-5 | softmax_hecbench | toolkit | 5/5 | 0/5 | 0/5 | 0/5 | $0.48 |
| claude-sonnet-5-5 | softmax_hecbench | baseline | 2/5 | 2/5 | 3/5 | 3/5 | $0.31 |
| claude-sonnet-5-5 | softmax_hecbench | baseline_shell | 1/5 | 0/5 | 4/5 | 3/5 | $0.43 |
