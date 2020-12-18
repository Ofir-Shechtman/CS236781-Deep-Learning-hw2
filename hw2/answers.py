r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0.1, 0.05, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.

    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 1e-3, 2e-2, 3e-3, 1e-4, 1e-3

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.

    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 1e-3, 3e-3
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1. The number of parameters for a Convolution layer is $Input_{featue\space map} ⋅ Output_{featue\space map} ⋅ Filter size$.<br>
Therefore,<br>
Regular Block: $256⋅256⋅3⋅3+256⋅256⋅3⋅3=1,179,648$<br>
Bottleneck Block: $256⋅64⋅1⋅1+64⋅64⋅3⋅3+64⋅256⋅1⋅1=69,632$<br>
we can see that the bottleneck reduces the number of parameters.

2. the number of multiplication on a single convolution layer is **width ⋅ lengh** of each output feature map, multuply by
the **#parameters** of the convolution.<br>
Therefore,<br>
Regular Block: $1179648⋅$**width⋅lengh**<br>
Bottleneck Block: $69632⋅$**width⋅lengh**<br>
we can see that the bottleneck reduces the number of operations.
"""

part3_q2 = r"""
the number of multiplication on a single convolution layer is **width ⋅ lengh** of each output feature map, multuply by
the **#parameters** of the convolution.<br>
Therefore,<br>>

Regular Block:1179648⋅**width ⋅ lengh**$<br>
Bottleneck Block: 69632⋅**width ⋅ lengh**$<br>

Thus, we can see that the bottleneck immensily reduces the number of operations.


"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q6 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
