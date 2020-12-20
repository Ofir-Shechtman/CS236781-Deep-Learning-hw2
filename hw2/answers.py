r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. The size of each output vector is equal to the size ot the weights W(x,y) = [2048,1024].
There are 128 Jacobians, because X containing a batch of 128 samples
and the output has the same dimantion, and every output in the batch has a Jacobian w.r.t it's input.
So the shape of the Jacobian tensor would be: [128, 2048, 1024].

2. Size of element (32 bits) * The shape of the Jacobian tensor =
32*128*2048*1024=8,589,934,592 bits = 1,073,741,824 Bytes.
So it will required 1 GB.
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
1. dropout works like regulation. With the help of dropout we ascend a certain amount of neurons during training at
random. In this way I force the model not to be too dependent on specific neurons. We therefore expect the use to overfitting the data.
We expect that without a dropout we will reach the highest accuracy during training, which will result in overfitting.
And as shown in the graph as we observed when there is a dropout we get better regulation and thus the test results were higher.
2. On the other hand a dropout as high as 0.8 will cause the model to be too general and it will not be able to study
the data well, and will therefore show low results in training as well as in the test.
We can see that for dropout = 0.4 we get the best results, since there is a good balance between the two ends."""

part2_q2 = r"""
**Your answer:**
Yes, this is a possible situation because as we know for classification problems, like when using softmax, accuracy
is based only on the maximum value of the predicted value, while the loss is defined by the cross entropy.
Therefore there is a possibility of incompatibility between the two indices and it is possible for the test loss to
increase for a few epochs while the test accuracy also increases
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
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
**Your answer:**
Analyze your results from experiment 1.1. In particular,
1. Our best result were with: `L=2, K=32/64`. the effect of depth on the accuracy is that large depth got low accuracy.<br>
We think that reason for such behaviour is probably vanishing gradients. Because the length of the network, the gradients diminish over the layers causing them to vanish.
2. In our case when `L=16` the accuracy is 10% which is random class choice in 10 classes options.<br>
two things which may be done to resolve it at least partially:
- skip connections, like in Residual Block.
- batch Normalization layers which make the gradient more stable."""

part3_q3 = r"""
**Your answer:**
We can see that when `L=8` the network was untrainable, similar to what happens in  experiment 1.1.
For L=2 and L=4, we can see that for larger values of K the test accuracy improves. we can explain that by the fact that for bigger
values of K, the model has more paramerts and calculation on each layer. and because of that find more features.
"""

part3_q4 = r"""
**Your answer:**
We can see that the network have good accuracy with `L=1` and `L=2`. However, for `L=3` and `L=4`, we got bad results.<br>
Similar to experiment 1.1, experiment 1.2 that happens because the net is too deep to train.
"""

part3_q5 = r"""
**Your answer:**
In this experiment, we use Residual Networks, we can see that the skip relationships that allow the gradient to
flow better through the network, help the model converge even in deep models.<br>
For example, in experiment 1.1, `L=16, K=32` is not trainable due to vanishing gradients. Now using skip connections,
we manage to train it succesfully, even for deepener network L=32. Moreover, in Exp 1.3, we could not converge
in L=3,4. Now using residual blocks and skip connections, the network converges for `L=4` and even for `L=8`.
"""

part3_q6 = r"""
**Your answer:**
1. We use our implementation of Residual Networks with Dropout layers and Batch normalization. We tried a lot of ideas
for other models but in the end we saw that the improvement is not noticeable in relation to our Residual Networks model.
2. With `L=3' we got our best test result with accuracy of 80%. It is likely that better results can be achieved, with the
help of optimizing hyper parameters and other small changes in the model.
"""
# ==============
