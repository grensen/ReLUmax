## ReLUmax - make the Softmax Function great again with ReLUs!
## Multi-class Perceptron with new kind of Output Activation Demo Using C#

So many times I've thought about improving the softmax function for the output neurons.
The conclussion was that this function is perfect in so many ways. 
Last weekend I was thinking about a crazy idea, 
but for that purpose the output needs to handle the ReLU neurons.

Finally, this idea was leading me to the softmax with ReLU activation, 
not for my purpose but it seems it could be a heavy speed up for linear models.

Think about the ReLU, if the netinput value is > 0 we take this value, if not we set this neuron to 0.
So we drop this neuron out, no matter which negative value was inside. 

Let's think about we would do that with any output neuron which has a value less then 0?
We could skip all iterations for this neuron on the backpropagation, which speeds up the training process.

Take another look, the demo uses (1 - output) to calculate the gradient of the target, 
for simplicity let's change the sign and the equation into (output - 1) for the target neuron.

So if every output neuron would be zero because of ReLU activation, 
the target neuron would be punished by the max of -1 and all other output neurons would be skipped, neat!
And it works fine and fast you can see in the figure below.

<p align="center">
<img src="https://raw.githubusercontent.com/grensen/ReLUmax/master/figure1.png">
</p>

Finally the best I know this approach is new, and some experiments for both implementations showed some lights and shadows.
But why should we really use functions like that?
2020 was leading to a breaktrough with GPT-3 and a crazy hype about that. 
GPT-3 is not so far from GPT-2, the main difference is just the size, it is a much bigger network.
The training of GPT-3 cost millions, and to train a theoretical GPT-4 network today, it would cost billions!!!

Other ideas inside GPT-3 are Transformers and the Attention mechanism.
And here we have many linear networks inside, but not only with 10 outputs like in the demo.
Imagine output vectors with 100, 1000 or more output neurons, in the best case instead of calculating every output neuron, we only train the target neuron, BAM!

In the spirit of Lex Fridman, I hope and fight for a better world and our future.
If you have your chance, be a part of it!
