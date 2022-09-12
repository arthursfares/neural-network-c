# Neural Network in C

:warning: _Run the application on the same directory as the matrices folder, otherwise it will not read the data._\
:warning: _Some of the formulas below are not displaying correctly. Will fix it soon._

## References

The logic and structure of the program were based on [this](https://pabloinsente.github.io/the-multilayer-perceptron) absolutely great tutorial by https://github.com/pabloinsente on how to develop an implementation of a multilayer perceptron in python by using matrix operations.\
Also, [this](http://neuralnetworksanddeeplearning.com/index.html) online book on Neural Networks by https://github.com/mnielsen was of incredible help on the understanding of the calculations involved in the backpropagation algorithm.\
The _eldritch_arrays.h_ functions for matrix operations were based on [this](https://www.andreinc.net/2021/01/20/writing-your-own-linear-algebra-matrix-library-in-c) tutorial, by https://github.com/nomemory. Easy to follow and renewed my interest in some linear algebra concepts.

##

What follows is the matrix caulculations for an example with only one hiden layer (_planning on adding more on the future_) which contains 3 neurons. It receives 2 features as inputs and produces one output.\
_It is structured that way to solve the XOR problem._

<p align="center">
  <img src="https://c.tenor.com/Jl-mymXzhywAAAAC/elmo-fire.gif" alt="elmo-flame-gif" />
</p>

### Features

<div>

$$\boldsymbol{X} =
\begin{bmatrix}
   feature_{00} \\
   feature_{10}
\end{bmatrix}_{2\times1}$$

</div>

### Targets

<div>

$$\boldsymbol{y} =
\begin{bmatrix}
   target_{00}
\end{bmatrix}_{1\times1}$$

</div>

### Parameters

<div>

$$\boldsymbol{W1} =
\begin{bmatrix}
   w_{00} & w_{01} \\
   w_{10} & w_{11}
\end{bmatrix}_{2\times2} 
\hspace{5em} 
\boldsymbol{b1} =
\begin{bmatrix}
   b_{00} \\
   b_{10}
\end{bmatrix}_{2\times1}$$

</div>

<div>

$$\boldsymbol{W2} =
\begin{bmatrix}
   w_{00} & w_{01}
\end{bmatrix}_{1\times2} 
\hspace{5em} 
\boldsymbol{b2} =
\begin{bmatrix}
   b_{00}
\end{bmatrix} _{1\times1}$$

</div>

The $w$ elements inside the $W1$ and $W2$ matrices do not represent the same values for the corresponding indices, the same goes for the $b$ elemets.\
(_e.g._) The $w_{00}$ from $W1$ is not the same as the $w_{00}$ from $W2$.\
I chose to omit the correspoing matrix index from the elements, because there are already enough indices in the calculations.\
Hope this doesn't spoil the understanding of the equations :persevere:

## Feedforward

<details>
<summary style="font-style: italic;">details</summary>

#### Linear function
Each element of the $\boldsymbol{Z}$ matrices result from a linear combination between a input ($feature$) reaching the neuron and its respective weight ($w$), plus its bias term ($b$).

<div>

$$z_{(x, w, b)} = (w \cdot x) + b$$

</div>

#### Sigmoid function
Acting as the neurons activation function, is the sigmoid. 

<div>

$$\sigma_{(z)} = \frac{1}{(1 + e^{-z})}$$

</div>

By applying it to every element of $\boldsymbol{Z}$, we get the resulting matrix $\boldsymbol{S}$.

---
</details>

<div>

$$\boldsymbol{Z1}_{2\times1} = (\boldsymbol{W1}_{2\times2} \times \boldsymbol{X}_{2\times1}) + \boldsymbol{b1}_{2\times1} =
\begin{bmatrix}
   z_{00} \\
   z_{10}
\end{bmatrix}_{2\times1}$$

</div>

<div>

$$\boldsymbol{S1}_{2\times1} = \sigma(\boldsymbol{Z1}_{2\times1}) =
\begin{bmatrix}
   a_{00} \\
   a_{10}
\end{bmatrix}_{2\times1}$$

</div>

<div>

$$\boldsymbol{Z2}_{1\times1} = (\boldsymbol{W2}_{1\times2} \times \boldsymbol{S1}_{2\times1}) + \boldsymbol{b2}_{1\times1} =
\begin{bmatrix}
   z_{00} \\
\end{bmatrix}_{1\times1}$$

</div>

<div>

$$\boldsymbol{S2}_{1\times1} = \sigma(\boldsymbol{Z2}_{1\times1}) =
\begin{bmatrix}
   a_{00}
\end{bmatrix}_{1\times1}$$

</div>

## Backpropagation

:alien: $\ \boldsymbol{\odot}$ represents the _Hadamard product operator_, which performs the element wise matrix multiplication operation.

:alien: $\ \boldsymbol{\eta}$ represents the _learning rate_.


<details>
<summary style="font-style: italic;">details</summary>

#### Sigmoid function
We'll need to apply the derivative of the sigmoid function on the $\boldsymbol{Z}$ matrices, so let's see how that is calculated element wise. 

<div>

$$\textcolor{green}{{\sigma_{(z)}} = \frac{1}{(1 + e^{-z})}}$$

</div>

<div>

$$\boxed{\sigma'_{(z)}} = \frac{\delta\sigma_{(z)}}{\delta z} = \frac{\delta}{\delta z}(1 + e^{-z})^{-1} = (-1)\cdot(1 + e^{-z})^{-2}\cdot\frac{\delta}{\delta z}(1 + e^{-z}) =$$

</div>

<div>

$$= (-1)\cdot(1 + e^{-z})^{-2}\cdot \Big[\cancel{\frac{\delta}{\delta z}(1)} + \frac{\delta}{\delta z}(e^{-z}) \Big] =$$

</div>

<div>

$$= (-1)\cdot(1 + e^{-z})^{-2}\cdot \Big[ e^{-z} \cdot \frac{\delta}{\delta z}(-z) \Big] = \cancel{(-1)}\cdot(1 + e^{-z})^{-2}\cdot \Big[ e^{-z} \cdot \cancel{(-1)} \Big] =$$

</div>

<div>

$$= \textcolor{green}{\frac{1}{(1 + e^{-z})}} \cdot \frac{e^{-z}}{(1 + e^{-z})} = \sigma_{(z)} \cdot \Big[ \frac{\textcolor{orange}{+1} +  e^{-z} \textcolor{orange}{-1}}{(1 + e^{-z})} \Big] = \sigma_{(z)} \cdot \Big[ \frac{1 + e^{-z}}{(1 + e^{-z})} - \textcolor{green}{\frac{1}{(1 + e^{-z})}} \Big] =$$

</div>

<div>

$$= \boxed{\sigma_{(z)} \cdot (1 - \sigma_{(z)})}$$

</div>

Therefore, all the derivatives of the sigmoids can be computed using the values from the sigmoids themselves, without the need to look back on the values of z.

<div>

$$\sigma'(\boldsymbol{Z}) = \boldsymbol{S} \  \odot \ (\boldsymbol{ones} - \boldsymbol{S})$$

</div>

Where $\boldsymbol{ones}$ is a matrix, with same dimension as $\boldsymbol{S}$, but filled with ones.

---
</details>

### Update output weights

<div>

$$\boldsymbol{\delta2}_{1\times1} = (\boldsymbol{S2}_{1\times1} - \boldsymbol{y}_{1\times1}) \ \odot \  \sigma'(\boldsymbol{Z2}_{1\times1}) =
\begin{bmatrix}
   d_{00}
\end{bmatrix}_{1\times1}$$

</div>

<div>

$$\boldsymbol{\nabla W2}_{1\times2} = \boldsymbol{\delta2}_{1\times1} \ \times \ (\boldsymbol{S1}^T)_{1\times2}  =
\begin{bmatrix}
   dw_{00} & dw_{01}
\end{bmatrix}_{1\times2}$$

</div>

<div>

$$\boldsymbol{new\  W2}_{1\times2} = \boldsymbol{W2}_{1\times2} - \eta\boldsymbol{\nabla W2}_{1\times2} = 
\begin{bmatrix}
   new \ w_{00} & new \ w_{01}
\end{bmatrix}_{1\times2}$$

</div>

### Update the output bias

<div>

$$\boldsymbol{\nabla b2}_{1\times1} = \boldsymbol{\delta2}_{1\times1}$$

</div>

<div>

$$\boldsymbol{new\  b2}_{1\times1} = \boldsymbol{b2}_{1\times1} - \eta\boldsymbol{\delta2}_{1\times1} = 
\begin{bmatrix}
   new \ b_{00}
\end{bmatrix}_{1\times1}$$

</div>

### Update the hidden weights

<div>

$$\boldsymbol{\delta1}_{2\times1} = (\ (\boldsymbol{W2}^T)_{2\times1} \times \boldsymbol{\delta2}_{1\times1}\ ) \ \odot \  \sigma'(\boldsymbol{Z1}_{2\times1}) =
\begin{bmatrix}
   d_{00} \\
   d_{10}
\end{bmatrix}_{2\times1}$$

</div>

<div>

$$\boldsymbol{\nabla W1}_{2\times2} = \boldsymbol{\delta1}_{2\times1} \ \times \ (\boldsymbol{X}^T)_{1\times2}  =
\begin{bmatrix}
   dw_{00} & dw_{01} \\
   dw_{10} & dw_{11}
\end{bmatrix}_{2\times2}$$

</div>

<div>

$$\boldsymbol{new\  W1}_{2\times2} = \boldsymbol{W1}_{2\times2} - \eta\boldsymbol{\nabla W1}_{2\times2} = 
\begin{bmatrix}
   new \ w_{00} & new \ w_{01} \\
   new \ w_{10} & new \ w_{11}
\end{bmatrix}_{2\times2}$$

</div>

### Update the hidden bias

<div>

$$\boldsymbol{\nabla b1}_{2\times1} = \boldsymbol{\delta1}_{2\times1}$$

</div>

<div>

$$\boldsymbol{new\  b1}_{2\times1} = \boldsymbol{b1}_{2\times1} - \eta\boldsymbol{\delta1}_{2\times1} = 
\begin{bmatrix}
   new \ b_{00} \\
   new \ b_{10} \\
\end{bmatrix}_{2\times1}$$

</div>

