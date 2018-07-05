Assume the network processes as time steps $\{t~\rvert~t\in[1, T]\}$ and the overall network error at time $t$ is denoted as $E(t)$. Define $E^{total}$ as $\displaystyle \sum_t E(t)$.

Each weight $w_{lm}$ in the network appears at each time step and is denoted as $w_{lm}(t)$ correspondingly. We have

$$\frac{\partial E(t)}{\partial w_{lm}} 
= \sum_{\tau\le t}\frac{\partial E(t)}{\partial w_{lm}(\tau)} $$

We define $d w_{lm}(t, \tau)$ as $\displaystyle\frac{\partial E(t)}{\partial w_{lm}(\tau)}$ for simplification, thus

$$d w_{lm} = \sum_t d w_{lm}(t)= \sum_t\sum_{\tau\le t}d w_{lm}(t, \tau)$$

It is the sum of elements in the lower-triangular weight matrix (denoted as $d W_{lm}$):

$$
d W_{lm} = \begin{bmatrix}
d w_{lm}(1, 1) \\
d w_{lm}(2, 1) & d w_{lm}(2, 2) \\
\vdots & \vdots & \ddots \\
d w_{lm}(T-1, 1) & d w_{lm}(T-1, 2) & \cdots  & d w_{lm}(T-1, T-1) \\
d w_{lm}(T, 1) & d w_{lm}(T, 2) & \cdots & d w_{lm}(T, T-1) & d w_{lm}(T, T)\\
\end{bmatrix}
$$

The $t^{th}$ row $d W_{lm}(t, :)$ represents the sensitivities of the error $E(t)$ to small perturbations in the weight $w_{lm}$ in previous time steps, i.e. $\big\{w_{lm}(\tau)\big\}_{\tau\le t}$. The $\tau^{th}$ column $dW_{lm}(:, \tau)$ implies how a minor oscillation in $w_{lm}$  can affect the overall performance at the subsequent time steps, i.e. $\big\{E(t)\big\}_{t\ge\tau}$.

Note that $\forall l_1, l_2, m_1, m_2, t_1, t_2, \tau_1, \tau_2$, $d w_{l_1~m_1}(t_1, \tau_1)$ and $d w_{l_2~m_2}(t_2, \tau_2)$ are independent since weights are assumed to be fixed during one train step.

#### References
1. Williams, Ronald J. and Zipser,
 David. Gradient-based learning algorithms for recurrent networks and their computational complexity. 
 Backpropagation, pages 433-486. 1995.