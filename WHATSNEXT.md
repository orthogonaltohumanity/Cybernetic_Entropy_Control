Let $L = \mathbb{R}^{W \times d}$ be the space of all possible context windows of fixed size $W$ with token embedding dimension $d$. Define the following subsets:

- $A \subset L$: the set of all contexts appearing in the training data.
- $T \subset L$: the set of all "factually true" contexts.

**Assumption (Truth Bubble).** $A \subset T$, and for all $x \in A$ and all $\epsilon > 0$, there exists $y \in T \setminus A$ such that $d(x, y) < \epsilon$. That is, the training data is surrounded by a neighborhood of factually true states, i.e. there exists a bubble of truth around the training subset in the token embedding space.

## The Transformer as an Iterated Map

Let $F: L \to \mathbb{R}^d$ be the transformer, treated as a continuous differentiable map from a context to a single token embedding vector. Given a context $c_t \in L$, the update rule appends the generated token and shifts the window:

$$c_{t+1} = [c_t,\; F(c_t),\; 0]$$

where the notation denotes appending $F(c_t)$ to the context and padding or shifting as needed to maintain the fixed window size $W$.

**Objective.** We want to ensure that the generated context remains within the truth bubble:

$$\min_{c' \in A} d(c',\; c_{t+1}) < \epsilon$$

for some upper bound $\epsilon > 0$. That is, we want to keep the trajectory close to the training data manifold.

## Expansion Rate Analysis

Suppose we start with a context in the training data, $c_0 \in A$. After one step:

$$c_1 = [c_0,\; F(c_0),\; 0]$$

The displacement from the initial state satisfies:

$$d(c_1, c_0) = \|F(c_0)\|$$

Now consider a perturbation $\delta$ of the initial context. The displaced trajectory gives:

$$d(F(c_0 + \delta),\; c_0 + \delta) = \|F(c_0 + \delta) - \delta\|$$

Taylor expanding $F$ about $c_0$:

$$F(c_0 + \delta) = F(c_0) + J_F(c_0)\,\delta + \mathcal{O}(\|\delta\|^2)$$

Substituting:

$$\|F(c_0 + \delta) - \delta\| = \|F(c_0) + J_F(c_0)\,\delta - \delta + \mathcal{O}(\|\delta\|^2)\|$$

$$= \|F(c_0) + (J_F(c_0) - I)\,\delta + \mathcal{O}(\|\delta\|^2)\|$$

**Remark.** If $\delta$ lies in the eigenspace of $J_F(c_0)$ corresponding to eigenvalue $\lambda = 1$, then $(J_F(c_0) - I)\,\delta = 0$ and the expression reduces to $\|F(c_0)\|$, the unperturbed case. That is, perturbations in the $\lambda = 1$ eigenspace are invisible to this expansion measure.

## Rate of Expansion

The rate of expansion under perturbation is:

$$r(\delta) = \frac{d(F(c_0 + \delta),\; c_0 + \delta) - d(F(c_0),\; c_0)}{\|\delta\|}$$

$$= \frac{\|F(c_0) + (J_F(c_0) - I)\,\delta + \mathcal{O}(\|\delta\|^2)\| - \|F(c_0)\|}{\|\delta\|}$$

Applying the reverse triangle inequality $\|x + y\| \geq \|x\| - \|y\|$ and $\|x + y\| \leq \|x\| + \|y\|$, we obtain:

$$|r(\delta)| \leq \frac{\|(J_F(c_0) - I)\,\delta\| + \mathcal{O}(\|\delta\|^2)}{\|\delta\|}$$

For $\|\delta\|$ sufficiently small (i.e., within the linear regime where $\mathcal{O}(\|\delta\|^2)$ is negligible):

$$r(\delta) \approx \frac{\|(J_F(c_0) - I)\,\delta\|}{\|\delta\|}$$

The worst-case expansion rate over all perturbation directions is therefore controlled by:

$$\sup_{\|\delta\|=1} \|(J_F(c_0) - I)\,\delta\| = \|J_F(c_0) - I\|$$

the operator norm of $(J_F(c_0) - I)$.

## Stability Criterion

The trajectory remains within the truth bubble of radius $\epsilon$ provided the accumulated expansion does not exceed $\epsilon$. After $n$ steps, under the linear approximation:

$$\|\delta_n\| \lesssim \|\delta_0\| \cdot \prod_{i=0}^{n-1} \|J_F(c_i) - I\|$$

**Stable generation** ($\|\delta_n\| \to 0$) requires $\|J_F(c_i) - I\| < 1$ on average, i.e., the eigenvalues of $J_F$ remain close to 1 in a ball of radius less than 1 in the complex plane.

**Unstable generation** ($\|\delta_n\| \to \infty$) occurs when $\|J_F(c_i) - I\| > 1$ persistently then perturbations from the training manifold grow exponentially, and the trajectory exits the truth bubble.

## Inference-Time Control via Jacobian-Guided Token Selection

The preceding analysis reduces the problem of controlling hallucination to balancing two quantities at each token position:

1. **Certainty**: the logit probability $p(t_k | c)$ of each candidate token.
2. **Stability**: the expansion rate $\|J_F(c_k) - I\|$ induced by choosing token $t_k$, where $c_k = [c,\; t_k,\; 0]$.

Given the top-$k$ candidate tokens $\{t_1, \ldots, t_k\}$ ranked by probability, we compute the expansion rate for each candidate by perturbing the context $c_k = [c,\; t_k,\; 0]$ with $N$ random unit vectors $\{v_j\}_{j=1}^N$:

$$r_j^{(k)} = \frac{\|F(c_k + \epsilon\, v_j) - F(c_k)\|}{\epsilon}$$

$$\|J_F(c_k)\| \approx \max_j\; r_j^{(k)}$$

The selection rule is:

$$t^* = \arg\max_{t_k} \left[\log p(t_k | c) - \beta \cdot \|J_F(c_k)\| \right]$$

where $\beta > 0$ controls the tradeoff between probability and dynamical stability.

For each candidate, $1 + N$ forward passes are required (one to compute $F(c_k)$, $N$ for the perturbations). I haven't gotten the chance to test this yet but I hope to return with results at some point. 
