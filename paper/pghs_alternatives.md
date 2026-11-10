# Replacing PGHS with CFI (characteristic-function inversion)

Starting point (thesis, Methods chapter, PGHS): we want

$$p(y \mid x^o) = \mathbb{E}_{p(x^m \mid x^o)}\big[\sigma(z)\big], \qquad z = w^T x = w_o^T x^o + w_m^T x^m .$$

After the Pólya-Gamma trick this is

$$p(y \mid x^o) = \tfrac12\, \mathbb{E}_{\gamma \sim \mathrm{PG}(1,0)}\, \mathbb{E}_{p(x^m \mid x^o)}\Big[\exp\big(\tfrac12 z - \tfrac12 \gamma z^2\big)\Big].$$

PGHS then completes the square and applies Hubbard-Stratonovich (HS), which gives

$$f(u,\gamma) = e^{\frac{1}{8\gamma}}\; e^{\,iu\left(\frac{1}{2\gamma} - w_o^T x^o\right)}\; \phi_{x^m \mid x^o}(-u\, w_m), \qquad u \sim \mathcal{N}(0,\gamma).$$

## The problems with PGHS

**1. The inner integral amplifies the error.**
- For small $\gamma$, the factor $e^{1/(8\gamma)}$ is huge, and the wave $e^{iu/(2\gamma)}$ oscillates very fast. The true result is small, so huge oscillating numbers have to cancel almost exactly. In floating point they don't.
- This is why Gauss-PG has to keep its nodes away from $\gamma = 0$.
- Near $0$ the PG(1,0) density behaves like $\gamma^{-3/2} e^{-1/(8\gamma)}/\sqrt{2\pi}$, so $e^{1/(8\gamma)}$ exactly cancels its decay, and $\mathbb{E}_\gamma[e^{1/(8\gamma)}] = \infty$. As a result, PGHS-MC (sampling $\gamma$ and $u$) has **infinite variance**.

**2. The bells fail for a wide $p(z)$.**
- A sum of bells approximates $\sigma(z)$ well for negative $z$, but tends to $0$ for large positive $z$, while $\sigma(z) \to 1$.
- The sign flip only helps if $p(z \mid x^o)$ lies mostly on one side of $0$.
- With many missing features, $p(z \mid x^o)$ is wide and has mass on both sides, so this error remains even when the inner integral is computed exactly ("PGHS exact inner").
- This cannot be fixed within PG + HS: it is a property of the bells, not of the inner integral.

## The fix: no PG and no HS at all

Everything only depends on the single number $z$, and the PC already gives us the CF of $z$:

$$\phi_z(s) = \mathbb{E}\big[e^{isz} \mid x^o\big] = e^{is\, w_o^T x^o}\; \phi_{x^m \mid x^o}(s\, w_m).$$

So we only need a formula for $\mathbb{E}[\sigma(z)]$ in terms of $\phi_z$. There is an exact one.

**Step 1: the sigmoid is a CDF.** $\sigma$ is the CDF of the standard logistic distribution: $\sigma(z) = P(L < z)$ for $L \sim \mathrm{Logistic}(0,1)$. Taking $L$ independent of $z$,

$$\mathbb{E}[\sigma(z)] = P(L < z) = P(z - L > 0).$$

**Step 2: the CF of $z - L$.** The CF of $L$ is $\phi_L(s) = \frac{\pi s}{\sinh(\pi s)}$, which is real and even. So

$$\phi_{z-L}(s) = \phi_z(s)\, \frac{\pi s}{\sinh(\pi s)}.$$

**Step 3: Gil-Pelaez inversion.** For any continuous random variable $Y$, $P(Y > 0) = \tfrac12 + \tfrac1\pi \int_0^\infty \frac{\operatorname{Im}\phi_Y(s)}{s}\, ds$. With $Y = z - L$ this gives

$$\boxed{\;p(y \mid x^o) = \mathbb{E}[\sigma(z)] = \frac12 + \int_0^\infty \frac{\operatorname{Im}\, \phi_z(s)}{\sinh(\pi s)}\, ds\;}$$

- It is exact and a single one-dimensional integral.
- The weight $1/\sinh(\pi s)$ decays like $e^{-\pi s}$: the part beyond $s = 12$ is below $3 \cdot 10^{-17}$.
- At $s = 0$ the integrand stays finite: it tends to $\mathbb{E}[z \mid x^o]/\pi$.
- Check with a fixed $z$: $\operatorname{Im}\phi_z(s) = \sin(sz)$ and $\int_0^\infty \frac{\sin(sz)}{\sinh(\pi s)} ds = \tfrac12 \tanh(z/2)$, so the formula gives $\tfrac12 + \tfrac12\tanh(z/2) = \sigma(z)$.

**Intuition:** the sigmoid is a weighted sum of sine waves, $\sigma(z) = \tfrac12 + \int_0^\infty \frac{\sin(sz)}{\sinh(\pi s)} ds$. The expectation replaces every wave by its expectation under $p(z \mid x^o)$, which is $\operatorname{Im}\phi_z(s)$.

## Computing it

Use the trapezoid rule with step $h$ and $K$ frequencies $s_k = kh$:

$$p(y \mid x^o) \approx \frac12 + h\left(\frac{\mathbb{E}[z \mid x^o]}{2\pi} + \sum_{k=1}^{K} \frac{\operatorname{Im}\, \phi_z(kh)}{\sinh(\pi k h)}\right).$$

- The error drops exponentially in $K$.
- The sum is a Fourier series of $\sigma$ with period $2\pi/h$, so $h$ sets the range of $|z|$ that is covered correctly (about $|z| < \pi/h$).
- Choose $h$ per point from the mean $\mu_z$ and standard deviation $\sigma_z$ of $z$: $h = \min\big(\pi / (|\mu_z| + 6\sigma_z + 1),\; 12/K\big)$.
- For a PC, all frequencies are scalar multiples of the same direction $w_m$, so one batched circuit pass over a vector of $s_k$ gives all of them. $\mu_z$ and $\sigma_z$ come from one moment pass.

Compared to PGHS, there is no $\gamma$, no $u$, no Gauss-PG construction, no $e^{1/(8\gamma)}$ and no sign flip. It needs only $K$ evaluations of $\phi_z$ instead of $M \cdot N$, and only the CF, so it also works for heavy-tailed leaves. It is differentiable in $w$ and in the density's parameters, so PC and LR can be trained jointly.

## Result (GMM, 50% missing, RMSE, same budget of 1180 evaluations)

| features | MC | PGHS | PGHS exact inner | CFI |
|---|---|---|---|---|
| 5 | 9.5e-3 | 5.1e-5 | 5.0e-5 | 3.2e-8 |
| 50 | 5.8e-3 | 6.5e-8 | 4.8e-8 | 3.5e-8 |
| 500 | 8.5e-3 | 5.9e-7 | 4.4e-8 | 3.4e-8 |
| 1581 | 1.0e-2 | 2.6e-4 | 1.2e-4 | 3.9e-8 |
| 5000 | 8.2e-3 | 2.8e-2 | 1.4e-2 | 4.4e-8 |

CFI stays at the float32 floor for every number of features. With 10 features, 40 frequencies already gave the same accuracy as 1180.
