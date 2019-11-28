# 高斯分布的后验分布

在看代码的时候遇到了，由于作者并没有给出过多的注释，让人觉得费解，所以现在做一些部分解释。**其中这是假设方差已知，预估均值。如果方差未知，均值已知，则是需要卡方分布作为先验，如果两者均为未知，则是gamma 分布作为先验**

## 1.1 方差已知，预测均值

高斯分布是一种重要的共轭分布，共轭分布指的是这个分布的先验分布以及后验分布属于同一种分布。数学表达为：

$$P(\mu|\mu_0,\sigma_0) \sim \mathbf{N}(\mu_0,\sigma_0)$$

$$P(\mathbf{X}|\mu) \sim N(\mu,\sigma)$$

后验：

$P(\mu|X)\propto P(\mu|\mu_0,\sigma_0)P(\mathbf{X}|\mu)$

展开：

$P(\mathbf{X}|\mu)=\frac{1}{\sigma\sqrt{2\pi}}\exp\bigg(-\frac{(x-\mu)^2}{2\sigma^2}\bigg)$

$P(\mu|\mu_0,\sigma_0) = \frac{1}{\sigma_0\sqrt{2\pi}}\exp\bigg(-\frac{(\mu-\mu_0)^2}{2\sigma_0}\bigg)$

$P(\mathbf{X}|\mu)P(\mu|\mu_0,\sigma_0)\propto \exp\Bigg[-\Big[\frac{(x-\mu)^2}{2\sigma^2}+\frac{(\mu-\mu_0)^2}{2\sigma_0}\Big]\Bigg]$

由理论已知，当已知方差，估计均值时，均值的后验分布仍旧服从高斯分布。

取指数部分推导

$(\frac{1}{2\sigma^2}+\frac{1}{2\sigma_0^2})\mu^2-(\frac{x}{\sigma^2}+\frac{\mu_0}{\sigma_0})\mu+(\frac{x^2}{2\sigma}+\frac{\mu^2}{2\sigma_0}) $

$\propto$

$(\frac{1}{2\sigma^2}+\frac{1}{2\sigma_0^2})\mu^2-(\frac{x}{\sigma^2}+\frac{\mu_0}{\sigma_0})\mu$

参考高斯分布

假设需要求的均值方差是$\mu_*,\sigma_*$

所以$\frac{(\mu-\mu_*)^2}{2\sigma_*^2}=\frac{\mu^2-2\mu\mu_*+\mu_*}{2\sigma_*}$

根据一一对应原则可以算出：

$\frac{1}{\sigma_*^2} = \frac{1}{\sigma}+\frac{1}{\sigma_0}$

$\mu_* = \frac{\sigma_0^2}{\sigma^2+\sigma_0^2}x + \frac{\sigma^2}{\sigma^2+\sigma_0}\mu$

若为X服从多元高斯分布则可推广为

$\frac{1}{\sigma_*^2} = \frac{1}{\sigma}+\frac{N}{\sigma_0}$

$\mu_* = \frac{N\sigma_0^2}{\sigma^2+N\sigma_0^2}x_N + \frac{\sigma^2}{\sigma^2+N\sigma_0}\mu$

