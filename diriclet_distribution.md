# Diriclet分布

## 疑问来源

iosband的代码

## 原理

### 概率密度函数

存在一堆参数$\alpha_1,...,\alpha_K>0$ 使得

$$f(x_1,...x_K;\alpha_1,...,\alpha_K) = \frac{1}{\mathbf{B}(\mathbf{\alpha})}  \Pi^K_{i=1}x_i^{\alpha_i-1}$$

其中$\sum_{i=1}^Kx_i=1$ 并且$x_i\ge 0,x\in[1,K]$

$\mathbf{B}(\mathbf{\alpha}) = \frac{\Pi_{i=1}^K\Gamma(\alpha_i)}{\Gamma(\sum_{i=1}^K(\alpha_i))}$

一些关注点，首先伽马函数，不能叫做分布。这个分布用的比较多的地方大多是作为一些函数的先验分布。其边缘概率密度是beta分布，还是挺好用的。对其进行采样能得到一些小于1的，由于其$\sum_{i=1}^Kx_i=1$是可以产生一些离散的PMF的

### 一些性质

假设$X=(X_1,...,X_K)\sim Dir(\mathbf{\alpha}) $ 

根据定义则有

$X_K = 1-\sum_{i=1}^{K-1}X_i$

令

$\alpha_0=\sum_{i=1}^K\alpha_i$

则有：

$E[X_i] = \frac{\alpha_i}{\alpha_0}$

$Var[X_i] = \frac{\alpha_i(\alpha_i(\alpha_0-\alpha_i))}{\alpha_o^2(\alpha_0+1)}$

$Cov[X_i,X_j]=-\frac{\alpha_i\alpha_j}{\alpha_0^2(\alpha_0+1)}$

显然，数学期望和方差都与其参数有关~~废话~~

### 在作为先验分布的时候

在做贝叶斯分析的时候，其会被当做一个先验分布。一般来说，初始的参数设置一般为相同

$\mathbf{\alpha}=(\alpha_1,...,\alpha_K) = 某一个常数C$

$p|\alpha = (p_1,...p_K)=Dir(K,\mathbf{\alpha})$

$\mathbb{X}|p=(\mathbf{{x}_1},...,\mathbf{x_K})\sim Cat(K,p)$

所以可以得出

$\mathbf{c} = (c_1,...c_K)$K代表类别，或者某些参数吧

$\mathbf{p}|\mathbb{X},\alpha\sim Dir(K,c+\alpha)=Dir(K,c_1+\alpha_1,...,c_K+\alpha_K)$

### 举个例子来说

假设一帮数据服从cat分布，然后，我采集了部分数据，那么我就可以更新我的模型，（其中，我得假设，我得到的数据必须是$[0,1]$），这样我就可以单纯的通过加假发来更新我的数据



