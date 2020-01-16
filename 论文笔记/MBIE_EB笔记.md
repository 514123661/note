# MBIE and MBIE-EB

这篇论文的作者是Littlman，做的一遍关于其在MDPs方面的exploration上的论文。

首先明确一个观点，如果一个学习算法是满足PAC理论的，那么这个学习算法则可以在有限的样本中达到学习的目的，**强学习的(strong learnable)**。

对于一个确定的MDP模型来说，提出模型探索模型的人来说，首先需要分析的就是这个算法的采样复杂度，前提是证明其采样复杂度是有限的，然后证明其采样复杂度的上限是关于$\mathbb{O}(\frac{1}{\delta},\frac{1}{\epsilon},\frac{1}{1-\gamma},|S|,|A|)$其中几个参数的多项式，就能保证该算法能够达到收敛的效果。这个地方可以直观的理解为需要采样的样本数量需要达到$\mathbb{O}(\frac{1}{\delta},\frac{1}{\epsilon},\frac{1}{1-\gamma},|S|,|A|)$，多的样本数量就行。只需要探索无数次，则在理论上是可以保证能获得最优解的。

## 文章中的算法和分析

作者实际上提出两种更新MBIE算法，并证明其是满足PAC-MDP的

第一种是比较复杂的MBIE，另一种则是比较简单的MBIE或者叫做MBIE with exploration Bonus 版本，作者都证明了他们是满足PAC-MDP的。

## MBIE

要明白这个算法需要搞明白以下几个算法。

### 关于奖励估计

明确一点。在强化学习中，agent是对MDP环境是无知的，在没有任何先验的情况下，agent的仅仅只会在做动作以后立即接受到来自环境的反馈，从而更新其$Q-value$。若采用无模型更新的。在各种状态下做出的动作的奖励应该采用如下公式更新：

$$\hat{R}(s,a) := \frac{1}{n(s,a)}\sum_{i=1}^{n(s,a)}r[i]$$

其中$n(s,a)$是统计agent在状态$s$，采取了$a$动作的次数。$r[i]$表示第$i$次所获得的奖励。

作者换了一条思路（应该有前人做过，但是我在这就把这个思路当成是作者独有的）：将$r[i]$视为是从分布$R(s,a)$采样得到的，那么可以认为真正的$R(s,a)$应该是在区间$CI(R)$中。其中$CI(R)$定义为：

$CI(R):=(\hat{R}(s,a)-\epsilon^R_{n(s,a)},\hat{R}(s,a)+\epsilon^R_{n(s,a)})$

$\epsilon^R_{n(s,a)}:=\sqrt{\frac{ln(2/\delta_R)}{2n(s,a)}}$

这个置信区间是通过Hoeffding不等式计算出来的。同时也能得出上面这个式子是满足PAC框架的。

### 关于转移概率的估计

先给出众所周知的预估概率的方法。假设$\hat{T}(s'|s,a):=\frac{n(s,a,s)}{n(s,a)}$，显然$\sum_{s'|s,a}T(\cdot|s,a)=1$就可以当做状态转移概率了。同样，依据上节的思路，可以认为这个真实值处于估计值表示的某一个区间中，论文是直接给出了

$CI(T):=\{{\tilde{T}(s,a)\in P_s|\lVert\tilde{T}(s,a)-\hat{T}(s,a)\rVert_1\leq}\epsilon^T_{n(s,a)}\}$

$\epsilon^T_{n(s,a)}=\sqrt\frac{2[ln(2^{|S|}-2)-ln(\delta_T)]}{m}$

同样可以证明上述的估计是满足PAC的

### 更新公式

作者给出了使用上述两个估计量的更新公式:

$Q'(s,a):=\max\limits_{\tilde{R}(s,a)\in CI(R)}\tilde{R}(s,a)+\max\limits_{\tilde{T}(s,a)\in CI(T)}\gamma\sum\limits_{s'}\tilde{T}(s'|s,a)\max\limits_{a'}Q(s',a')$

几点说明：

1.该算法可以才每个常数时间更新。

2.其中的m就表示这个常数时间。

3.计算式子中的第二项需要一定的算法。littman给出了算法和证明，写到伪代码里。

正文并没有给出算法的代码，让人觉得很费解，我会在稍后的内容中给出算法的伪代码。然后给Q出算法的时间复杂度分析

#### 第二项公式说明

令

$M_v=\sum\limits_{s'}\tilde{T}(s'|s,a)V(s')$

这个式子为一个优化问题

$\max\limits_{(s,a)} M_v$

$subject\ to$

$\lVert\tilde{T}(\cdot|s,a) - \hat{T}(\cdot|s,a)\rVert\leq \epsilon$

解决优化问题。

论文中给出了一个方法，这里我先做文字说明，然后给出伪代码，再给出Python代码。

在某一个循环中:



1. 修改R(s,a)的值 by  $R(s,a)\leftarrow R(s,a)+\epsilon^R_{n(s,a)}$

2. 排序$Q_{max}[t+1]$

3. 并找到其中最大值对应的状态设为$s^*$

4. $T(s^*|s,a)\leftarrow T(s^*|s,a)+\frac{\epsilon^T_{n(s,a)}}{2}$

5. if $T(s^*|s,a)\geq1$

   ​	$T(s^*|s,a)\leftarrow1$	

   ​    其余$T(s'|s,a)\leftarrow0\ if \ s'\ne s^*$

6. while $\sum\limits_{s'}T(s'|s,a)>1$:

   ​	6.1找到最小的非0且$Q_{max}[t+1]$所对应的状态$s\_$	

   ​    6.2更新$T(s\_|s,a)\leftarrow max(0,T(s\_|s,a)+(1-\sum\limits_{s'}T(s'|s,a)))$

7. 更新$Q(s,a)$ by $Q(s,a)\leftarrow R(s,a)+\gamma\sum\limits_{s'}T(s'|s,a)Q_{max}[t+1](s')$

8. 修正整个Q表，以及当前阶段的$Q_{max}[t+1]$

   

   分析：在采样阶段

   根据递归法则更新的算法

   $T(|S|) = T(|S|-1) + |S|$可以推出$T(|S|) = |S|ln|S|$

   即在更新Q值时算法复杂度为$O(|S|ln|S|)$

对于上述式子的value iteration 的时间复杂度分析：

首先，我们更新算法时，需要便利所有的(s,a)，故总的时间复杂度应为$|S||A|*O(一次迭代的时间复杂度)$，一次循环的时间复杂度可从伪代码计算：

步骤2的时间复杂度为$O(|S|ln|S|)$ 快速排序和归并都可以做到

步骤6的时间复杂度为$O(|S|)$

步骤8采用二分查找法修改当前(s,a)所对应的Q值时间复杂度为$O(ln(|S||A|))$,以及获得当前timestep 各个状态所对应的最大Q值，其时间复杂度为$O(ln(|A|))$

故得到时间复杂度为$O(|S|ln|S|+ln(|S||A|)+|S|+ln(|A|)))$，根据时间复杂度的性质可得最终时间复杂度为

$O\bigg(|S||A|N\big(S|ln|S|+ln(|S||A|)\big)\bigg)$

python 代码如下

```python
def compute_qVals_MBIEVI(self, R, P, R_confident, P_confident):
        '''
        通过MBIE计算Q值表
        Args:
            R - R[s,a] ： 奖励平均值 数据类型是浮点型
            P - P[s,a] ： 状态转移频率 数据类型是 |S|维向量
            R_confident - R_confident[s,a] = R的置信度
            P_confident - P_confident[s,a] = P的置信度

        Returns:
            qVals - qVals[state, timestep] 是timestep的Q值
            qMax - qMax[timestep] 是当前timestep的最大Q值
        '''

    qVals = {}
    qMax = {}
    qMax[self.epLen] = np.zeros(self.nState)
    for i in range(self.epLen):
        j = self.epLen - i - 1
        qMax[j] = np.zeros(self.nState)
        for s in range(self.nState):
            qVals[s, j] = np.zeros(self.nAction)
            for a in range(self.nAction):
                rOpt = R[s, a] + R_confident[s, a]
                pInd = np.argsort(qMax[j + 1]) #排序对应步骤2
                pOpt = P[s, a]
                #求最大值
                if pOpt[np.where[pInd==0]] + P_confident[s, a] * 0.5 > 1:
                    pOpt = np.zeros(self.nState)
                    pOpt[np.where[pInd==(self.nState-1)]] = 1
                else:
                    pOpt[np.where[pInd==(self.nState-1)]] += P_confident[s, a] * 0.5
                    #步骤6
                    while np.sum(pOpt) > 1:
                        worst = pInd[np.where(pInd==sLoop)]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1
                    #步骤7&8
                    qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])
				#步骤8
                qMax[j][s] = np.max(qVals[s, j])
        return qVals, qMax
```





## MBIE-EB

该论文的作者在研究PAC-MDP[^1]中，则认为Q值可以做如下估计方式：

$$\hat{Q}(s,a)  =R(s,a)+\gamma\sum_{s'}T(s'|s,a)max_{a'}Q(s',a')+\frac{\beta}{\sqrt{n(s,a)}}$$

​	然后采取greedy策略来选取下一个动作，即：

$a_{t+1} = \arg \max _{a\in \mathbb{A}} \hat{Q}(s,a)$

这个时间复杂度就十分简单了，不做分析



### 采样复杂度以及完整性分析

### 采样复杂度

作者先给出了MBIE的采样复杂度结论：

**Theorem 1.**假设$\epsilon,\delta\in(0,1)$，任意一个MDP($M = (S,A,T,R,\gamma)$)来说，存在输入$\delta_R=\delta_T=\frac{\delta}{2\left|S\right|\left|A\right|m}，\& \ m(\frac{1}{\epsilon},\frac{1}{\delta})=O(\frac{|S|}{\epsilon^2(1-\gamma)^4}+\frac{1}{\epsilon^2(1-\gamma)^4}\ln\frac{|S||A|}{\epsilon(1-\gamma)\delta})$，那么当agent采用MBIE去探索整个MDP的时候，当timestep $t = O(\frac{|S||A|}{\epsilon^3(1-\gamma)^6}(|S|+\ln\frac{|S||A|}{\epsilon(1-\gamma)\delta})\ln\frac{1}{\delta}\ln\frac{1}{\epsilon(1-\gamma)})$，那么我们就可以以至少概率$1-\delta$的认为，$V^{A_t}_M(s_t) \geq V^*_M(s_t)-\epsilon$。

只要证明了上述理论是正确的，则根据PAC的定义，就表示MBIE算法是PAC-MDP的。

总而言之，这个算法是个强学习性的。可见两者差的不多。

### 为证明该理论的所有lemma解析

这个理论看起来很优美，但是证明起来确实复杂，作者为证明该理论提出了4个lemma，（文中给出了6个，其中2个是别的论文上的）。这些个引理还是值得一看，值得分析的，里面阐述了很多这个算法为什么可行的一些道理，以及其对未知环境的探索是如何做到的。

##### lemma1.

假设有两个相似的MDP：$M_1 = (S,A,T_1,R_1,\gamma),M_2=(S,A,T_2,R_2,\gamma)$。所谓的相似性为，两者有相同的状态和动作空间。并且$R_1,R_2\in(0,+\infin)$。如果两者的奖励函数，以及状态转移概率有如下关系：

$$|R_1(s,a)-R_2(s,a)|\leq \alpha$$

$\lVert T_1(s,a,\cdot)-T_2(s,a,\cdot)\rVert_1\leq\beta$

那么对于一个确定的策略$\pi$，则有

$|Q_1^\pi(s,a)-Q_2^\pi(s,a)|\leq\frac{\alpha+\gamma R_{max}\beta}{(1-\gamma)^2}$

这个作者并没有给出证明。简要说明一下这个引理在后面证明中的价值：这个引理在后来的行文证明中，可以假设其中一个MDP是最优的MDP，或者说是算法想要无限逼近的最优值。那么我们需要做的事情是，把算法的策略上述最后的公式中，证明两个Q值的距离或者说差值，是在高概率情况下是比较小的。从而推到出算法的采样复杂度实在一个多项式范围内的。里面的变量是$\alpha,\beta$是可调的，我们接下来就能在这两个参数上做文章了。

##### lemma2.

在lemma1的基础上，lemma2加了些限制：$R_{max}\geq 1$ 。假定lemma1的两个限制条件是成立的：$|R_1(s,a)-R_2(s,a)|\leq \alpha，\lVert T_1(s,a,\cdot)-T_2(s,a,\cdot)\rVert_1\leq\beta$，这样就能得出一个结论：存在一个常数$C$，使得对于任意的$\epsilon\in(0,\frac{R_{max}}{1-\gamma}]$，在采用一个固定策略$\pi$时，如果$\alpha=\beta=C(\frac{\epsilon(1-\gamma)^2}{R_{max}})$两个MDP的$Q$值存在以下关系

$|Q_1^\pi(s,a)-Q^\pi_2(s,a)|\leq\epsilon$

论文中有不太严谨的证明。

我证明出来是：

当$C=\frac{1}{1+\gamma}$ 上述式子都成立，有因为$\gamma\in(0,1]$的，所以$C\in[\frac{1}{2},1)$。所以Littlman在论文中选取$C=\frac{1}{2}$是合理的。

所以证明两个MDP在某些情况下（上述的一些情况），两者的$Q$误差是在可控范围之内的。其实简单来说，最差的情况就是一个取到了最好的情况$V^\pi(s)=\frac{R_{max}}{1-\gamma}$,一个取到了最差的情况，就是0，所以两者最大的差距就很显而易见了。但是要求两个MDP模型啊，要足够的相似，否则两者的差距就会很大。

##### lemma3

这个lemma讲$(s,a)$分成了两个部分，一个部分是充分探索过的$K$，$A_M$是agent按照策略$\pi$ 去了之前没有探索过的$(s,a)$的事件，$\mathbf{Pr}$指的就是发生这件事情的概率，等算法运行了$H$步之后，真实的价值函数$V^\pi_M(s_1,H)\geq V^\pi_{M'}(s_1,H)-\frac{1}{1-\gamma}\mathbf{Pr}(A_M)$。其中$M'$是探索过得$(s,a)$所组成的MDP，$M$是整个的MDP。

这个lemma仔细研读就会发现，$V^\pi_M(s_1,H)，V^\pi_{M'}(s_1,H)$两者error的上界是$\frac{1}{1-\gamma}\mathbf{Pr}(A_M)$，当未探索的事情发生的越多，那么这个概率值就会很大，我们可以不断的通过探索来减小这个上界。~~换句话说，算法是总想探索下新的$(s,a)$来使得自己的$V(s_1)$变大一些。~~这个式子应该这么看：

$\frac{1}{1-\gamma} \mathbf{Pr}(A_M)\geq V^\pi_M(s_1,H)-V^\pi_{M'}(s_1,H)$

意味着，只要两个value值差别太大，那么$A_M$事件，即，采样未知(*unknown*)的$(s,a)$发生的概率就会很大。

之后在证明其算法探索能力的时候，这个lemma起到了很足的铺垫作用。

##### lemma4

没什么好说的，无非是当程序运行到一定程度了，两者的误差就会小一点了。

如果$H\geq \frac{1}{1-\gamma}\ln\frac{1}{\epsilon(1-\gamma)}$，那么$|V^\pi(s,H)-V^\pi(s)|\leq\epsilon$

之后的两个引理在

##### lemma5,lemma6

这两个lemma是导出算法的采样复杂度的关键引理。

首先谈lemma5:

这个引理提出，假设$\delta_R = \delta_T=\frac{\delta}{2|S||A|m}$，存在一个上界$m=O(\frac{|S|}{\tau^2}+\frac{1}{\tau^2}\ln\frac{|S||A|}{\tau\delta})$，只要所有的$(s,a)$被采样了至少m次以上的时候，那么就能保证MBIE算法评估的R和T与实际的差距是不大的。数学描述是：

$$|\tilde{R}(s,a)-R(s,a)|\leq\tau$$

$\lVert \tilde{T}(s,a,\cdot)-T(s,a,\cdot)\rVert_1\leq\tau$

这个是显然的，将两个假设回带到和对R和T的评估公式中，可以推到出来

阐述了一个什么问题呢，就是当采用MBIE算法的时候，只要采样数量足够，满足了那个上界之后，我们就能保证R 和 T 的评估是一定程度上是准确的。

然后lemma6

一旦满足了上述的bound之后 对于所有的Q值，在迭代过程中，都会满足

$\tilde{Q}(s,a)\geq Q^*(s,a)$。



关于采样复杂度的证明。

暂时不想赘述这些

首先证明确实有点意思，我在这里想要阐述的事实就是，作者用了一个思路去证明这个采样复杂度是比较小的。就是使用了仿真理论，简单来说就是，算法(即agent)，在一个MDP里面晃悠的时候，他自己根据自己的model(model base)，使用样本，来评估一个MDP，倘若其评估的MDP($\tilde{M}$) 与真实的MDP足够相似的话，由这个算法得出的决策结果，就有理由相信是足够接近最优值的。那么采样复杂度的影响因素就来自了两个，一个是对R的评估，一个是对T的评估，采样得到的样本量假设到达了一定程度之后，两个评估将会有一定概率是接近真实的，那么算法做出的觉得，也就一定的概率接近最优值了。所以，作者花了很多时间在lemma1-6上，而采样复杂度的证明就很少了。不过，里面有个耐人寻味的证明：

针对与**lemma4**，以及后面的证明(隐式探索)：

对于一个$Pr(A_M)\geq\epsilon_1(1-\gamma)$的时候，也就是说，agent会以至少$\epsilon_1(1-\gamma)$的概率进行探索，根据Hoeffding不等式，这个探索在执行了$O(\frac{m|S||A|}{\epsilon_1(1-\gamma)})$timestep之后，有$1-\frac{\delta}{2}$的概率相信所有$(s,a)$都会被知道。探索一段时间之后，$Pr(A_M)$会变小，接着转成利用阶段

此时$Pr(A_M)<\epsilon_1(1-\gamma)$

此时可以此时推导的出来的$V^{A_t}_M \geq V_M^{*}(s_t)-\epsilon$ 保证算法在至少为$1-\delta$的概率$A_t$的有$\epsilon-optimal$

[^1]: PAC-MDP是这其算法的采样复杂度是关于$(|S|,|A|,\frac{1}{\epsilon},\frac{1}{\delta},\frac{1}{1-\gamma}) $的多项式小于某个数的该概率之多为$1-\delta$

