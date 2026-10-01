import numpy as np
from scipy import stats

data = [495, 498, 492, 497, 494, 496]

# H0 : mu = 500
# H1 : mu != 500

mu_0 = 500    # claimed average
alpha = 0.05

n = len(data)
mean = np.mean(data)
std = np.std(data, ddof=1)
t_stat_manual= (mean -mu_0) / (std / np.sqrt(n))

print(f"Sample Mean   : {mean:.2f}")
print(f"Sample std dev: {std:.2f}")
print(f"t-statistic   : {t_stat_manual:.2f}")



t_stat, p_value = stats.ttest_1samp(data, popmean=mu_0)
# print(t_stat, p_value)

df = n-1
t_critical = stats.t.ppf(1-alpha/2, df)

print(f"\nDegrees of Freedom:  {df}")
print(f"Critical value:        {t_critical:.3f}")
print(f"p-value:               {p_value:.4f}")


# Decision:

if p_value <= alpha:
    print("\nResult: Reject H0 -> average weight is 500gm")
else:
    print("\nResult: Fail to Reject H0 -> not enough evidence")
