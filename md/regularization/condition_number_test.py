# -*- coding: utf-8 -*-

import numpy as np


l1 = [
	[1.0, 2.0],
	[2.0, 3.0]
]
m1 = np.mat(l1)
print "m1 condition number=", np.linalg.cond(m1)


l2 = [
	[1.001, 2.001],
	[2.001, 3.001]
]
m2 = np.mat(l2)

print "m2 condition number=", np.linalg.cond(m2)

l3 = [
	[1, 2],
	[2, 3.999]
]
m3 = np.mat(l3)
print "m3 condition number=", np.linalg.cond(m3)

l4 = [
	[1.001, 2.001],
	[2.001, 3.998]
]
m4 = np.mat(l4)
print "m4 condition number=", np.linalg.cond(m4)

# m1 condition number= 17.94427191
# m2 condition number= 17.9603257225
# m3 condition number= 24992.0009601
# m4 condition number= 12478.2859907

# => m1,m2 well-conditioned
# => m3,m4 ill-conditioned
