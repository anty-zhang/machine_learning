# -*- coding:utf-8 -*-
__author__ = 'anty'

ten = set(range(10))
lows = set([0,1,2,3,4])
odds = set([1,3,5,7,9])

print('ten is ',ten)
print('lows is ' ,lows)
print('odds is ',odds)

lows.add(9)
print('lows add 9 is ',lows)
diff = lows.difference(odds)   # equal set1 - set2
print('lows difference odds is:',diff)

inter = lows.intersection(odds)    #equal set1&set2
print('lows intersection odds is:',inter)

union = lows.union(odds)    #equal set1 | set2
print('lows union odds is:',union)

subset = lows.issubset(ten)   #equal set1 <= set2
print('lows subset odds is:',subset)
upset = lows.issuperset(odds)   #equal set1 >= set2
print('lows upset odds is:',upset)

syn_diff = lows.symmetric_difference(odds)   #equal set1 ^ set2
print('lows synmetric differ odds is:',syn_diff)

low_remove = lows.remove(2)
print('lows remove 2 low_remove is:',low_remove,'lows is :',lows)

lows.clear()
print('lows clear is:',lows)


