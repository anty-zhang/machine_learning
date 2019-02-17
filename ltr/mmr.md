# MMR(Maximal Marginal Relevance)

## 算法原理

$MMR=arg \max \limits_{D_i \in R-S } \left[\lambda (Sim_1(D_i, Q)) - (1 - \lambda) ((\max \limits_{D_j \in S} Sim_2(D_i, D_j))   \right]$


$D_i$ : Documents in the collection C

Q: Query,

R: Relevant documents in C,

S: Current result set

- Constructs the result set incrementally
- User-tunable diversity through λ parameter
- High λ = Higher accuracy
- Low λ = Higher diversity

以上公式中是针对搜索引擎中的查询与文档的关系

## 算法应用-文本摘要

将MMR转化为文本摘要的公式

$max[\lambda * score(i) - (1 - \lambda) * max[similarity(i,j)]]$

score(i): 计算句子的得分, similarity(i,j)句子i于j的相似度

左边的score计算的是句子的重要性分值，右边的计算的是句子与所有已经被选择成为摘要的句子之间的相似度最大值，注意这里的是负号，说明成为摘要的句子间的相似度越小越好。此处体现了MMR的算法原理，即均衡考虑了文章摘要的重要性和多样性。这种摘要提取方式与textrank不同，textrank只取全文的重要句子进行排序形成摘要，忽略了其多样性。

对于一篇文档，计算当前句子Q在全文中的相似度，MMR认为，对于相似度全文排名越高的表示为重要性越高。这里的相似度一般为余弦相似度，原来的论文就是cos。推荐介绍的两篇论文都有写到一些计算相似度的算法，可以研究研究。

实例见： machine_learning_project/ltr/mmr/mmr.py


## reference

https://github.com/fajri91/Text-Summarization-MMR