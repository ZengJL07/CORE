分析：
GEPA的帕累托前沿维护鼓励prompt向包含更多前沿的方向拓展，但是过于低效。（就像是树搜索，为了探索到深度为N的很优prompt，探索近似于O(n)量级）
我认为这是因为GEPA的这个操作同时在实现存档回退（拒绝错误采样，软性降低低分prompt概率）/ 鼓励模型探索有潜力的题目（极少的prompt能完成）（只有在sample出来的题目分数不是完美时才会更新prompt。在前沿且很少prompt完成的题目会极大的增加对应prompt的采样概率，进而提升采样此题目的概率）。

动机：
鼓励模型探索有潜力的题目应该从题目的角度去操作而不是从prompt的角度

我们将其解耦，分为单独的拒绝采样和采样加权
我们为每一个样本维护一个分数。我们利用样本分数控制采样

1. 对于评测数1, 2, 3, ..., B：
  - 分裂 N 个 分支 prompt_candidate_i = chosen_prompt
  - 对于分支 i = 1, 2, 3, ..., N
    - 从训练集中sample train_batch_num 个样本作为小训练集 train_batch。
    - 利用GEPA相同的改进算法获得 prompt_candidate_i = evolve(prompt_candidate_i)
    - 存储在这些题目上的记录，记录正确性