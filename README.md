## 代码说明

1. MainRun.py

   ```python
   # 有早停机制，earlyStopping，不指定epoch
   ```

   ```python
   # 可以固定随机种子 seed_flag=1，seed=124
   # or 不固定随机种子 seed_flag=0
   # 需要指定训练轮数 n_epoch，
   # 重复执行几次 n_replicates,
   # 数据训练前有另外的MinMaxScaler操作，适用于SCAD数据
   ```

   ```python
   # 可以固定随机种子 seed_flag=1，seed=124
   # or 不固定随机种子 seed_flag=0
   # 需要指定训练轮数 n_epoch，
   # 重复执行几次 n_replicates,
   # 数据训练前没有另外的MinMaxScaler操作，适用于scDEAL数据
   ```

   


