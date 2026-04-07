# 环境配置
所有步骤请参照[这份3DGS论文中给出的文件](./3DGS_README.md)

# 运行

```shell
SUBJECT=306

python my_train.py \
  -s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
  -m output/UNION10EMOEXP_${SUBJECT}_fed_600k \
  --num_clients 4 \
  --rounds 300 \
  --local_steps 2000 \
  --iid \
  --bind_to_mesh \
  --white_background \
  --eval
```

# 实验训练得到的数字人可视化
本论文由于实验条件限制（本人PC的GPU过于新，导致GPU、cuda以及各种代码之间有一些兼容性问题，没能成功在本地配置环境，所有程序的运行都使用了学校的服务器。），没能在论文中展示可视化结果的图片，因此下面给出查看实验可视化结果的命令，以便审稿人判断实验的真实性，本实验得到指标时对应保存的模型参数已经放置到对应路径中，直接运行应该可以看到结果。
```shell
python local_viewer.py --point_path media/my/point_cloud.ply
```

# 其他说明
本人主要的工作集中于my_train.py文件中，对于其他文件只是进行了小的更改