# D0.2 设计稿:触觉 env wrapper + 路点专家 + 采集脚本

> 版本:v1(2026-09-14)。上游文档:《VTLA_支路二设计_v1.md》(权重/归一化契约)、《VTLA_支路二_仿真验证计划_v1.md》(分诊与四臂)。
> 范围:LIBERO 酒架任务(put_the_wine_bottle_on_the_rack, libero_goal task 9)的
> ① 触觉 env wrapper ② 路点脚本专家 ③ LeRobot 采集脚本。
> 不含:训练侧 TouchEncoder 接入(D0.4)、真机 T-Piezo(D1 之后)。

---

## 1. 触觉通道:touch site 布局与 obs 契约

### 1.1 布局(参数化,默认 60 点)

| 项 | 设计 |
|---|---|
| 载体 | Panda 夹爪两指指面 geom(robosuite PandaGripper 的 finger collision geom) |
| site 布置 | 每指一条 2×15 网格(长条覆盖指面),双指共 **60 site** |
| 传感器 | MuJoCo `<touch>` 传感器逐 site 绑定,读法向合力(标量) |
| F/T | 腕部法兰处加 site + `<force>`+`<torque>` 传感器 → 6 维 |
| obs 键 | `tactile60: float[60]`(mN 级原始值)+ `wrist_ft: float[6]`(N / N·m) |

- **布局是参数**不是承诺:60 = 与真机 T-Piezo 点数一致;真实布局(几×几、单双指)确认后只改 site 排布表,obs 维度与下游编码器不动;
- MuJoCo touch 传感器语义:site 体积内接触的法向力之和——site 用小盒子贴进指面表面,略微凸出以捕获接触;
- **实现方式**:LIBERO 环境 由 BDDL → robosuite → MuJoCo XML 编译;在 XML 编译**前**做字符串/模型级注入(给 finger geom 所属 body 追加 site 与 sensor 定义),封装成 `tactile_patch.py` 单模块。**采集脚本与 RLinf 训练 rollout 共用同一 patch**——信号契约:demo 里的触觉和训练/推理看到的触觉逐位一致。

### 1.2 信号纪律

- 落盘存**原始值**,signed-log + z-score 在训练管线做(与真机设计一致);
- 采样率 = 环境控制频率(LIBERO/robosuite 默认),不做重采样——仿真信号契约从简;
- 零接触时读数为噪声地板(仿真无噪声,是精确零)——**这正是"机制筛"要的上界条件**,不做人为噪声注入。

---

## 2. 路点专家:酒架任务状态机

利用 MuJoCo 免费真值(瓶子/酒架位姿),OSC_POSE 增量控制器直指路点;成功判定用环境原生 `check_success()`(状态式:瓶子立在架上)。

```
S0 移动到预抓取位(瓶子位姿 + 上方 10cm 偏移)
S1 对齐抓取轴(夹爪开口垂直于瓶轴,水平逼近)
S2 闭环逼近瓶身(真值位置伺服,步进收敛)
S3 合爪(夹紧 N 步,夹持瓶子)
S4 抬升 10cm(直线)
S5 移动到酒架槽位上方(真值槽位 + 上方 8cm)
S6 缓慢下放(速度减半,到槽位放置位)
S7 松爪 → 撤退到初始高度
每步:check_success() → True 则立即结束(成功)
安全阀:任何状态超时(>80 步)判失败结束
```

**随机化**:每条 demo 从该任务的 init states 池抽取初始状态(官方机制);瓶子位姿随之变化 → 策略必须视觉定位,不能死记轨迹。可选增强:BDDL 容差内的瓶子位姿微抖。

**验收**:随机 init × 30 次专家自测,成功率 ≥ 90%;抓取段/放置段各至少 80% 的 episode 触觉信号有事件(接触尖峰)——不达标先调路点不进采集。

---

## 3. 采集脚本与 LeRobot 落盘

`scripts/collect_wine_rack.py`:

1. 循环 N 次:重置环境(随机 init)→ 跑状态机 → `check_success()` 过滤;
2. 每步记录:`agentview_image`(256²)、`wrist_image`(256²)、`robot0_eef_pos/quat/gripper`(9d 本体)、`action`(7d)、`tactile60`、`wrist_ft`;
3. 成功 episode 写 LeRobot v2.1 格式(HF_LEROBOT_HOME=/data_vtlax/datasets);失败 episode 丢弃(计数留档);
4. QC 硬门:全部成功;抽 5 条人工看视频;触觉事件抽查(抓取段与放置段各出现 >阈值的接触尖峰)。

**数量**:100 条(目标)/ 50 条(下限)。四臂共用同一份数据,分毫不差。

---

## 4. 评测侧的对应改动

- 评测 env 同样挂 `tactile_patch.py`(rollout 时触觉进 obs,但 bare 臂的模型不消费它——通道在,数据在,臂间只差模型结构);
- 评测输出目录按任务/臂分开(t9/t3 视频互覆盖的教训):`video_base_dir` 一律带 run 标识。

---

## 5. 实现文件清单

| 文件 | 内容 | 侧 |
|---|---|---|
| `rlinf/envs/sim/libero/tactile_patch.py` | XML 注入(60 site + touch 传感器 + 腕 F/T),obs 键挂接 | 容器/仓库 |
| `scripts/libero_expert.py` | 酒架路点状态机(真值伺服) | 仓库 |
| `scripts/collect_wine_rack.py` | 采集循环 + LeRobot 落盘 + QC | 仓库 |
| 验收记录 | 本文档追加"实测结果"节 | 仓库 |

## 6. 验收清单(全部通过才算 D0.2 完成)

- [ ] touch 传感器:自由空间读零;抓取时对应指面 site 出现尖峰;空间特异性(左指接触左指亮);
- [ ] 腕 F/T:加载重物/接触时数值合理变化;
- [ ] 专家:30 次随机 init 成功率 ≥90%;
- [ ] 触觉事件:抓取段/放置段尖峰覆盖率 ≥80%;
- [ ] 采集:100 条(或 50 条下限)全成功,LeRobot 格式被 norm stats 工具接受;
- [ ] bare 臂能以此数据微调(训练管线验证,D0-a 合并执行)。
