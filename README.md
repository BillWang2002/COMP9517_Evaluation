# COMP9517_Evaluation：运行指令与结果

A1 = Faster R-CNN，B0 = YOLOv8s，FUSION = B0 + 选择性补入A1预测。**默认最终方法：B0，工作阈值0.4。**

从外层Project进入：`cd COMP9517_Evaluation`，然后运行以下命令。数据位于`data/beach_bird/`，权重为`weights/A/A1.pth`、`weights/B/B0.pt`；参数在`bird_eval/settings.py`。输出目录或图片若已存在，请使用新的名称；不会覆盖已有结果。

## 1. 环境与文件检查

已有环境：

```bash
source .venv/bin/activate
python run.py check
```

返回：val **291张**、test **240张**，以及两份权重的SHA-256；文件或权重不符时报错，不运行模型。

首次安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

本机使用M4/MPS、Python 3.12、FP32；换机器请重建`.venv`。无MPS时可改`--device cpu`，耗时须分开比较。查看全部参数：`python run.py --help`。

### 从 GitHub 获取

```bash
git clone https://github.com/BillWang2002/COMP9517_Evaluation.git
cd COMP9517_Evaluation
```

私有仓库只包含源码、测试、本文、依赖及`data/splits.csv`。克隆后，按上面的首次安装步骤建立环境，并另外放入以下文件再运行`check`：

| 本地路径 | 需要准备的内容 |
|---|---|
| `data/beach_bird/images/{val,test}/` | 官方val 291张、test 240张图片 |
| `data/beach_bird/labels/{val,test}/` | 同名YOLO标签，保留空标签文件 |
| `weights/A/A1.pth`、`weights/B/B0.pt` | 组员提供的两份固定权重；哈希由`check`核对 |

图片、标签、权重、运行输出和`.venv`均被Git忽略，不随源码上传。数据来源见文末；权重通过组内渠道另行提供。

## 2. 完整验证集评估

```bash
python run.py evaluate --models A1 B0 FUSION --split val --device mps --output outputs/val
```

返回：各系统运行进度，随后打印指标表。生成：

- `A1_val.json`、`B0_val.json`、`FUSION_val.json`：完整预测缓存。
- `A1_metrics.json`、`B0_metrics.json`、`FUSION_metrics.json`：详细指标。
- `comparison.csv`：AP、P/R/F1、工作阈值和TP/FP/FN对照。

**2026-11-13完整291图、1,843 GT实测：**

| 系统 | AP50 | AP50:95 | 工作阈值 | 工作点P | 工作点R | 工作点F1 |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 79.34% | 40.56% | 0.75 | 87.77% | 74.77% | 80.75% |
| B0 | **87.87%** | **49.28%** | 0.40 | **89.03%** | 82.80% | **85.80%** |
| FUSION | 87.69% | 49.19% | 0.40 | 88.36% | **83.18%** | 85.69% |

当前`outputs/val`完整验证单扫描耗时（均值/中位数/P95，ms/图）：A1 **160.707/158.694/173.232**，B0 **21.086/20.789/23.260**，FUSION **182.750/179.923/204.316**。这是一次实测，不是每次运行的固定返回值；与下面重复计时分开。

只比较A/B时用`--models A1 B0`。五图快速检查：

```bash
python run.py evaluate --models A1 B0 FUSION --split val --limit 5 --allow-partial --device mps --output outputs/smoke
```

返回相同类型文件，但标记`partial=true`，不作为正式性能结果。

## 3. 仅预测／从缓存复评

仅预测：

```bash
python run.py predict --model B0 --split val --device mps --output outputs/cache_b
```

生成`outputs/cache_b/B0_val.json`，打印进度和均值/中位数/P95耗时，不计算指标。将`B0`改为`A1`或`FUSION`可运行对应系统，并使用新的输出目录。

使用第2节缓存复评，不重新推理：

```bash
python run.py evaluate --models A1 B0 FUSION --split val --cache-dir outputs/val --output outputs/val_metrics
```

生成三个`*_metrics.json`与`comparison.csv`，指标应与第2节一致。

## 4. 后处理消融

先完成第2节，或准备完整A/B验证缓存：

```bash
python run.py postprocess --a-cache outputs/val/A1_val.json --b-cache outputs/val/B0_val.json --output outputs/postprocess
```

生成`FUSION_val.json`、`B0_metrics.json`、`FUSION_metrics.json`和`comparison.csv`；打印B0与FUSION指标。

固定方法：A1分数≥0.9、预测框面积≥96²；拒绝与B0支持框（分数≥0.1）IoU≥0.5的候选；新增A1框以IoU≥0.5去重，分数乘0.75，联合保留最多300框。筛选不使用GT。

实测AP50:95 **49.28% → 49.19%**，工作点F1 **85.80% → 85.69%**；增加7个TP和14个FP，**融合未提升整体表现，默认仍用B0**。B0采用的0.4阈值使验证F1由共同0.5阈值下的85.05%提升至85.80%，不改变完整缓存AP。

## 5. 重复计时

```bash
python run.py benchmark --models A1 B0 FUSION --rounds 4 --device mps --output outputs/timing
```

生成`per_run.csv`、`per_image.csv`、`summary.csv`、`environment.json`，打印各轮均值及最终汇总。

本轮完整实测：每系统4×291次，共12轮运行、3,492次测量；预热不计入。结果如下：

| 系统 | 四轮均值平均ms/图 | 轮均值范围ms/图 |
|---|---:|---:|
| A1 | 200.572 | 153.676–231.099 |
| B0 | 24.235 | 20.629–28.506 |
| FUSION | 219.794 | 181.054–236.714 |

奇偶轮反转顺序；每次预热5次，4线程、FP32、640、batch=1，设备同步。计时包含输入转换、传输、网络、内部后处理、CPU输出及融合，排除解码、模型加载、预热、指标与画图。后台/热/电源未控制，耗时会波动；不与第2节单次扫描混合平均。

五图快速检查也可使用：

```bash
python run.py benchmark --models A1 B0 FUSION --rounds 2 --limit 5 --device mps --output outputs/timing_smoke
```

快速检查标记`partial_val=true`，不作为完整验证集速度结果。

## 6. 单图演示

```bash
python run.py demo --image data/beach_bird/images/val/beach_bird_1408.jpg --models A1 B0 FUSION --device mps --output outputs/demo_1408.png
```

生成`outputs/demo_1408.png`四联图：原图、A1、B0、FUSION；打印输入路径、各系统预测数及单次耗时。该样例实测显示 **A1=5、B0=3、FUSION=4个预测框**，并非人工确认的真实鸟数。融合面板橙框来自B0，蓝框来自补入A1。图中注明输入文件名，框标签保留四位置信度；该数值不是IoU或准确率。此图为A1占优的验证样例，仅展示补充行为，不代表整体排名。

可替换图片路径或只用`--models B0`；`--score`仅改变展示阈值。单图耗时不代替重复计时结论。

## 7. 正式test复现

仅在需要复现时执行，不用于调参：

```bash
python run.py evaluate --models A1 B0 --split test --allow-test --device mps --output outputs/test_reproduction
```

生成A1/B0的test预测缓存、详细指标和`comparison.csv`。必须显式`--allow-test`，不允许部分test或FUSION。

**2026-11-12原正式test成绩，240图/1,518 GT；本轮更名后以固定参数完整复现，AP及工作点P/R/F1与原成绩一致：**

| 系统 | AP50 | AP50:95 | 工作阈值 | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 77.68% | 39.44% | 0.75 | 86.19% | 71.54% | 78.19% |
| B0 | **90.23%** | **50.22%** | 0.40 | **89.42%** | **84.58%** | **86.93%** |

原test单扫描均值：A1 **290.041**、B0 **33.663 ms/图**。这是原正式测试的历史耗时；本轮复现未覆盖这些计时记录，也未依据test调整参数。

## 8. 代码测试与指标说明

```bash
python -m unittest discover -s tests -v
```

已测返回：**Ran 14 tests … OK**。只做数值与接口测试，不运行检测器、不解码test。

- 主AP使用COCO maxDets=100，补充300；保留全部缓存分数，不按工作阈值预筛。CSV指标为0–1，本文为百分数。
- 共同诊断为score=0.5、IoU=0.5；工作点P/R/F1按全体TP/FP/FN汇总。目标检测TN不定义；空标签不保证无鸟。
- A1/B0输出分数下限为0.05/0.001，内部NMS IoU为0.5/0.7，最多300框。训练与预处理不同，此处比较实际系统。
- 本目录只含评估，不含训练。提交源码时排除环境、权重、图片/标签与运行输出；全组训练代码保留在外层交接包。

## 来源与许可

A1权重、多尺度训练与数据/模型/指标核心来自2号交接包，原A0基线来自1号/丁同学；B0来自B线。第5号负责统一评估、计时、选择性后处理和整合演示。使用过AI编码辅助，应按课程规定披露。

实现依赖[PyTorch](https://pytorch.org/)、[TorchVision](https://github.com/pytorch/vision)、[Ultralytics](https://github.com/ultralytics/ultralytics)、[TorchMetrics](https://github.com/Lightning-AI/torchmetrics)、[COCO API](https://github.com/cocodataset/cocoapi)、[NumPy](https://numpy.org/)、[Pillow](https://python-pillow.org/)、[Matplotlib](https://matplotlib.org/)，遵守各自许可证。方法参考：[Faster R-CNN](https://arxiv.org/abs/1506.01497)、[FPN](https://arxiv.org/abs/1612.03144)、[Fast R-CNN／多尺度训练](https://arxiv.org/abs/1504.08083)。

数据为Kai Liu的[Beach Bird Detection Dataset v1.0](https://www.kaggle.com/datasets/liugalhigh/beach-bird)，Kaggle，2026。许可为Other，源图像来自[Wikimedia Commons](https://commons.wikimedia.org/)、[Pexels](https://www.pexels.com/)、[Flickr](https://www.flickr.com/)、[Visual WetlandBirds](https://github.com/3dperceptionlab/Visual-WetlandBirds)、[Big Bird](https://doi.org/10.48610/27809f1)等，仍须遵守源许可与署名条件；版权问题联系liukai@tio.org.cn。
