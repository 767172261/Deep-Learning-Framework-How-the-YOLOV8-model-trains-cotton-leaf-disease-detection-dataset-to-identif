棉花叶片病害检测数据集】
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae2483c484da4e9381b09988a9e2cd1d.png)

nc: 6
names: ['blight', 'curl', 'grey mildew','healthy', 'leaf spot', 'wilt']
名称：【'枯萎病', '卷叶病', '灰霉菌'，'萎蔫病' ，'叶斑病', ‘萎蔫病’】共4173张，8:1:1比例划分，（train;3708张，val：232张，test：233张标注文件为YOLO适用的txt格式和xml格式文件。可以直接用于模型训练。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0bfd4d8db1d549c9ad6fdc3ad189998d.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a14a2215ba59423a906b8d201bcf9f43.png)

---

## ✅ 一、数据集概览表

| 项目 | 内容 |
|------|------|
| **数据集名称** | 棉花叶片病害检测数据集 |
| **总图像数量** | **4,173 张**（高清叶片特写图像） |
| **类别数量（nc）** | **6 类** |
| **类别标签（英文）** | `['blight', 'curl', 'grey mildew', 'healthy', 'leaf spot', 'wilt']` |
| **类别标签（中文）** | `['枯萎病', '卷叶病', '灰霉菌', '健康', '叶斑病', '萎蔫病']` |
| **数据划分比例** | **训练集 : 验证集 : 测试集 = 8 : 1 : 1** |
| **各集数量** | - 训练集（train）：3,708 张- 验证集（val）：232 张- 测试集（test）：233 张 |
| **标注格式** | - **YOLO 格式**（`.txt` 文件，归一化坐标）- **VOC 格式**（`.xml` 文件，含 bounding box） |
| **即用性** | ✅ 可直接用于 YOLOv5/v6/v8/v11 等模型训练 |
| **典型应用场景** | - 智慧农业病害预警- 无人机田间巡检- 手机 App 辅助诊断- 农业保险定损 |

> ⚠️ 注意：  
> - “枯萎病”与“萎蔫病”在中文中易混淆，但本数据集中为**两个独立类别**（对应 `blight` 和 `wilt`）  
> - `healthy` 类别用于区分正常叶片，提升模型鲁棒性

---

## ✅ 二、推荐目录结构（YOLO 格式）

```
cotton_leaf_disease_dataset/
├── images/
│   ├── train/      # 3,708 张
│   ├── val/        # 232 张
│   └── test/       # 233 张
├── labels/
│   ├── train/      # 对应 .txt 标签
│   ├── val/
│   └── test/
└── data.yaml       # YOLO 配置文件
```

---

## ✅ 三、`data.yaml` 配置文件

创建 `cotton_leaf_disease_dataset/data.yaml`：

```yaml
# cotton_leaf_disease_dataset/data.yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 6
names: [
  'blight',
  'curl',
  'grey mildew',
  'healthy',
  'leaf spot',
  'wilt'
]
```

> 🔤 类别顺序必须与 `.txt` 标签中的 class ID 严格一致（0=blight, 1=curl, ..., 5=wilt）

---

## ✅ 四、YOLOv8 完整训练代码（Python）

保存为 `train_cotton_leaf.py`：

```python
# train_cotton_leaf.py
from ultralytics import YOLO
import torch
import os

# 检查数据路径
dataset_path = "cotton_leaf_disease_dataset"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ 数据集路径 {dataset_path} 不存在！")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 使用设备: {device}")

# 加载预训练模型（推荐 yolov8s；若部署到手机可用 yolov8n）
model = YOLO('yolov8s.pt')  # 首次运行自动下载权重

# 开始训练 —— 针对叶片病害（小目标 + 背景复杂）优化
results = model.train(
    data=os.path.join(dataset_path, 'data.yaml'),
    epochs=100,                     # 建议 80~120 轮（收敛较快）
    imgsz=640,                      # 叶片图像通常无需超高分辨率
    batch=64,                       # RTX 3060 可设 64
    name='cotton_leaf_yolov8s',
    
    device=device,
    
    # 🔧 数据增强策略（针对植物病害定制）
    hsv_h=0.02,     # 色调扰动（模拟不同光照/季节）
    hsv_s=0.5,      # 饱和度（应对绿叶颜色变化）
    hsv_v=0.4,      # 亮度
    degrees=15.0,   # 旋转 ±15°（模拟叶片角度）
    translate=0.1,
    scale=0.5,
    mosaic=1.0,     # ✅ 必开！提升小病斑召回
    mixup=0.1,
    flipud=0.0,     # 不上下翻转
    fliplr=0.5,     # 左右翻转（合理）
    
    # 训练策略
    patience=20,    # 早停：20 轮无改善
    save_period=10,
    workers=8,
    cache=False
)

print("✅ 训练完成！")
print(f"最佳模型路径: runs/detect/cotton_leaf_yolov8s/weights/best.pt")
```

---

## ✅ 五、命令行训练（快速启动）

```bash
yolo detect train \
  data=cotton_leaf_disease_dataset/data.yaml \
  model=yolov8s.pt \
  epochs=100 \
  imgsz=640 \
  batch=64 \
  name=cotton_leaf_yolov8s \
  device=0
```

---

## ✅ 六、模型评估

```bash
# 在测试集上评估
yolo detect val \
  model=runs/detect/cotton_leaf_yolov8s/weights/best.pt \
  data=cotton_leaf_disease_dataset/data.yaml
```

> 预期性能：
> - **mAP@0.5 > 0.95**
> - **Recall > 0.90**（避免漏诊病害）

---

## ✅ 七、推理脚本（单图 + 批量）

### 单图推理（返回病害类型与置信度）

```python
# infer_single.py
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict(
    source='leaf_image.jpg',
    conf=0.4,          # 置信度阈值（病害检测建议 ≥0.4）
    iou=0.45,
    save=True,
    show=False
)

# 打印结果
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        print(f"🔍 检测到: {cls_name} (置信度: {conf:.2f})")
```

### 批量处理并生成诊断报告

```python
# infer_batch_report.py
from ultralytics import YOLO
import pandas as pd

model = YOLO('best.pt')
image_dir = 'field_images/'
results_list = []

for img in os.listdir(image_dir):
    if img.endswith(('.jpg', '.png')):
        result = model.predict(os.path.join(image_dir, img), conf=0.4)
        for r in result:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                results_list.append({
                    'image': img,
                    'disease': model.names[cls_id],
                    'confidence': conf
                })

# 保存为 CSV 报告
df = pd.DataFrame(results_list)
df.to_csv('disease_diagnosis_report.csv', index=False)
print("✅ 诊断报告已生成: disease_diagnosis_report.csv")
```

---

## ✅ 八、部署建议

| 场景 | 方案 |
|------|------|
| **手机 App** | 导出 ONNX：`yolo export model=best.pt format=onnx` |
| **无人机巡检** | 部署于 Jetson Nano/NX，实时检测 |
| **Web 平台** | 用 Flask 提供 API：`POST /diagnose → 返回 JSON` |

---


