# Ground Truth 标注说明

## 格式

每张图片一个 JSON 文件，命名为 `{图片编号}_gt.json`。

```json
{
  "image": "1.jpg",
  "total_count": 113,
  "annotator": "your_name",
  "date": "2026-09-02",
  "points": [
    {"x": 597.0, "y": 486.0},
    {"x": 623.0, "y": 512.0}
  ]
}
```

## 标注要求

1. 标注所有可见细胞，包括边缘处不完整的细胞
2. 每个细胞标注其中心点坐标
3. 坐标以图片左上角为原点 (0,0)，x 向右，y 向下
4. 建议使用 X-AnyLabeling 的点标注模式，或本项目配套的标注工具

## 示例

对于 `对照实验/1.jpg`，ground truth 文件为 `对照实验/ground_truth/1_gt.json`