# 菌落与细胞计数分析软件 V1.0

基于 LocateAnything-3B 模型的菌落与细胞计数分析工具，提供 Web 可视化界面，支持自动检测、手动修正和结果导出。

## 功能特性

### 菌落计数
- 自适应计数策略：根据菌落密度自动选择稀疏/密集/拥挤处理算法
- 支持参数调节：稀疏阈值、密集阈值、NMS IoU、去重距离、CLAHE 对比度、CLAHE 网格大小、分水岭面积因子、分水岭距离比、分水岭最小面积比
- 手动添加/删除：点击画布添加点，点击已有点删除

### 细胞计数
- 基于 LocateAnything 视觉语言模型的细胞检测
- 支持框选区域：可框选感兴趣区域进行局部计数
- 支持参数调节：NMS IoU、推理模式（MTP/NTP/Hybrid）

### 通用功能
- 图片缩放与平移：支持滚轮缩放、拖拽平移
- 撤销/重置：支持多步撤销和一键重置
- 结果导出：导出 CSV 文件（含坐标、类型、序号）
- 双模式切换：标点模式与框选模式互不干扰

## 项目结构

```
colony_cell_gui_fastapi/
├── frontend/
│   └── index.html          # 前端页面
├── backend/
│   ├── main.py             # FastAPI 后端
│   └── requirements.txt    # Python 依赖
├── uploads/                # 上传图片缓存
├── results/                # 结果图片缓存
└── run_fastapi.sh          # 启动脚本
```

## 环境要求

- Python >= 3.10
- CUDA 支持的 GPU（推荐显存 >= 8GB）
- 模型文件：`LocateAnything-3B/`（放置在项目根目录）

## 安装

### 1. 创建虚拟环境

```bash
cd locateanthing_colony_count
python -m venv locateanthing
source locateanthing/bin/activate
```

### 2. 安装依赖

```bash
pip install -r colony_cell_gui_fastapi/backend/requirements.txt
pip install transformers accelerate
```

### 3. 准备模型

确保 `LocateAnything-3B/` 目录位于项目根目录下：

```
locateanthing_colony_count/
├── LocateAnything-3B/    # 模型文件
├── colony_cell_gui_fastapi/
└── ...
```

## 启动

### 方式一：使用启动脚本（推荐）

```bash
cd colony_cell_gui_fastapi
chmod +x run_fastapi.sh
./run_fastapi.sh
```

### 方式二：手动启动

```bash
cd colony_cell_gui_fastapi/backend
source ../../locateanthing/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 访问

- Web 界面：http://localhost:8000
- API 文档：http://localhost:8000/docs

## 使用说明

### 菌落计数

1. 切换到「菌落计数」标签页
2. 上传培养皿图片
3. （可选）在「参数调节面板」调整参数
4. 点击「开始计数」，等待自动检测完成
5. 在画布上点击添加遗漏的菌落，点击已有红点删除误检
6. 点击「导出结果」保存 CSV

### 细胞计数

1. 切换到「细胞计数」标签页
2. 上传显微镜图片
3. 选择交互模式：
   - **标点模式**：点击画布添加/删除标记点
   - **框选模式**：拖拽框选区域，仅对选中区域计数
4. 点击「裁剪并计数」
5. 手动修正后导出结果

### 画布操作

| 操作 | 方式 |
|------|------|
| 缩放 | Ctrl + 滚轮 / 工具栏 +/- 按钮 |
| 平移 | 点击平移按钮后拖拽 |
| 适应 | 点击适应按钮 |
| 添加点 | 标点模式下点击空白处 |
| 删除点 | 点击已有的点 |

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/colony/count` | POST | 菌落自动计数 |
| `/api/cell/count` | POST | 细胞自动计数（支持框选） |
| `/api/health` | GET | 健康检查 |

详细参数见 http://localhost:8000/docs

## 远程连接

服务默认监听 `0.0.0.0:8000`，支持远程访问。

### 方式一：SSH 端口转发（推荐）

在本地终端执行：

```bash
ssh -L 8000:localhost:8000 user@remote-server
```

然后在本地浏览器访问 http://localhost:8000

优点：安全、无需开放防火墙端口。

### 方式二：局域网直接访问

1. 确保服务绑定 `0.0.0.0`（默认已配置）
2. 在防火墙开放 8000 端口：

```bash
# Linux (ufw)
sudo ufw allow 8000/tcp

# Linux (firewalld)
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

3. 在其他设备浏览器访问 `http://<服务器IP>:8000`

### 方式三：自定义端口

如需修改端口，编辑 `run_fastapi.sh` 或手动启动时指定：

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

对应 SSH 转发：

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

然后访问 http://localhost:8080

## 注意事项

- 首次启动会自动下载/加载模型，耗时约 1-2 分钟
- 菌落计数使用自适应策略，根据密度自动切换算法
- 细胞计数依赖 GPU，处理时间取决于图片大小和参数设置
- 导出 CSV 使用 UTF-8 BOM 编码，可直接用 Excel 打开
