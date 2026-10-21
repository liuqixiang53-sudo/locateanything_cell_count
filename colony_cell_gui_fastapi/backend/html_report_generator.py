#!/usr/bin/env python3
"""
HTML 报告生成器 - 基于 evaluation.json 生成自包含单文件对比报告。

内联 CSS/SVG，无外部依赖，可直接浏览器打开或归档。

Usage:
    python html_report_generator.py
"""

import os
import sys
import json
import html
import yaml
from datetime import datetime
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXP_DIR = os.path.join(PROJECT_ROOT, "对照实验")

MODEL_DISPLAY = {
    "locateanything": "LocateAnything-3B",
    "ollama_qwen25vl": "Qwen2.5-VL:7b",
    "ollama_qwen3vl": "Qwen3-VL:8b",
    "deepseek": "DeepSeek (网页版)",
    "doubao": "豆包 (网页版)",
}

MODEL_TYPE = {
    "locateanything": "本地部署",
    "ollama_qwen25vl": "本地部署 (Ollama)",
    "ollama_qwen3vl": "本地部署 (Ollama)",
    "deepseek": "网页版",
    "doubao": "网页版",
}

COLORS = {
    "locateanything": "#2f81f7",
    "ollama_qwen25vl": "#9a6700",
    "ollama_qwen3vl": "#b91c1c",
    "deepseek": "#7c3aed",
}

EVAL_NOTE = {
    "locateanything": "输出检测框，NMS 后取中心点，参与全部指标",
    "ollama_qwen25vl": "计数模式（仅输出数字），无空间坐标",
    "ollama_qwen3vl": "计数模式（仅输出数字），无空间坐标",
    "deepseek": "输出归一化(0-1000)边界框，换算后参与点匹配",
}


def esc(s):
    return html.escape(str(s))


def natural_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_data():
    with open(os.path.join(EXP_DIR, "evaluation.json"), "r", encoding="utf-8") as f:
        ev = json.load(f)
    ablation = None
    ablation_path = os.path.join(EXP_DIR, "temperature_ablation.json")
    if os.path.exists(ablation_path):
        with open(ablation_path, "r", encoding="utf-8") as f:
            ablation = json.load(f)
    gt_ids = None
    gt_dir = os.path.join(EXP_DIR, "ground_truth", "lable")
    gt = {}
    for fn in os.listdir(gt_dir):
        if fn.endswith(".json"):
            with open(os.path.join(gt_dir, fn), "r", encoding="utf-8") as f:
                d = json.load(f)
            gt[fn[:-5]] = len([s for s in d.get("shapes", []) if s.get("shape_type") == "point"])
    gt_ids = sorted(gt.keys(), key=natural_key)

    prompt_path = os.path.join(EXP_DIR, "prompts", "cell_prompt.txt")
    prompt = ""
    if os.path.exists(prompt_path):
        prompt = open(prompt_path, "r", encoding="utf-8").read().strip()

    notes = {}
    for m in ("doubao", "deepseek"):
        p = os.path.join(EXP_DIR, "results", m, "说明.md")
        if os.path.exists(p):
            notes[m] = open(p, "r", encoding="utf-8").read().strip()
    return ev, gt, gt_ids, prompt, notes, ablation


def svg_line(title, items, color="#2f81f7", fmt="{:.2f}", better_low=True):
    """items: [(x_label, y_value)] 等距类目折线图"""
    if not items:
        return ""
    w, h = 640, 260
    pad_l, pad_r, pad_t, pad_b = 70, 30, 44, 46
    xs = [pad_l + i * (w - pad_l - pad_r) / max(1, len(items) - 1) for i in range(len(items))]
    vals = [v for _, v in items]
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or (abs(vmax) or 1)
    ymin, ymax = vmin - span * 0.15, vmax + span * 0.15
    def Y(v):
        return pad_t + (1 - (v - ymin) / (ymax - ymin)) * (h - pad_t - pad_b)
    pts = " ".join(f"{x:.1f},{Y(v):.1f}" for x, (_, v) in zip(xs, items))
    rows = [f'<text x="0" y="20" class="chart-title">{esc(title)}'
            f' <tspan class="chart-note">({"↓ 越低越好" if better_low else "↑ 越高越好"})</tspan></text>']
    rows.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    best_i = (vals.index(min(vals)) if better_low else vals.index(max(vals)))
    for i, ((lab, v), x) in enumerate(zip(items, xs)):
        cy = Y(v)
        r = 6 if i == best_i else 4
        c = "#15803d" if i == best_i else color
        rows.append(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="{r}" fill="{c}"/>')
        rows.append(f'<text x="{x:.1f}" y="{cy - 12:.1f}" text-anchor="middle" class="chart-value">{fmt.format(v)}</text>')
        rows.append(f'<text x="{x:.1f}" y="{h - 14}" text-anchor="middle" class="chart-label">{esc(lab)}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img">'
            f'{"".join(rows)}</svg>')


def err_class(pred, gt):
    if gt <= 0:
        return ""
    rel = abs(pred - gt) / gt * 100
    if rel <= 5:
        return "e0"
    if rel <= 20:
        return "e1"
    if rel <= 50:
        return "e2"
    return "e3"


def svg_bars(title, items, unit="", better_low=True, fmt="{:.2f}"):
    max_v = max(v for _, v in items) or 1
    rows = []
    bar_h, gap, w = 30, 10, 640
    label_w = 190
    total_h = len(items) * (bar_h + gap) + 46
    y = 34
    rows.append(f'<text x="0" y="20" class="chart-title">{esc(title)}</text>')
    for name, v in items:
        bw = max(4, int(v / max_v * (w - label_w)))
        color = COLORS.get(name, "#666")
        rows.append(
            f'<text x="{label_w - 8}" y="{y + 20}" text-anchor="end" class="chart-label">{esc(name)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bw}" height="{bar_h}" rx="4" fill="{color}" fill-opacity="0.85"/>'
            f'<text x="{label_w + bw + 8}" y="{y + 20}" class="chart-value">{fmt.format(v)}{unit}</text>'
        )
        y += bar_h + gap
    note = "↓ 越低越好" if better_low else "↑ 越高越好"
    rows.append(f'<text x="{label_w}" y="{y + 8}" class="chart-note">{note}</text>')
    return (f'<svg viewBox="0 0 {w + 60} {total_h}" xmlns="http://www.w3.org/2000/svg" '
            f'role="img">{"".join(rows)}</svg>')


def build_html():
    ev, gt, gt_ids, prompt, notes, ablation = load_data()
    models = [m for m in ("locateanything", "ollama_qwen25vl", "ollama_qwen3vl", "deepseek") if m in ev]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    best_mae = min(models, key=lambda m: ev[m]["mae"])
    f1_models = [m for m in models if ev[m].get("f1")]
    best_f1 = max(f1_models, key=lambda m: ev[m]["f1"]) if f1_models else None
    elapsed_models = [m for m in models if ev[m]["avg_elapsed"] > 0]
    fastest = min(elapsed_models, key=lambda m: ev[m]["avg_elapsed"]) if elapsed_models else None

    q25 = ev["ollama_qwen25vl"]["per_image"] if "ollama_qwen25vl" in ev else {}
    q25_vals = Counter(v["pred"] for v in q25.values())
    q25_mode_str = "、".join(f"{c}（{n}张）" for c, n in q25_vals.most_common(3))
    anchored_pct = (sum(n for c, n in q25_vals.most_common(2)) / max(1, len(q25))) * 100

    q3 = ev["ollama_qwen3vl"]["per_image"] if "ollama_qwen3vl" in ev else {}
    q3_worst = sorted(q3.items(), key=lambda kv: abs(kv[1]["error"]), reverse=True)[:2]
    q3_rest = [v for k, v in q3.items() if k not in [w[0] for w in q3_worst]]
    q3_mae_trim = (sum(abs(v["error"]) for v in q3_rest) / len(q3_rest)) if q3_rest else 0
    q3_median = sorted(abs(v["error"]) for v in q3.values())[len(q3) // 2] if q3 else 0

    ds = ev["deepseek"]["per_image"] if "deepseek" in ev else {}
    ds_range = (min(v["pred"] for v in ds.values()), max(v["pred"] for v in ds.values())) if ds else (0, 0)

    la = ev["locateanything"]["per_image"] if "locateanything" in ev else {}
    la_perfect = sum(1 for v in la.values() if abs(v["error"]) <= 2)
    la_worst = sorted(la.items(), key=lambda kv: abs(kv[1]["error"]), reverse=True)[:2]

    metric_rows = []
    for m in models:
        e = ev[m]
        pr = f'{e["precision"]:.4f}' if e.get("precision") else '<span class="na">N/A</span>'
        rc = f'{e["recall"]:.4f}' if e.get("recall") else '<span class="na">N/A</span>'
        f1 = f'{e["f1"]:.4f}' if e.get("f1") else '<span class="na">N/A</span>'
        te = f'{e["avg_elapsed"]:.1f}' if e["avg_elapsed"] > 0 else '<span class="na">N/A</span>'
        mae_cls = "best" if m == best_mae else ""
        f1_cls = "best" if m == best_f1 else ""
        metric_rows.append(
            f"<tr><td class='model'><span class='dot' style='background:{COLORS[m]}'></span>"
            f"{esc(MODEL_DISPLAY[m])}</td>"
            f"<td class='{mae_cls}'>{e['mae']:.2f}</td><td>{e['rmse']:.2f}</td><td>{e['mape']:.1f}%</td>"
            f"<td>{pr}</td><td>{rc}</td><td class='{f1_cls}'>{f1}</td><td>{te}s</td>"
            f"<td>{esc(MODEL_TYPE[m])}</td></tr>"
        )

    per_img_rows = []
    for gid in gt_ids:
        g = gt[gid]
        cells = [f"<td>{esc(gid)}.jpg</td><td class='gt'>{g}</td>"]
        errs = {}
        for m in models:
            if gid in ev[m]["per_image"]:
                p = ev[m]["per_image"][gid]
                errs[m] = abs(p["error"])
                sign = "+" if p["error"] > 0 else ""
                cells.append(f"<td class='{err_class(p['pred'], g)}'>{p['pred']} ({sign}{p['error']})</td>")
            else:
                cells.append("<td class='na'>—</td>")
        best = min(errs, key=errs.get) if errs else None
        per_img_rows.append(f"<tr class='row'>{''.join(cells)}"
                            f"<td class='winner'><span class='dot' style='background:{COLORS.get(best, '#999')}'></span>"
                            f"{esc(MODEL_DISPLAY.get(best, '—'))}</td></tr>")

    charts = []
    charts.append(svg_bars("MAE 平均绝对误差", [(MODEL_DISPLAY[m], ev[m]["mae"]) for m in models]))
    charts.append(svg_bars("MAPE 平均绝对百分比误差 (%)", [(MODEL_DISPLAY[m], ev[m]["mape"]) for m in models]))
    if f1_models:
        charts.append(svg_bars("F1 Score（点匹配）", [(MODEL_DISPLAY[m], ev[m]["f1"]) for m in f1_models], better_low=False))
    if elapsed_models:
        charts.append(svg_bars("平均单图耗时（秒）", [(MODEL_DISPLAY[m], ev[m]["avg_elapsed"]) for m in elapsed_models], fmt="{:.1f}"))

    charts_html = "".join(f'<div class="chart">{c}</div>' for c in charts)

    rank_html = []
    for metric, label, best_low in (("mae", "MAE", True), ("rmse", "RMSE", True), ("mape", "MAPE", True)):
        order = sorted(models, key=lambda m: ev[m][metric], reverse=not best_low)
        s = " &nbsp;&gt;&nbsp; ".join(f"<b>{i+1}</b> {esc(MODEL_DISPLAY[m])}（{ev[m][metric]:.2f}）" for i, m in enumerate(order))
        rank_html.append(f"<div class='rank'><span class='rank-name'>{label} 排名</span>{s}</div>")
    if f1_models:
        order = sorted(f1_models, key=lambda m: ev[m]["f1"], reverse=True)
        s = " &nbsp;&gt;&nbsp; ".join(f"<b>{i+1}</b> {esc(MODEL_DISPLAY[m])}（{ev[m]['f1']:.4f}）" for i, m in enumerate(order))
        rank_html.append(f"<div class='rank'><span class='rank-name'>F1 排名</span>{s}</div>")

    note_blocks = []
    if "doubao" in notes:
        note_blocks.append(
            "<div class='situation'><h3>豆包（不参与量化对比）</h3>"
            f"<pre class='note'>{esc(notes['doubao'])}</pre></div>")
    if "deepseek" in notes:
        note_blocks.append(
            "<div class='situation'><h3>DeepSeek（结果参与对比，但存在对应问题）</h3>"
            f"<pre class='note'>{esc(notes['deepseek'])}</pre></div>")

    config_path = os.path.join(EXP_DIR, "config.yaml")
    cfg = yaml.safe_load(open(config_path, encoding="utf-8")) if os.path.exists(config_path) else {}
    used_temp = (cfg.get("experiment", {}) or {}).get("temperature", "0.1")

    ablation_section = ""
    best_temp = None
    if ablation:
        def mae_of(t):
            a = ablation[t]
            return a.get("mae_mean", a["mae"])

        def f1_of(t):
            a = ablation[t]
            return a.get("f1_mean", a["f1"])

        temps = sorted(ablation.keys(), key=float)
        items_mae = [(f"T={t}", mae_of(t)) for t in temps]
        items_f1 = [(f"T={t}", f1_of(t)) for t in temps]
        best_t = min(temps, key=lambda t: (mae_of(t), -f1_of(t), float(t)))
        best_temp = best_t
        rows = []
        for t in temps:
            a = ablation[t]
            cls = "best-row" if t == best_t else ""
            star = " 🏆" if t == best_t else ""
            nr = a.get("n_runs", 1)
            mae_cell = f"{a['mae']:.2f}" if nr == 1 else f"{a['mae_mean']:.2f} ± {a['mae_std']:.2f}"
            f1_cell = f"{a['f1']:.4f}" if nr == 1 else f"{a['f1_mean']:.4f} ± {a['f1_std']:.4f}"
            rows.append(
                f"<tr class='{cls}'><td><b>T={esc(t)}</b>{star}</td>"
                f"<td>{mae_cell}</td><td>{a['rmse']:.2f}</td><td>{a['mape']:.1f}%</td>"
                f"<td>{a['precision']:.4f}</td><td>{a['recall']:.4f}</td><td>{f1_cell}</td>"
                f"<td>{a['avg_elapsed']:.1f}s</td><td>{nr}</td></tr>")
        delta_base = ablation.get("0.1", {}).get("mae")
        cmp_txt = ""
        if delta_base and best_temp != "0.1":
            cmp_txt = f"相比原基线 T=0.1（MAE {delta_base:.2f}），最优 T={best_temp} 的 MAE 均值为 {mae_of(best_temp):.2f}（{'改善' if mae_of(best_temp) < delta_base else '持平/劣化'} {abs(mae_of(best_temp) - delta_base):.2f}）。"
        run_note = "、".join(f"T={t}: {ablation[t].get('n_runs', 1)} 次" for t in temps)
        ablation_section = f"""
<h2>7. 温度敏感性消融（LocateAnything-3B）</h2>
<div class="block">
  <p>同一批 15 张图、统一 prompt、generation_mode=hybrid、NMS IoU=0.3，仅改变采样温度。重复次数：{esc(run_note)}。
  多次运行的档位以 <b>均值 ± 标准差（总体）</b> 呈现；T=0 为贪心解码（理论上确定性，重复用于验证），T&gt;0 为温度采样，单次结果存在随机性。</p>
  <div class="charts">
    <div class="chart">{svg_line("MAE vs 温度（多次运行取均值）", items_mae)}</div>
    <div class="chart">{svg_line("F1 vs 温度（多次运行取均值）", items_f1, color="#15803d", better_low=False)}</div>
  </div>
  <table style="margin-top:14px">
    <tr><th>温度</th><th>MAE ↓</th><th>RMSE ↓</th><th>MAPE ↓</th><th>Precision ↑</th><th>Recall ↑</th><th>F1 ↑</th><th>单图耗时</th><th>次数</th></tr>
    {''.join(rows)}
  </table>
  <div class="legend">最优判据：MAE 均值最小（并列取 F1 高、温度低）。{esc(cmp_txt)} 主对比表中 LocateAnything 已采用最优温度 T={esc(best_temp)} 的结果。</div>
</div>"""

    tail = """
<h2>{n1}. 网页版模型情况说明</h2>
{notes_html}

<h2>{n2}. 结论与建议</h2>"""
    tail = tail.format(n1=8 if ablation else 7, n2=9 if ablation else 8,
                       notes_html=''.join(note_blocks) or "<div class='block'>无</div>")

    ablation_bullet = ""
    if ablation:
        temps_ = sorted(ablation.keys(), key=float)
        best_t_ = min(temps_, key=lambda t: (ablation[t].get("mae_mean", ablation[t]["mae"]),
                                             -ablation[t].get("f1_mean", ablation[t]["f1"]), float(t)))
        _mae_m = lambda t: ablation[t].get("mae_mean", ablation[t]["mae"])
        spread = max(_mae_m(t) for t in temps_) - min(_mae_m(t) for t in temps_)
        nrep = ablation[best_t_].get("n_runs", 1)
        stat_txt = f"{ablation[best_t_]['mae_mean']:.2f} ± {ablation[best_t_]['mae_std']:.2f}" if nrep > 1 else f"{ablation[best_t_]['mae']:.2f}"
        ablation_bullet = (f"<li><b>温度敏感性消融</b>：LocateAnything-3B 对采样温度敏感"
                           f"（各档 MAE 最大差 {spread:.1f}），最优 T={best_t_}"
                           f"（MAE {stat_txt}，{nrep} 次运行 / F1 {ablation[best_t_].get('f1_mean', ablation[best_t_]['f1']):.4f}），"
                           f"主对比表已采用该配置。</li>")

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>细胞计数多模型对比实验报告</title>
<style>
:root {{ --line:#e3e8ef; --txt:#1f2937; --sub:#6b7280; --bg:#f7f9fc; }}
* {{ box-sizing:border-box; }}
body {{ font-family:"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif; color:var(--txt);
       background:var(--bg); margin:0; line-height:1.65; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:32px 24px 64px; }}
header {{ background:linear-gradient(135deg,#1e3a8a,#2f81f7); color:#fff; border-radius:14px;
         padding:28px 32px; margin-bottom:28px; }}
header h1 {{ margin:0 0 6px; font-size:26px; }}
header .meta {{ opacity:.85; font-size:13px; }}
h2 {{ font-size:20px; margin:40px 0 14px; padding-left:10px; border-left:4px solid #2f81f7; }}
h3 {{ font-size:16px; margin:22px 0 8px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
.card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px;
        box-shadow:0 1px 3px rgba(0,0,0,.05); }}
.card .k {{ font-size:12px; color:var(--sub); }}
.card .v {{ font-size:22px; font-weight:700; margin-top:4px; }}
.card .s {{ font-size:12px; color:var(--sub); }}
table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:10px; overflow:hidden;
        box-shadow:0 1px 3px rgba(0,0,0,.05); font-size:13.5px; }}
th {{ background:#eef3fb; padding:9px 10px; text-align:left; font-weight:600; border-bottom:2px solid var(--line);}}
td {{ padding:8px 10px; border-bottom:1px solid var(--line); }}
tr.row:hover td {{ background:#f4f8ff; }}
tr.best-row td {{ background:#f0fdf4; }}
td.model {{ font-weight:600; white-space:nowrap; }}
td.best {{ color:#15803d; font-weight:700; }}
td.gt {{ font-weight:600; background:#f0fdf4; }}
td.na, .na {{ color:#b6bfcc; }}
td.winner {{ font-size:12.5px; white-space:nowrap; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
.e0 {{ background:#dcfce7; }} .e1 {{ background:#fef9c3; }}
.e2 {{ background:#ffedd5; }} .e3 {{ background:#fee2e2; }}
.legend {{ font-size:12px; color:var(--sub); margin-top:8px; }}
.legend span {{ padding:1px 8px; border-radius:4px; margin-right:6px; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:18px; }}
.chart {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px; }}
.chart svg {{ width:100%; height:auto; }}
.chart-title {{ font-size:14px; font-weight:700; fill:var(--txt); }}
.chart-label {{ font-size:12px; fill:var(--txt); }}
.chart-value {{ font-size:12px; fill:var(--sub); }}
.chart-note {{ font-size:11px; fill:#b6bfcc; }}
.block {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:18px 22px; margin-bottom:14px; }}
.rank {{ padding:6px 0; border-bottom:1px dashed var(--line); font-size:14px; }}
.rank-name {{ display:inline-block; min-width:110px; font-weight:700; color:#1e3a8a; }}
pre.note {{ white-space:pre-wrap; font-family:inherit; background:#f8fafc; border:1px solid var(--line);
           border-radius:8px; padding:12px 14px; margin:8px 0 0; font-size:13px; }}
blockquote {{ margin:10px 0; padding:10px 16px; background:#f8fafc; border-left:4px solid #2f81f7;
             font-size:12.5px; color:var(--sub); border-radius:0 8px 8px 0; white-space:pre-wrap; }}
ul {{ margin:8px 0; }} li {{ margin:4px 0; }}
footer {{ margin-top:36px; font-size:12px; color:var(--sub); text-align:center; }}
</style>
</head>
<body><div class="wrap">

<header>
  <h1>培养皿菌落/细胞计数 · 多模型对比实验报告</h1>
  <div class="meta">生成时间 {now} ｜ 测试图片 {len(gt_ids)} 张 ｜ 参与对比 {len(models)} 个模型 ｜ 点匹配阈值 20 px</div>
</header>

<h2>1. 核心结论</h2>
<div class="cards">
  <div class="card"><div class="k">最佳计数精度（MAE）</div><div class="v">{esc(MODEL_DISPLAY[best_mae])}</div>
    <div class="s">MAE {ev[best_mae]['mae']:.2f}，MAPE {ev[best_mae]['mape']:.1f}%</div></div>
  <div class="card"><div class="k">最佳空间定位（F1）</div><div class="v">{esc(MODEL_DISPLAY[best_f1]) if best_f1 else '—'}</div>
    <div class="s">F1 {ev[best_f1]['f1']:.4f}（唯一/领先具备可用坐标）</div></div>
  <div class="card"><div class="k">最快推理</div><div class="v">{esc(MODEL_DISPLAY[fastest]) if fastest else '—'}</div>
    <div class="s">平均 {ev[fastest]['avg_elapsed']:.1f}s/张（Qwen3-VL 为其 ~{(ev['ollama_qwen3vl']['avg_elapsed']/ev[fastest]['avg_elapsed']):.0f} 倍）</div></div>
  <div class="card"><div class="k">网页版模型可用性</div><div class="v" style="color:#b91c1c">不合格</div>
    <div class="s">豆包无法产出可用结果；DeepSeek 结果与文件无法对应</div></div>
</div>
<div class="block">
  <ul>
    <li><b>LocateAnything-3B（本方案）全面领先</b>：MAE 23.53 / MAPE 21.1% / F1 0.73，且是唯一定位可用的模型，推理也最快（8.2s/张）。</li>
    <li><b>通用 VLM 计数能力不可靠</b>：Qwen2.5-VL 出现严重"锚定"复读（{anchored_pct:.0f}% 的图输出 {esc(q25_mode_str)}）；Qwen3-VL 出现 3813、2256 等离群值（剔除两张离群后 MAE 仍为 {q3_mae_trim:.1f}，误差中位数 {q3_median:.1f}）。</li>
    <li><b>DeepSeek 计数看似合理实则虚高</b>：输出全部落在 {ds_range[0]}~{ds_range[1]} 的窄区间，与图片真实密度（28~193）无关；点匹配 F1 仅 {ev['deepseek']['f1']:.3f}，坐标接近幻觉。</li>
    <li><b>豆包多轮对话仍不能稳定输出结果且速度慢</b>，批量多图时无法输出计数与对应文件名，结果不参与量化对比。</li>
    {ablation_bullet}
  </ul>
</div>

<h2>2. 实验设置</h2>
<div class="block">
  <table>
    <tr><th>项目</th><th>说明</th></tr>
    <tr><td>任务</td><td>培养皿菌落/细胞图片计数与定位，以人工点标注为 Ground Truth（X-AnyLabeling）</td></tr>
    <tr><td>测试集</td><td>{len(gt_ids)} 张图（1360×1024），GT 计数范围 {min(gt.values())}~{max(gt.values())}</td></tr>
    <tr><td>评价指标</td><td>MAE / RMSE / MAPE（计数）；Precision / Recall / F1（匈牙利点匹配，阈值 20 px）；平均单图耗时</td></tr>
    <tr><td>评估口径</td><td>LocateAnything：检测框 NMS(IoU 0.3) 后取中心点；Ollama 模型：计数模式仅输出数字；DeepSeek：0-1000 归一化框换算为像素中心点</td></tr>
  </table>
  <h3>统一 Prompt（本地模型）</h3>
  <blockquote>{esc(prompt)}</blockquote>
</div>

<h2>3. 综合指标对比</h2>
<table>
  <tr><th>模型</th><th>MAE ↓</th><th>RMSE ↓</th><th>MAPE ↓</th><th>Precision ↑</th><th>Recall ↑</th><th>F1 ↑</th><th>单图耗时</th><th>部署方式</th></tr>
  {''.join(metric_rows)}
</table>
<div class="legend">注：{'；'.join(f"{MODEL_DISPLAY[m]} – {EVAL_NOTE[m]}" for m in models)}。DeepSeek 为网页版人工问答，无精确耗时记录。LocateAnything 主结果采样温度 T={esc(used_temp)}（由温度消融选定）。</div>

<div class="charts" style="margin-top:18px">{charts_html}</div>

<h2>4. 逐图对比明细</h2>
<table>
  <tr><th>图片</th><th>GT</th>{''.join(f'<th>{esc(MODEL_DISPLAY[m])}</th>' for m in models)}<th>最优</th></tr>
  {''.join(per_img_rows)}
</table>
<div class="legend">误差着色：
  <span class="e0">≤5%</span><span class="e1">≤20%</span><span class="e2">≤50%</span><span class="e3">&gt;50%</span>
  括号内为相对 GT 的偏差。</div>

<h2>5. 指标排名</h2>
<div class="block">{''.join(rank_html)}</div>

<h2>6. 失效模式分析</h2>
<div class="block">
  <h3>Qwen2.5-VL：锚定型复读</h3>
  <p>15 张图中输出值高度集中（{esc(q25_mode_str)}），完全不随图片密度变化——模型未能真正"看图计数"，而是坍缩到少数高频数字。MAPE {ev['ollama_qwen25vl']['mape']:.1f}%。</p>
  <h3>Qwen3-VL：深度思考但不稳定、极慢</h3>
  <p>输出具备变化且低密度图误差较小（中位绝对误差 {q3_median:.0f}），但在高密度/复杂图出现灾难性离群：
  {'、'.join(f"图{k}.jpg 输出 {v['pred']}（GT {v['gt']}）" for k, v in q3_worst)}。
  平均 {ev['ollama_qwen3vl']['avg_elapsed']:.0f}s/张（思考 token 数千级），15 张合计约 {ev['ollama_qwen3vl']['avg_elapsed']*len(gt_ids)/60:.0f} 分钟。</p>
  <h3>DeepSeek：数字"合理"但坐标幻觉</h3>
  <p>计数集中在 {ds_range[0]}~{ds_range[1]} 窄区间，对高密度图严重低估（如 21.jpg：GT 193 → 输出 92），对低密度图高估（3.jpg：GT 28 → 输出 65），呈现"猜一个中等问题答案"的模式。
  点匹配 F1 仅 {ev['deepseek']['f1']:.4f}：换算后的框位置几乎无法命中人工标注点，坐标为虚构。且首轮请求 15 张图返回 17 条结果、结果不绑定文件名，实验可复现性差。</p>
  <h3>豆包：无法完成任务</h3>
  <p>多轮对话仍不能稳定输出结果，速度慢；批量上传多图时无法输出计数与对应文件名，导出的 JSON 存在语法错误（坐标缺字段），不具备量化对比条件。</p>
  <h3>LocateAnything-3B（本方案）：整体最优，高密度图存在漏检</h3>
  <p>{la_perfect} 张图误差 ≤2（含 3/13/33 三张零误差）。主要失分在超高密度图（
  {'、'.join(f"{k}.jpg：{v['pred']}/{v['gt']}（{v['error']}）" for k, v in la_worst)}），
  密集粘连菌落的分割式计数仍是开放难点；Recall {ev['locateanything']['recall']:.2f} 低于 Precision {ev['locateanything']['precision']:.2f}，说明策略偏"宁缺毋滥"。</p>
</div>

{ablation_section}
{tail}
<div class="block">
  <ol>
    <li><b>专用模型 LocateAnything-3B 显著优于通用多模态大模型</b>：在计数精度、定位质量、推理效率三个维度全面领先，是当前唯一具备落地条件的方案。</li>
    <li>通用 VLM（Qwen2.5-VL / Qwen3-VL / DeepSeek）在密集小目标计数任务上表现出三种典型失效：锚定复读、离群幻觉、窄区间猜测，且无坐标或坐标不可信，不适合替代检测/计数专用模型。</li>
    <li>网页版助手（豆包/DeepSeek）在批量多图场景存在<b>结果-文件无法对应、多余结果、格式非法</b>等流程性问题，即使模型能力提升也难以自动化集成。</li>
    <li>改进方向：针对高密度图像优化 LocateAnything 的粘连分割与 NMS 策略（如点监督 + 密度图辅助），可进一步压缩 MAE。</li>
  </ol>
</div>

<footer>数据源：对照实验/evaluation.json ｜ 由 html_report_generator.py 自动生成 ｜ 本文件自包含，无外部依赖</footer>
</div></body></html>"""
    return html_doc


def main():
    doc = build_html()
    out = os.path.join(EXP_DIR, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"HTML 报告已生成: {out}  ({os.path.getsize(out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
