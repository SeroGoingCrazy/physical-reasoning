# HANDOFF: Metaculus leakage study (Physical Reasoning take-home)

最后更新：2026-11-02。本文件供接手的人或 AI 会话（本机 Claude Code / Codex）从零继续推进。

> **最新修订（2026-11-02，优先于下方历史记录）**：根据审稿意见重新核对并修改英文/中文报告，保留原始80题、预测、标签不变。本轮没有新模型调用、没有重抽样。新建 `REPRODUCE.md`，完整命令和输出在 `analysis/score-opus.txt`、`analysis/score-haiku.txt`。`scripts/score.py` 新增 `--robustness`、`--pilot`、`--horizon-split`、`--seed`：明确组内配对差值百分位 bootstrap，补齐价格偏移、同78题基准、24小时窗口与新增题区间。Opus 主差分 +0.177 [+0.055,+0.300]；未标记子集 +0.049 [−0.079,+0.175]；24小时价格子集 +0.352 [+0.215,+0.499]；53道新增题 +0.216 [+0.075,+0.370]。结论改为组间评分关联与晚信息文本迹象，不能估计记忆贡献。`knows_outcome` 保留为历史字段，明确不是记忆真值；历史提示词和标签未改。AI辅助核验528974的美联储结果陈述与官方来源一致、输入中没有这些结果细节，另列正负标签的歧义案例；不称人工盲审。剩余实证工作：修复后独立重抽、历史输入版本/同时间背景包、双组正负标签人工审查、重复生成。下方关于30题、旧管线、旧数值和已完成工作的段落仅为历史背景；对外以新版报告与复现文档为准。
>

> **状态（2026-11-02）**：样本已扩到每组 40 题（`data_v2/`，首版 30 题保留在 `data/`）。Gamma 列表接口在 offset 约 2000 处返回 422，`scripts/pull_polymarket.py` 新增 `--slice-days`（按日期切片翻页）、`--keep`（保留旧样本的题）、`--max-per-topic`，并输出创建日市场价 `cp_open_prob`。`scripts/score.py` 新增创建日基准、双侧置换检验和 `--table`；新增 `scripts/leak_probe.py`（Haiku 判定推理是否用了结果/事后信息）。Opus 双重差分 +0.177 [+0.055, +0.300]，p = 0.007；只看探针未标记的题则缩到 +0.049 且区间含 0。两版报告已更新。Codex（gpt-6-astra，只读）评审后修复：切片之间空一天（旧题 2734313、532742 因此漏掉）、两组共用去重集合、422 一律视为翻页上限、创建日价格窗口边界；裁判改为看不到预测概率（与旧判定 78/80 一致，差异都是 post 组误判）。`data_v2/` 样本是在修复前抓的，候选池略不完整，未重抽。剩余：用修复后的脚本重抽并重跑（约 1.5 美元）、Opus 4.7 / 2026-02-01 第二切分、探针人工校验、推到公开仓库。
>
> 旧状态（2026-10-29 晚）：数据源已从 Metaculus 改为 Polymarket（原因见第 5、6 节）。管线已在本机跑通：`scripts/pull_polymarket.py` → `scripts/forecast.py`（opus-5-5、haiku-5-5 各 30 题）→ `scripts/score.py`。

## 1. 任务背景

- 来源：Physical Reasoning（physicalreasoning.ai）的 research take-home PDF。
- 选定方向：Direction 1 "Forecasting against the market" 中的 **Leakage** 起点：
  同一模型分别预测训练截止日之前和之后结算的题，论证差距说明了什么关于预测评测中的污染。
- 交付格式：Google Doc / 博客 / GitHub README，约两页，加代码或 notebook 链接。
- 时间预算：1 到 2 小时。评分只看三点：问题是否尖锐、baseline/control 能否排除无聊解释、结论是否与证据相符。排版、长度、算力不计分。
- 写作必须覆盖五点：
  1. 研究问题及其对 world model 的意义
  2. 方法：模型、数据、指标、baseline/control
  3. 一个真跑过的小实验及结果（负结果也行）
  4. 什么结果会改变结论；再有一周会跑什么
  5. 想和他们做什么
- 另加一行说明 AI 工具如何使用。

## 2. 实验设计（已定稿）

**问题**：一个 LLM 相对于 Metaculus 社区预测的优势，在训练截止日前后是否不同？

**分组**（边界是我们定的分组日期，不是对模型的断言）：
- `post`：结算时间晚于边界 **且** 边界之后仍在接受预测（close_time > 边界）。后一条用来排除"结果早已公开、平台后来才结算"的题。
- `pre`：结算时间落在边界前 120 天再往前一年的窗口内。口径是"可能有训练暴露"，不是"一定见过"。

**参数**：
- 边界 `--post-start 2026-07-01`
- 官方截止日标签 `--cutoff-label "2026-06 training data cutoff (Anthropic docs)"`，原样存入 `data/meta.json`
- 社区预测取点：结算前 7 天，且不晚于关闭前一分钟（`--lead-days 7`）
- 只取二元题、结算为 yes/no、预测者至少 10 人
- 每组 15 题，总共 30 题（先试跑规模）

**模型**：
- 主模型 `claude-opus-5-5`，副模型 `claude-haiku-5-5`。两者官方文档的 training data cutoff 与 reliable knowledge cutoff 均为 2026 年 6 月（由用户查证，云端会话未能联网复核）。
- 调用时不挂任何工具（无 web search），system prompt 声明"今天是 <open_time>"，只给题目、描述、结算标准、fine print。不给结算后的任何信息、社区概率、评论。
- 退路：若 post 组题目不足，换 `claude-opus-4-7`（Transparency Hub 标注 knowledge cutoff 2026 年 1 月），边界改为 2026-02-01，并在写作中注明该日期不等于精确训练截止日。

**指标与 control**：
- Brier 分数：模型、社区预测、常数 0.5（Brier = 0.25）三者按组报告。
- 核心量：双重差分 = (post 组 模型Brier − 社区Brier) − (pre 组 模型Brier − 社区Brier)。用社区作参照抵消两组难度差异。
- bootstrap 95% 置信区间（5000 次重采样）。区间含 0 就如实报告无法区分。
- 每组报告平均 horizon_days（开放到结算的天数）和 yes 基率，用于回应"差异来自题目类型或预测期限"的质疑。

**已承认的局限（写作时要点名）**：
- pre 组不保证模型见过；post 组不保证答案在边界前完全未知，close_time 过滤只是代理。
- 把"今天"设为开放日不能删除模型后来学到的知识，只是历史预测指令。
- 不提供搜索只排除实时检索渠道；题目描述后续编辑仍可能带入信息。
- 官方只给到月份，边界 7 月 1 日是保守选择。
- 30 题样本小，差异也可能来自题目难度、主题、预测期限。
- 成本：Opus 5.5 输入 $4 / 输出 $20 每百万 token；每题 2k 输入 + 1k 输出估算 30 题约 $0.84，深度推理或重试会更高。

## 3. 代码现状

目录 `leakage/`，三个文件：

| 文件 | 状态 |
|---|---|
| `scripts/legacy/pull_metaculus.py` | 按 Metaculus 新版 `/api/posts/` 接口编写。**未对真实 API 跑过**（云端会话被网络策略拦截）。 |
| `scripts/score.py` | 在合成数据上跑通。 |
| `README.md` | 使用说明。 |

`scripts/legacy/pull_metaculus.py` 依赖的接口假设，第一次跑时核对：
- 列表：`GET /api/posts/?statuses=resolved&forecast_type=binary&order_by=-published_at&limit=50&offset=N`，可选 `published_at__gt/__lt`，返回 `results` 和 `next`。
- 详情：`GET /api/posts/{id}/?with_cp=true`，字段 `question.type == "binary"`、`question.resolution in {"yes","no"}`、`question.open_time`、`question.actual_close_time` 或 `scheduled_close_time`、`question.actual_resolve_time`、`question.aggregations.recency_weighted.history[*]` 含 `start_time`、`end_time`、`centers`、`forecaster_count`。
- 可选环境变量 `METACULUS_TOKEN`，以 `Authorization: Token ...` 发送，缓解 429。
- 出错时看 stderr 的 skip 原因和 `data/raw/<id>.json`，改 `extract()` 和 `times_of()` 里的字段名。

输出：
- `data/questions.csv`：含 cp_prob、resolution、horizon_days、group。**不要给模型看。**
- `data/prompts.jsonl`：只含题目文本和时间，给模型看。
- `data/meta.json`：全部参数与官方截止日标签。

## 4. 下一步（按顺序）

1. 本机安装依赖并试统计题量：
   ```bash
   pip install requests
   python scripts/legacy/pull_metaculus.py --post-start 2026-07-01 --cutoff-label "2026-06 training data cutoff (Anthropic docs)" --count-only
   ```
   看 stderr 两行 `candidates passed the date filters`。post 组候选 ≥ 20 才够稳；不够则启用退路模型。
2. 正式拉取：
   ```bash
   python scripts/legacy/pull_metaculus.py --post-start 2026-07-01 --cutoff-label "2026-06 training data cutoff (Anthropic docs)" --n-per-group 15
   ```
3. 写 `scripts/forecast.py`（尚未写）。规格：
   - Anthropic Python SDK，模型 ID 从命令行传入，依次跑 `claude-opus-5-5` 和 `claude-haiku-5-5`。
   - 不传 tools。system prompt：你是预测员，今天是 {open_time}，只能用此前的知识；输出 P(yes)。
   - 用 structured output（`output_config.format`）或严格的 JSON 要求拿到一个 0 到 1 的数，解析失败重试一次后记为缺失。
   - 逐题记录 `usage` 的输入输出 token，汇总成本写进文档。
   - 输出 `forecasts/forecasts_<model>.csv`，两列 `id,model_prob`。
   - 对同一题可跑 3 次取均值降低采样噪声，预算允许的话。
4. 打分：
   ```bash
   python scripts/score.py --forecasts forecasts/forecasts_claude-opus-5-5.csv
   python scripts/score.py --forecasts forecasts/forecasts_claude-haiku-5-5.csv
   ```
5. 写两页文档，按第 1 节五点组织。把第 2 节的局限逐条写进"什么结果会改变结论"。
6. 代码推到公开仓库，文档里附链接。

## 5. 决策记录

- 2026-10-29（晚）：**数据源改为 Polymarket**。Metaculus 在 2026 年给 API 加了过滤网关：默认账号拿到的 `resolution`、已关闭题的题目文本、全部社区预测字段一律为 null；只有 bot testing area（项目 32977，约 50 题，全是 10 月 19 日后开放、尚未结算）对所有账号开放当前社区预测。更大的集合（项目 32979，约 250 开放 + 250 已结算）需填 Metaculus Data Needs Form 申请 Bot Benchmarking 或 Unrestricted 层级；条款另要求评测 AI 模型需书面许可。Polymarket 的 Gamma/CLOB API 公开、无需鉴权、有逐日价格历史，且更贴合 "forecasting against the market"。
- 2026-10-29（晚）：Polymarket 版过滤规则：关键词排除体育/电竞/币价/大宗商品盘（列表接口没有标签）；每个 event 只取一个市场；每组同一大写标题词最多 2 题（避免巴西大选类多盘聚集）；领先点价格须在 [0.03, 0.97]（排除市场早已定论的长尾盘）；post 组要求名义截止日 > 边界 + 3 天（排除 "by June 30?" 在 7 月 1 日结算的盘）。社区预测列 `cp_prob` = 结算前 7 天的 YES 价（日线），`cp_forecasters` 列存的是美元成交量。
- 2026-10-29（晚）：`scripts/forecast.py` 不开 refusal fallback（换模型会污染实验），拒答/解析失败重试一次后记缺失；structured output 的 schema 不支持 number 的 minimum/maximum，范围在代码里校验。

- 2026-10-29：放弃 Post-cutoff forecasts 单独做，改做 Leakage，因为它是前者加对照组，直接命中"排除无聊解释"评分项。
- 2026-10-29：放弃 Crowd calibration（跨平台匹配费时、偏离主线）和 Scoring rules（纯模拟、无真实数据）。
- 2026-10-29：`--cutoff` 改名 `--post-start`，区分"我们定的边界"和"官方截止日"。
- 2026-10-29：post 组增加 close_time > 边界的过滤。

## 6. 环境备注

- 本机：项目目录 `~/Desktop/Physical Reasoning `（**目录名末尾有一个空格**）。依赖装在 `.venv`（anaconda 自带的 httpx 解压库与 anthropic SDK 冲突，不要用系统 python）。`.env` 里有 `ANTHROPIC_API_KEY` 与 `METACULUS_TOKEN`，已 gitignore。
- Metaculus API 现状（metaculus.com/api，2026-10-29）：所有请求须带 `Authorization: Token ...`；默认层级字段按上面第 5 节所述被置空；`/api/posts/{id}/download-data/` 返回 403。

- 之前的 Claude 云端会话无法访问 metaculus.com、api.openai.com，容器内没有 Codex CLI，也未连接用户本机。所有网络相关步骤需在本机完成。
- 用户本机已有 Codex CLI。可用 `codex exec "<任务>"` 让它执行第 4 节的步骤。
