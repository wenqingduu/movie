# Movie项目评测skill

`evaluate-face-injection-v2/`保存当前安装skill的可版本控制副本：官方EntityBench非VLM评价、
完整episode分片、两路并发、输入核验续跑及官方统一聚合。
代码/参考说明与`/root/.codex/skills/evaluate-face-injection-v2/`当前安装版逐字节一致。

本项目流水线从安装路径调用helper；新的机器需先将此目录安装到Codex用户技能目录。
如果已有同名skill，先比较版本并保留旧副本，再按需要更新，避免覆盖其他任务的改动。
安装后无需为了skill版本控制重新生成媒体或重跑已有有效评分。

```bash
mkdir -p /root/.codex/skills
cp -a skills/evaluate-face-injection-v2 /root/.codex/skills/
```

模型权重、环境、原始媒体及评分产物不在Git里；固定路径和依赖见skill参考说明及项目交接文档。
旧`evaluate-face-injection`仍用于配对InsightFace/参考DINO诊断，本副本不替代其算法。
