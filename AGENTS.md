# 项目开发约定

- 所有 Python 命令必须通过当前项目目录的 uv 环境运行，例如 `uv run python ...`。不要调用系统 Python，也不要创建或切换到其他虚拟环境。
- 运行任何涉及 CUDA、Torch 或 vLLM 的命令前，必须设置 `CUDA_MODULE_LOADING=LAZY`。例如：`CUDA_MODULE_LOADING=LAZY uv run python ...`。
- 除非任务明确要求，否则不要修改仓库中的其他 Python 文件；环境验证应使用临时的内联脚本或一次性命令完成。
