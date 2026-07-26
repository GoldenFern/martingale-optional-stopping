"""pytest 配置：将 src/ 加入导入路径，使测试可从任意工作目录运行。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
