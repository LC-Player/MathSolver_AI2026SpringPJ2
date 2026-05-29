# postprocess_submit.py
"""后处理 submit_*.csv 结果文件：按第一列去重（后面覆盖前面），补全 0~7999 缺失项。"""

import argparse
import sys


def postprocess(input_path: str, output_path: str | None = None) -> None:
    results: dict[int, str] = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            results[idx] = parts[1]

    out = output_path or input_path

    missing = 0
    with open(out, "w", encoding="utf-8", newline="") as f:
        for i in range(8000):
            if i in results:
                val = results[i]
            else:
                val = "000000"
                missing += 1
            f.write(f"{i},{val}\n")

    if missing > 0:
        print(f"补全 {missing} 个缺失条目", file=sys.stderr)
    print(f"输出: {out} (8000 行)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="后处理 submit_*.csv 文件")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（默认覆盖输入）")
    args = parser.parse_args()
    postprocess(args.input, args.output)


if __name__ == "__main__":
    main()
