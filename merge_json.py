import json
import sys
import argparse

def merge_json_files(input_files, output_file):
    """
    合并多个 JSON 文件，基于 'id' 字段去重。
    
    Args:
        input_files (list): 输入 JSON 文件路径列表
        output_file (str): 输出 JSON 文件路径
    """
    data_dict = {}
    
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"警告：文件 {file_path} 的根元素不是数组，跳过", file=sys.stderr)
                    continue
                
                for item in data:
                    if not isinstance(item, dict) or 'id' not in item:
                        print(f"警告：文件 {file_path} 中存在缺少 'id' 字段或非字典元素，跳过该项", file=sys.stderr)
                        continue
                    
                    item_id = str(item['id'])  # 确保 id 为字符串类型进行比较
                    if item_id not in data_dict:
                        data_dict[item_id] = item
                        
        except json.JSONDecodeError as e:
            print(f"错误：文件 {file_path} 不是有效的 JSON - {e}", file=sys.stderr)
        except Exception as e:
            print(f"错误：处理文件 {file_path} 时发生异常 - {e}", file=sys.stderr)
    
    # 输出去重后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(list(data_dict.values()), f, ensure_ascii=False, indent=2)
    
    print(f"合并完成！共 {len(data_dict)} 条记录，已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='合并多个 JSON 文件（数组格式），按 id 去重')
    parser.add_argument('input_files', nargs='+', help='输入 JSON 文件路径（至少一个）')
    parser.add_argument('-o', '--output', required=True, help='输出 JSON 文件路径')
    args = parser.parse_args()
    
    merge_json_files(args.input_files, args.output)

if __name__ == '__main__':
    main()