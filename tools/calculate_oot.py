import json

def calculate_oot_probability(file_path):
    """
    计算检索结果中 Out-of-Task (OOT) 内容的平均概率。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的 JSON 格式")
        return

    # 提取数据部分
    # 注意：根据你的文件结构，数据可能直接在根目录，或者在 'data' 字段下
    # 这里假设结构是 {'data': [...]}，如果不是，请根据实际情况调整
    if isinstance(data, dict) and 'data' in data:
        items = data['data']
    elif isinstance(data, list):
        items = data
    else:
        print("错误: 无法识别数据结构")
        return

    total_oot_ratio = 0
    count = 0
    
    print(f"正在处理文件: {file_path} ...")

    for item in items:
        # 检查是否存在检索标签
        if 'retrieval_labels' in item:
            labels = item['retrieval_labels']
            k = len(labels)
            
            # 防止除以零
            if k == 0:
                continue
                
            # 统计包含 "Out-of-Task" 的标签数量
            # 这里使用了字符串包含判断，只要标签里有这个词就算
            oot_count = sum(1 for label in labels if label and "Out-of-Task" in str(label))
            
            # 计算当前样本的 OOT 比例
            ratio = oot_count / k
            
            # 累加
            total_oot_ratio += ratio
            count += 1
            
            # (可选) 打印每个样本的详细信息
            # print(f"Query ID {item.get('query_id')}: Top-{k} 中有 {oot_count} 个 OOT (概率: {ratio:.2f})")

    # 计算平均值
    if count > 0:
        average_ratio = total_oot_ratio / count
        print("-" * 30)
        print(f"统计样本数 (有检索记录的): {count}")
        print(f"平均 Out-of-Task 检索概率: {average_ratio:.4f} ({average_ratio*100:.2f}%)")
        print("-" * 30)
    else:
        print("未找到包含 'retrieval_labels' 的有效样本。")

# --- 使用方法 ---
if __name__ == "__main__":
    # 请将此处替换为你的实际文件名
    json_file = '/root/autodl-tmp/MemoryAgentBench/outputs/glm-z1-9b/Accurate_Retrieval/ruler_qa1_197K_repeat_noise_attack_in220000_size50_shots0_max_samples100_k10_chunk512_results.json'
    
    calculate_oot_probability(json_file)