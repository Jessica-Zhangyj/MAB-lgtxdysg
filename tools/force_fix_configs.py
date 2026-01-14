import os
import glob
import yaml

# OOT 配置文件的路径
config_dir = "configs/data_conf/Test_Time_Learning/OOT_Generated"
yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))

print(f"🔍 Found {len(yaml_files)} YAML files in {config_dir}")

for file_path in yaml_files:
    try:
        # 1. 读取现有内容
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        # 2. 强制补全缺失字段
        updated = False
        
        # 补全 sub_dataset
        if "sub_dataset" not in data:
            data["sub_dataset"] = "generated_oot_slice"
            updated = True
            print(f"  [Fixing] Added 'sub_dataset' to {os.path.basename(file_path)}")
            
        # 补全 max_test_samples
        if "max_test_samples" not in data:
            data["max_test_samples"] = 0
            updated = True
            print(f"  [Fixing] Added 'max_test_samples' to {os.path.basename(file_path)}")

        # 3. 如果有修改，写回文件
        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                # default_flow_style=False 保证生成易读的块状 YAML
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            print(f"  [OK] {os.path.basename(file_path)} is already correct.")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

print("\n✅ All configurations fixed!")