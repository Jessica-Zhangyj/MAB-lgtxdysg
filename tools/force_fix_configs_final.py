import os
import glob
import yaml

# OOT 配置文件的路径
config_dir = "configs/data_conf/Test_Time_Learning/OOT_Generated"
yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))

print(f"🔍 Found {len(yaml_files)} YAML files in {config_dir}")

for file_path in yaml_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        updated = False

        # 1. 补全 sub_dataset
        if "sub_dataset" not in data:
            data["sub_dataset"] = "generated_oot_slice"
            updated = True
            print(f"  [Fix] Added 'sub_dataset' to {os.path.basename(file_path)}")

        # 2. 补全 max_test_samples
        if "max_test_samples" not in data:
            data["max_test_samples"] = 0
            updated = True
            print(f"  [Fix] Added 'max_test_samples' to {os.path.basename(file_path)}")

        # 3. 补全 seed (这次的罪魁祸首)
        if "seed" not in data:
            data["seed"] = 42
            updated = True
            print(f"  [Fix] Added 'seed' to {os.path.basename(file_path)}")

        # 4. 补全 split (以防万一)
        if "split" not in data:
            data["split"] = "test"
            updated = True
            print(f"  [Fix] Added 'split' to {os.path.basename(file_path)}")

        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            print(f"  [OK] {os.path.basename(file_path)} is already perfect.")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

print("\n✅ All configurations fixed successfully!")