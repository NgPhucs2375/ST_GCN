import os
import json
import re

# Đường dẫn tuyệt đối chuẩn đét trỏ thẳng vào folder chứa đống JSON của bro
folder_path = r'D:\code\Nam3_HK2_25-26\Deep learning\ST_GCN\data\raw_ipn_merged_clean' 

count = 0
print("🚀 Đang khởi động máy quét tối tân, tiến hành dọn sạch rác...")

# Kiểm tra xem folder có tồn tại thật không, tránh việc gõ sai chính tả
if not os.path.exists(folder_path):
    print(f"❌ Ôi đứng hình! Thư mục này đéo tồn tại bro ơi, check lại đường dẫn xem: {folder_path}")
    exit()

for filename in os.listdir(folder_path):
    if not filename.endswith('.json'):
        continue
        
    file_path = os.path.join(folder_path, filename)
    
    try:
        # 1. PHẪU THUẬT PHẦN RUỘT: Sửa label bên trong file JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        old_label = data.get('label', '')
        
        # Dùng radar Regex bốc chữ và số đầu tiên (VD: "G04_02" hay "g04_02" -> "G04")
        match_label = re.match(r'^([a-zA-Z0-9]+)', old_label)
        if match_label:
            new_label = match_label.group(1).upper() # Ép chuẩn viết HOA
            data['label'] = new_label
            
            # Ghi đè nội dung sạch lại vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            new_label = old_label

        # 2. PHẪU THUẬT PHẦN VỎ: Đổi tên file bên ngoài cho đồng bộ
        # Radar quét dạng file dính rác ở giữa: G04_01_177987...json
        match_file = re.match(r'^([a-zA-Z0-9]+)_(\d{1,2})_(\d+\.json)$', filename)
        
        if match_file:
            action = match_file.group(1).upper()
            timestamp = match_file.group(3)
            new_filename = f"{action}_{timestamp}"
        else:
            # Dành cho file đã đúng form nhưng chưa viết hoa đầu: g04_17798...json
            match_file_norm = re.match(r'^([a-zA-Z0-9]+)_(\d+\.json)$', filename)
            if match_file_norm:
                action = match_file_norm.group(1).upper()
                timestamp = match_file_norm.group(2)
                new_filename = f"{action}_{timestamp}"
            else:
                new_filename = filename
        
        # Tiến hành đổi tên nếu tên cũ chưa chuẩn
        if new_filename != filename:
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(file_path, new_file_path)
            print(f"✅ Sạch 100%: {filename} ➡️ {new_filename} (Label: {new_label})")
        else:
            print(f"✅ Sửa ruột ok: {filename} (Label: {new_label})")
            
        count += 1
        
    except Exception as e:
        print(f"❌ Lỗi xử lý tại file {filename}: {e}")

print(f"\n🚀 Thành công mỹ mãn! Đã xử lý triệt để cả ruột lẫn vỏ {count} file!")