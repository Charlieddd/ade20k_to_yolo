from general_json2yolo import convert_coco_json
import json
from pathlib import Path
import shutil
import os
from collections import defaultdict

def filter_annotations(json_path, target_classes):
    """Filter JSON annotations to keep only the target classes
    
    Args:
        json_path (str): Path to JSON file
        target_classes (dict): Dictionary mapping original names to new IDs
    """
    with open(json_path) as f:
        data = json.load(f)
    
    # Create filtered categories list
    filtered_categories = []
    for cat in data['categories']:
        if cat['name'] in target_classes:
            filtered_categories.append({
                'id': target_classes[cat['name']],
                'name': cat['name']
            })
    
    # Create image id to size mapping
    image_to_size = {img['id']: (img['width'], img['height']) for img in data['images']}
    
    # Filter annotations to keep only target classes
    filtered_annotations = []
    used_image_ids = set()  # 跟踪哪些图片包含我们需要的标注
    
    for ann in data['annotations']:
        cat_name = next((cat['name'] for cat in data['categories'] if cat['id'] == ann['category_id']), None)
        if cat_name in target_classes:  # 只处理目标类别的标注
            # 验证分割点坐标
            if 'segmentation' in ann:
                # 获取图片尺寸
                img_width, img_height = image_to_size[ann['image_id']]
                
                # 检查并修复分割点坐标
                valid_segments = []
                for segment in ann['segmentation']:
                    # 将坐标对齐到图片边界
                    fixed_segment = []
                    for i in range(0, len(segment), 2):
                        x = min(max(0, segment[i]), img_width)
                        y = min(max(0, segment[i + 1]), img_height)
                        fixed_segment.extend([x, y])
                    
                    # 只有当分割包含足够的点时才添加
                    if len(fixed_segment) >= 6:  # 至少3个点才能形成多边形
                        valid_segments.append(fixed_segment)
                
                # 只有当有有效的分割时才添加标注
                if valid_segments:
                    ann_copy = ann.copy()
                    ann_copy['segmentation'] = valid_segments
                    ann_copy['category_id'] = target_classes[cat_name]  # 使用新的类别ID
                    filtered_annotations.append(ann_copy)
                    used_image_ids.add(ann['image_id'])
    
    # 只保留包含目标类别标注的图片
    filtered_images = [img for img in data['images'] if img['id'] in used_image_ids]
    
    # Create new JSON with filtered data
    filtered_data = {
        'images': filtered_images,  # 只包含有目标类别标注的图片
        'categories': filtered_categories,
        'annotations': filtered_annotations
    }
    
    # Save filtered JSON
    filtered_path = Path(json_path).parent / f"filtered_{Path(json_path).name}"
    with open(filtered_path, 'w') as f:
        json.dump(filtered_data, f)
    
    return filtered_path

def generate_statistics(json_file, output_file, split_name):
    """Generate statistics for annotations in a COCO format JSON file
    
    Args:
        json_file (str): Path to COCO format JSON file
        output_file (str): Path to output statistics file
        split_name (str): Name of the dataset split (e.g. 'Training Set' or 'Validation Set')
    """
    with open(json_file) as f:
        data = json.load(f)
    
    # Create category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Count annotations per category
    category_counts = defaultdict(int)
    invalid_categories = set()
    
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id in cat_id_to_name:
            category_counts[cat_id] += 1
        else:
            invalid_categories.add(cat_id)
    
    # Calculate total annotations
    total_annotations = sum(category_counts.values())
    
    # Sort categories by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Write statistics to file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{split_name}:\n")
        f.write(f"Total categories: {len(data['categories'])}\n")
        f.write(f"Total annotations: {total_annotations}\n")
        if invalid_categories:
            f.write(f"Warning: Found {len(invalid_categories)} invalid category IDs: {sorted(invalid_categories)}\n")
        f.write("\n")
        
        # Write header for category statistics
        f.write("Top categories by annotation count:\n")
        f.write("ID     Name                           Count      Percentage\n")
        f.write("-" * 60 + "\n")
        
        # Write statistics for each category
        for cat_id, count in sorted_categories:
            name = cat_id_to_name[cat_id]
            percentage = (count / total_annotations) * 100
            f.write(f"{cat_id:<6} {name:<30} {count:<10} {percentage:>6.2f}%\n")

def clean_label_files(labels_dir):
    """Clean label files by removing lines with -1 class id
    
    Args:
        labels_dir (Path): Directory containing label files
    """
    for label_file in labels_dir.glob("*.txt"):
        # 读取所有行
        with open(label_file) as f:
            lines = f.readlines()
        
        # 过滤掉包含-1的行
        filtered_lines = [line for line in lines if not line.startswith("-1")]
        
        # 如果有行被过滤掉，重写文件
        if len(filtered_lines) != len(lines):
            with open(label_file, 'w') as f:
                f.writelines(filtered_lines)
            
            # 如果文件为空，删除文件
            if not filtered_lines:
                label_file.unlink()

def convert_with_debug():
    json_dir = "/home/charlie/fiftyone/ADE20K_2021_17_01/annotations"
    dataset_dir = "/home/charlie/fiftyone/ADE20K_2021_17_01"
    
    # 创建必要的目录结构
    output_base = Path("dataset")
    output_base.mkdir(exist_ok=True)
    
    # 创建images和labels目录
    for split in ['train', 'val']:
        (output_base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_base / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 复制并重命名图片到新的目录结构
    def copy_images(json_file, split):
        with open(json_file) as f:
            data = json.load(f)
            for img in data['images']:
                src = Path(img['file_name'])
                new_name = src.name
                dst = output_base / 'images' / split / new_name
                shutil.copy2(src, dst)
                # 只使用文件名
                img['file_name'] = new_name
            
            # 保存修改后的JSON文件
            new_json_path = output_base / f"instances_{split}.json"
            with open(new_json_path, 'w') as f:
                json.dump(data, f)
            return new_json_path
    
    # 复制训练集和验证集的图片
    train_json = copy_images(Path(json_dir) / "ade20k_instance_train.json", "train")
    val_json = copy_images(Path(json_dir) / "ade20k_instance_val.json", "val")
    
    # 定义要保留的类别（按频率排序，使用原始名称）
    target_classes = {
        "wall": 0,
        "person, individual, someone, somebody, mortal, soul": 1,
        "tree": 2,
        "car, auto, automobile, machine, motorcar": 3,
        "plant, flora, plant life": 4,
        "leg": 5,
        "book": 6,
        "sidewalk, pavement": 7,
        "grass": 8,
        "earth, ground": 9,
        "bottle": 10,
        "rock, stone": 11,
        "pot, flowerpot": 12,  # 合并 pot 相关
        "flower": 13,
        "fence, fencing": 14,
        "vase": 15,
        "glass, drinking glass": 16,
        "pole": 17,
        "rug, carpet, carpeting": 18,
        "stairs, steps": 19,  # 合并 stairs 相关
        "stool": 20,
        "jar": 21,
        "plaything, toy": 22,
        "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin": 23,
        "water": 24,
        "path": 25,
        "ball": 26,  # 合并 ball 相关
        "switch, electric switch, electrical switch": 27,
        "base, pedestal, stand": 28,
        "fruit": 29,
        "shoe": 30,
        "umbrella": 31,
        "bicycle, bike, wheel, cycle": 32,
        "minibike, motorbike": 33,
        "fireplace, hearth, open fireplace": 34,
        "sculpture": 35,
        "sand": 36,
        "bucket, pail": 37,
        "animal, animate being, beast, brute, creature, fauna": 38,
        "mug": 39,
        "land, ground, soil": 40,
        "curb, curbing, kerb": 41,
        "snow": 42,
        "stove": 43,
        "trunk, tree trunk, bole": 44
    }
    
    # 过滤JSON只保留目标类别
    filtered_train_json = filter_annotations(train_json, target_classes)
    filtered_val_json = filter_annotations(val_json, target_classes)
    
    # 生成统计信息
    stats_file = output_base / "annotation_statistics.txt"
    if stats_file.exists():
        stats_file.unlink()
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== Annotation Statistics ===\n")
    
    generate_statistics(filtered_train_json, stats_file, "Training Set")
    generate_statistics(filtered_val_json, stats_file, "Validation Set")
    
    print(f"\nStatistics saved to {stats_file}")
    
    # 转换标注
    print("\nStarting conversion...")
    
    # 如果存在旧的new_dir，先删除
    if Path('new_dir').exists():
        shutil.rmtree('new_dir')
    
    # 重命名过滤后的JSON文件以匹配默认名称
    shutil.copy2(filtered_train_json, output_base / "instances_train.json")
    shutil.copy2(filtered_val_json, output_base / "instances_val.json")
    
    # 使用默认参数进行转换
    convert_coco_json(
        json_dir=str(output_base),
        use_segments=True,
        cls91to80=False
    )
    
    # 移动生成的标签文件到正确的位置
    for split in ['train', 'val']:
        src_dir = Path('new_dir/labels') / split
        dst_dir = output_base / 'labels' / split
        if src_dir.exists():
            # 移动所有txt文件
            for label_file in src_dir.glob('*.txt'):
                shutil.move(str(label_file), str(dst_dir / label_file.name))
    
    # 清理标签文件
    for split in ['train', 'val']:
        labels_dir = output_base / 'labels' / split
        clean_label_files(labels_dir)
        print(f"\nCleaned {split} labels")
    
    # 清理临时目录
    if Path('new_dir').exists():
        shutil.rmtree('new_dir')
    
    # 检查输出
    print("\nChecking output:")
    for split in ['train', 'val']:
        label_dir = output_base / 'labels' / split
        if label_dir.exists():
            n_labels = len(list(label_dir.glob("*.txt")))
            print(f"{split} labels: {n_labels}")
            if n_labels > 0:
                example_file = next(label_dir.glob("*.txt"))
                print(f"\nExample {split} label ({example_file}):")
                with open(example_file) as f:
                    print(f.read())
    
    # 创建data.yaml文件
    data_yaml = {
        'path': str(output_base.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': '',
        'names': {i: name.split(',')[0].strip() for name, i in target_classes.items()}  # 使用第一个名称作为显示名称
    }
    
    with open(output_base / 'data.yaml', 'w') as f:
        import yaml
        yaml.dump(data_yaml, f, sort_keys=False)

def get_categories(json_file):
    with open(json_file) as f:
        data = json.load(f)
        return [cat['name'] for cat in sorted(data['categories'], key=lambda x: x['id'])]

if __name__ == "__main__":
    convert_with_debug()