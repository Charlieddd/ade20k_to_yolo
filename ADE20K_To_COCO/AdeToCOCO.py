# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_ade20k_ins_seg.py
# ADE20K dataset: https://groups.csail.mit.edu/vision/datasets/ADE20K/
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util
from collections import defaultdict

# For demo
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2, random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances


def pickleload(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def saveJson(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


class AdeToCOCO():
    # 修改 COMMON_CATEGORIES，将相似类别统一映射
    COMMON_CATEGORIES = {
        # 垃圾桶相关
        "trash_can": [
            "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
            "rubbish, trash, scrap"
        ],
        
        # 花盆相关
        "pot": [
            "pot, flowerpot",
            "pot",
            "planter"
        ],
        
        # 楼梯相关
        "stairs": [
            "stairs, steps",
            "stairway, staircase"
        ],
        
        # 球类
        "ball": [
            "ball",
            "ball, globe, orb"
        ],
        
        # 其他高频类别的映射
        "wall": "wall",
        "person": "person, individual, someone, somebody, mortal, soul",
        "tree": [
            "tree",
            "bamboo",
            "trunk, tree trunk, bole"
        ],
        "car": "car, auto, automobile, machine, motorcar",
        "plant": "plant, flora, plant life",
        "sky": "sky",
        "leg": "leg",
        "book": "book",
        "sidewalk": "sidewalk, pavement",
        "grass": "grass",
        "earth": "earth, ground",
        "bottle": "bottle",
        "rock": "rock, stone",
        "flower": "flower",
        "fence": "fence, fencing",
        "vase": "vase",
        "glass": "glass, drinking glass",
        "pole": "pole",
        "carpet": "rug, carpet, carpeting",
        "floor": "floor, flooring",
        "light": [
            "light, light source",
            "fluorescent, fluorescent fixture",
            "light bulb, lightbulb, bulb, incandescent lamp, electric light, electric-light bulb",
            "skylight, fanlight"
        ],
        # 统一窗户相关的类别到 "window"
        "window": [
            "windowpane, window",
            "window",
            "shop window",
            "screen door, screen",
            "pane, pane of glass, window glass"
        ],
        "sign": "signboard, sign",
        "road": "road, route",
        "streetlight": [
            "streetlight, street lamp",
            "street lamp",
            "street light"
        ],
        "wheel": "wheel",
        "outlet": "wall socket, wall plug, electric outlet, electrical outlet, outlet, electric receptacle",
        "railing": [
            "railing",
            "railing, rail",
            "partition, divider"
        ],
        "basket": "basket, handbasket",
        "fluorescent": "fluorescent, fluorescent fixture",
        "paper": [
            "paper",
            "paper towel"
        ],
        "switch": "switch, electric switch, electrical switch",
        "faucet": [
            "faucet",
            "faucet, spigot"
        ],
        "lid": "lid",
        "candle": "candle, taper, wax light",
        "curb": "curb, curbing, kerb",
        "hood": "hood, exhaust hood",
        "tube": "tube",
        "pipe": [
            "pipe",
            "pipe, pipage, piping",
            "pipe, tube"
        ],
        "vent": "vent",
        "doorframe": [
            "doorframe",
            "doorframe, doorcase"
        ],
        "stall": "stall, stand, sales booth",
        "antenna": [
            "antenna",
            "antenna, aerial, transmitting aerial"
        ],
        "cue": "cue, cue stick, pool cue, pool stick",
        "audio": [
            "speaker",
            "microphone, mike"
        ],
        "clothes": [
            "jersey, T-shirt, tee shirt",
            "trouser, pant",
            "sweater, jumper"
        ],
        "boat": [
            "boat",
            "sailboat, sailing boat",
            "aircraft carrier, carrier, flattop, attack aircraft carrier"
        ],
        "furniture": [
            "rocking chair, rocker",
            "pedestal, plinth, footstall"
        ],
        "computer_peripheral": [
            "mouse, computer mouse"
        ],
        "handle": [
            "handle",
            "handle, grip, handgrip, hold",
            "handle, grip"
        ],
        "elevator": [
            "elevator, lift"
        ],
        "gas_pump": [
            "gas pump, gasoline pump, petrol pump, island dispenser"
        ],
        "grille": [
            "grille",
            "grill, grille, grillwork",
            "arcade, colonnade"
        ],
        "label": [
            "label, recording label"
        ],
        "shell": [
            "carapace, shell, cuticle, shield"
        ],
        "arch": [
            "arch, archway",
            "terrace"  # 相关的建筑结构
        ],
        "scale": [
            "scale, weighing machine"
        ],
        # 叶子相关
        "leaf": [
            "leaf, leafage, foliage",
            "leaves"  # 如果数据集中有这个变体
        ],
    }

    def __init__(self, pklPath, datasetDir, objectNames):
        """
        Args:
            pklPath (str): path to the ADE20K index pickle file
            datasetDir (str): path to the ADE20K dataset directory
            objectNames (list): list of object names to convert
        """
        self.statics = pickleload(pklPath)
        self.datasetDir = datasetDir
        print("Available objects in pkl file:")
        print(self.statics["objectnames"][:10])
        
        # 创建名称映射
        self.name_mapping = {}
        
        # 处理 COMMON_CATEGORIES 中的映射
        for target_name, source_names in self.COMMON_CATEGORIES.items():
            if target_name not in objectNames:
                continue

            if isinstance(source_names, str):
                # 处理单个映射
                # 将完整字符串作为一个整体进行匹配
                if source_names in self.statics["objectnames"]:
                    self.name_mapping[source_names] = target_name
                # 同时也检查每个单独的同义词
                for name in [n.strip() for n in source_names.split(",")]:
                    if name in self.statics["objectnames"]:
                        self.name_mapping[name] = target_name
                    
            elif isinstance(source_names, list):
                # 处理多个来源映射到同一个目标
                for source_name in source_names:
                    # 将完整字符串作为一个整体进行匹配
                    if source_name in self.statics["objectnames"]:
                        self.name_mapping[source_name] = target_name
                    # 同时也检查每个单独的同义词
                    for name in [n.strip() for n in source_name.split(",")]:
                        if name in self.statics["objectnames"]:
                            self.name_mapping[name] = target_name
        
        # 处理其他未在 COMMON_CATEGORIES 中定义的类别
        for name in objectNames:
            if name in self.name_mapping.values():
                continue  # 跳过已经映射的类别
                
            # 在statics中查找匹配的完整名称
            for full_name in self.statics["objectnames"]:
                # 如果完整名称完全匹配
                if name == full_name:
                    self.name_mapping[full_name] = name
                    break
                # 检查同义词
                if "," in full_name:
                    synonyms = [s.strip() for s in full_name.split(",")]
                    if name in synonyms:
                        self.name_mapping[full_name] = name
                        break
        
        self.objectNames = list(self.name_mapping.keys())
        print(f"\nFound {len(self.objectNames)} valid objects in pkl file")
        print("\nName mappings:")
        for full_name, simple_name in self.name_mapping.items():
            print(f"{simple_name} <- {full_name}")
        
        self.annId = 1

    def getObjectIdbyName(self, name):
        """Get object id by object name
        
        Args:
            name (str): object name
        Returns:
            objId (int): object id
        """
        try:
            objId = np.where(np.array(self.statics["objectnames"]) == name)[0][0]
            return int(objId)
        except IndexError:
            print(f"Warning: Object name '{name}' not found in pkl file")
            return None

    def getImageIds(self, names):
        """Get image ids by object names
        
        Args:
            names (list): list of object names
        Returns:
            imgIds (list): list of image ids
        """
        all_image_ids = []

        for name in names:
            objId = self.getObjectIdbyName(name)
            current_image_ids = np.where(
                self.statics["objectPresence"][objId] > 0)[0]
            all_image_ids.append(current_image_ids)

        imgIds = np.unique(np.concatenate(all_image_ids))
        return imgIds.tolist()

    def getImagePathbyId(self, imageId):
        """Get image path by image id
        
        Args:
            imageId (int): image id
        Returns:
            path (str): image path
        """
        # 从 statics 中获取的路径可能包含 'ADE20K_2021_17_01/images/'，需要清理
        folder = self.statics["folder"][imageId]
        # 只保留 'ADE/...' 之后的部分
        if 'ADE/' in folder:
            folder = folder[folder.index('ADE/'):]
        
        base_path = Path(self.datasetDir) / "images"
        path = base_path / folder / self.statics["filename"][imageId]
        assert path.exists(), f"Image file not exist: {path}"
        return str(path)

    def getInfoJsonbyId(self, imageId):
        """Get image information json file path by image id and load its content
        
        Args:
            imageId (int): image id
        Returns:
            imageInfo (dict): image information
        """
        # 从 statics 中获取的路径可能包含 'ADE20K_2021_17_01/images/'，需要清理
        folder = self.statics["folder"][imageId]
        # 只保留 'ADE/...' 之后的部分
        if 'ADE/' in folder:
            folder = folder[folder.index('ADE/'):]
        
        base_path = Path(self.datasetDir) / "images"
        json_path = base_path / folder / \
            self.statics["filename"][imageId].replace("jpg", "json")
        assert json_path.exists(), f"Image information json file not exist: {json_path}"

        # 尝试不同的编码
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(json_path, 'r', encoding=encoding) as f:
                    return json.load(f)['annotation']
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error with {encoding} encoding: {str(e)}")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error reading {json_path}: {str(e)}")
                continue
            
        raise ValueError(f"Could not decode {json_path} with any of the attempted encodings")

    def generateAnnotations(self, imageId, imageInfo, stats):
        """Generate annotations for a single image in COCO format
        Args:
            imageId (int): image id
            imageInfo (dict): image information
            stats (dict): 统计信息字典
        Returns:
            annotations (list): list of annotations
        """
        objects = imageInfo["object"]
        annotations = []
        failed_objects = []
        
        for obj in objects:
            try:
                # 使用完整名称检查
                if obj["name"] not in self.objectNames:
                    stats["skipped_by_reason"]["not_in_objectNames"] += 1
                    stats["skipped_by_category"][obj["name"]] += 1
                    continue
                
                # 使用映射后的简单名称
                obj_name = self.name_mapping[obj["name"]]

                # 检查必要的字段是否存在
                if "polygon" not in obj or "name_ndx" not in obj:
                    stats["skipped_by_reason"]["missing_fields"] += 1
                    failed_objects.append({
                        'name': obj.get('name', 'unknown'),
                        'reason': 'Missing required fields (polygon or name_ndx)'
                    })
                    continue

                polygon = obj["polygon"]
                if "x" not in polygon or "y" not in polygon:
                    failed_objects.append({
                        'name': obj['name'],
                        'reason': 'Invalid polygon data (missing x or y coordinates)'
                    })
                    continue

                # 检查坐标是否有效
                if len(polygon['x']) < 3 or len(polygon['y']) < 3:
                    if not obj["name"] in {'gaze', 'eye', 'point', 'stick', 'backplate', 'hand', 'seagull'}:
                        failed_objects.append({
                            'name': obj['name'],
                            'reason': 'Invalid polygon (less than 3 points)'
                        })
                    continue

                annotation = {
                    "id": int(self.annId),
                    "image_id": int(imageId),
                    "category_id": int(obj["name_ndx"]),
                    "segmentation": [],
                    "area": float,
                    "bbox": [],
                    "iscrowd": int(0)
                }

                # 添加分割点
                polygon_points = []
                for x, y in zip(polygon['x'], polygon['y']):
                    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                        continue
                    polygon_points.extend([float(x), float(y)])
                
                if len(polygon_points) < 6:  # 至少需要3个点
                    failed_objects.append({
                        'name': obj['name'],
                        'reason': 'No valid segmentation points'
                    })
                    continue

                annotation["segmentation"] = [polygon_points]  # 包装成列表格式 [[x1,y1,x2,y2,...]]

                # calculate bounding box
                xs = polygon_points[::2]  # 所有x坐标
                ys = polygon_points[1::2]  # 所有y坐标
                annotation["bbox"] = [
                    int(min(xs)),
                    int(min(ys)),
                    int(max(xs) - min(xs) + 1),
                    int(max(ys) - min(ys) + 1)
                ]

                # calculate area - 使用多边形面积公式
                area = 0
                for i in range(0, len(polygon_points), 2):
                    j = (i + 2) % len(polygon_points)
                    area += polygon_points[i] * polygon_points[j+1]
                    area -= polygon_points[i+1] * polygon_points[j]
                annotation["area"] = abs(area) / 2

                annotations.append(annotation)
                self.annId += 1

            except Exception as e:
                print(f"Failed to process object in image {imageId}: {str(e)}")
                failed_objects.append({
                    'name': obj.get('name', 'unknown'),
                    'reason': f'Unexpected error: {str(e)}'
                })
                continue
        
        return annotations, failed_objects

    def generateImage(self, imageId, imagePath, imageInfo):
        """ Generate image information for a single image in COCO format
        Args:
            imageId (int): image id
            imagePath (str): image path
            imageInfo (dict): image information in ADE20K format
        Returns:
            image (dict): image information in COCO format
        """
        image = {"id": int, "file_name": str, "width": int, "height": int}
        image["id"] = int(imageId)
        image["file_name"] = imagePath
        image["width"] = int(imageInfo["imsize"][1])
        image["height"] = int(imageInfo["imsize"][0])
        return image

    def convert(self, debug=False):
        # Convert Category
        adeCategories = []
        skipped_objects = set()  # 记录跳过的对象
        processed_objects = set()  # 记录成功处理的对象
        
        for name in self.objectNames:
            print(f"Convert {name}")
            categoryDict = {"id": int, "name": str}
            try:
                id = self.getObjectIdbyName(name)
                if id is None:
                    skipped_objects.add(name)
                    continue
                id = id + 1  # consist with seg json name_ndx
                categoryDict["id"] = id
                categoryDict["name"] = name
                adeCategories.append(categoryDict)
                processed_objects.add(name)
            except Exception as e:
                print(f"Error converting {name}: {str(e)}")
                skipped_objects.add(name)
                continue

        trainDict = {}
        valDict = {}

        trainImages = []
        trainCategory = adeCategories
        trainAnnotations = []

        valImages = []
        valCategory = adeCategories
        valAnnotations = []
        
        # 统计信息字典
        stats = {
            "total_objects": 0,
            "converted_objects": 0,
            "skipped_by_category": defaultdict(int),
            "skipped_by_reason": defaultdict(int)
        }

        for imgId in tqdm(self.getImageIds(self.objectNames)):
            try:
                imageInfo = self.getInfoJsonbyId(imgId)
            except Exception as e:
                print(f"Failed to process image {imgId}: {str(e)}")
                continue

            imagePath = self.getImagePathbyId(imgId)
            image = self.generateImage(imgId, imagePath, imageInfo)
            
            # 统计原始对象
            objects = imageInfo["object"]
            stats["total_objects"] += len(objects)
            
            annotations, failed_objects = self.generateAnnotations(imgId, imageInfo, stats)
            stats["converted_objects"] += len(annotations)
            
            if debug and failed_objects:
                print(f"Failed objects in image {imgId}:")
                for obj in failed_objects:
                    print(f"  - Object '{obj['name']}': {obj['reason']}")

            if "ADE/training" in imagePath:
                trainImages.append(image)
                trainAnnotations.extend(annotations)
            elif "ADE/validation" in imagePath:
                valImages.append(image)
                valAnnotations.extend(annotations)
            else:
                print(f"{imagePath} is not in training or validation set")

        # 打印统计信息
        print("\n=== Conversion Statistics ===")
        print(f"Total objects found in objects.txt: {len(self.objectNames)}")
        print(f"Successfully processed objects: {len(processed_objects)}")
        print(f"Skipped objects: {len(skipped_objects)}")
        if skipped_objects:
            print("First 10 skipped objects:", list(skipped_objects)[:10])
        
        print(f"\nTotal images processed: {stats['total_objects']}")
        print(f"Successfully converted: {stats['converted_objects']}")
        print(f"Objects skipped: {stats['total_objects'] - stats['converted_objects']}")
        
        # 保存统计信息
        stats_file = Path(self.datasetDir) / "conversion_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=== ADE20K to COCO Conversion Statistics ===\n\n")
            f.write(f"Total objects found: {stats['total_objects']}\n")
            f.write(f"Successfully converted: {stats['converted_objects']}\n")
            f.write(f"Objects skipped: {stats['total_objects'] - stats['converted_objects']}\n\n")
            
            f.write("=== Skipped by Reason ===\n")
            for reason, count in stats["skipped_by_reason"].items():
                f.write(f"{reason}: {count}\n")
            
            f.write("\n=== Top 20 Skipped Categories ===\n")
            sorted_categories = sorted(stats["skipped_by_category"].items(), key=lambda x: x[1], reverse=True)
            for cat, count in sorted_categories[:20]:
                f.write(f"{cat}: {count}\n")
        
        print(f"\nDetailed statistics saved to {stats_file}")
        
        # 添加总标注数量信息
        print(f"\nTotal annotations generated:")
        print(f"Training set: {len(trainAnnotations)}")
        print(f"Validation set: {len(valAnnotations)}")
        print(f"Total: {len(trainAnnotations) + len(valAnnotations)}")
        
        print("\nGenerating output files...")
        
        trainDict["images"] = trainImages
        trainDict["categories"] = trainCategory
        trainDict["annotations"] = trainAnnotations

        valDict["images"] = valImages
        valDict["categories"] = valCategory
        valDict["annotations"] = valAnnotations

        trainOutputFilePath = Path(self.datasetDir) / "annotations" / "ade20k_instance_train.json"
        valOutputFilePath = Path(self.datasetDir) / "annotations" / "ade20k_instance_val.json"
        
        # 确保输出目录存在
        trainOutputFilePath.parent.mkdir(parents=True, exist_ok=True)
        
        saveJson(trainDict, trainOutputFilePath)
        saveJson(valDict, valOutputFilePath)
        
        print(f"\nOutput files saved to:")
        print(f"Train: {trainOutputFilePath}")
        print(f"Val: {valOutputFilePath}")


class DemoTest():

    def __init__(self, datasetDir):
        """ A class to run demo to check the converted COCO format
        Args:
            datasetDir (str): path to the ADE20K dataset directory
        """
        self.datasetDir = datasetDir

    def startDemo(self):
        datasetName = "ade20k2021_train"
        trainJsonFilePath = Path(self.datasetDir) / "annotations" / "ade20k_instance_train.json"
        register_coco_instances(datasetName, {}, trainJsonFilePath, self.datasetDir)
        dataset = DatasetCatalog.get(datasetName)
        for data in random.sample(dataset, 3):
            fileName = data["file_name"]
            img = cv2.imread(fileName)
            visualizer = Visualizer(img[:, :, ::-1],
                                    metadata=MetadataCatalog.get(datasetName))
            out = visualizer.draw_dataset_dict(data)
            plt.title(fileName.split('/')[-1])
            plt.imshow(
                cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            plt.show()


def read_object_names(file_path):
    """Read object names from objects.txt
    
    Args:
        file_path (str): path to objects.txt
    Returns:
        object_names (list): list of object names
    """
    object_names = []
    try:
        with open(file_path, 'r', encoding='iso-8859-1') as f:
            # Skip header line
            next(f)
            for line in f:
                # Split by tab
                parts = line.strip().split('\t')
                if len(parts) >= 5:  # Make sure we have at least 5 columns
                    # Get the ADE names (5th column)
                    ade_names = parts[4].strip()
                    # Split by comma and clean up each name
                    names = [name.strip() for name in ade_names.split(',')]
                    # Filter out empty names and add to list
                    names = [name for name in names if name and not name[0].isdigit()]
                    object_names.extend(names)
        return object_names
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ADE20K to COCO format")
    parser.add_argument("--datasetDir",
                        type=str,
                        required=True,
                        help="Path to the ADE20K dataset directory")
    parser.add_argument(
        "--pklPath",
        type=str,
        required=True,
        help="Path to the ADE20K index pickle file (index_ade20k.pkl)")
    parser.add_argument("--objectNames",
                        type=str,
                        nargs='+',
                        required=False,
                        help="List of specific object names to convert. If not provided, all objects will be converted")
    parser.add_argument("--demo",
                        type=bool,
                        default=False,
                        help="Run demo after converting")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Show debug information for failed images")

    args = parser.parse_args()

    datasetDir = args.datasetDir
    pklPath = args.pklPath
    
    if args.objectNames:
        objectNames = args.objectNames
    else:
        # Read all object names from objects.txt
        objects_txt_path = Path(datasetDir) / "objects.txt"
        objectNames = read_object_names(objects_txt_path)
        print(f"Found {len(objectNames)} objects in objects.txt")
    
    print(f"Converting {len(objectNames)} objects in {datasetDir}")
    converter = AdeToCOCO(pklPath, datasetDir, objectNames)
    print("Start Converting.....")
    converter.convert(debug=args.debug)  # 传递debug参数
    print("Finish Conversion")

    if args.demo:
        print("Start Demo.....")
        test = DemoTest(datasetDir)
        test.startDemo()
        print("Finish Demo")
