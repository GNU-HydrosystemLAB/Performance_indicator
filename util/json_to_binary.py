import json
import numpy as np
import cv2
import os


def json_to_binary_mask_labelme(json_path):
    """
    LabelMe JSON과 같은 경로의 이미지 파일을 기반으로 
    주석 polygon으로 구성된 이진 마스크 생성

    Parameters:
        json_path (str): LabelMe 형식의 .json 파일 경로

    Returns:
        mask (np.ndarray): polygon이 흰색(255)으로 채워진 이진 마스크
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 이미지 경로 구성
    image_filename = data["imagePath"]
    image_dir = os.path.dirname(json_path)
    image_path = os.path.join(image_dir, image_filename)

    # 이미지 크기 가져오기
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    height, width = image.shape[:2]

    # 마스크 초기화
    mask = np.zeros((height, width), dtype=np.uint8)

    # polygon 채우기
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=255)

    return mask

def json_to_binary_mask_anylabel(json_path):
    """
    LabelMe JSON 파일에 imageHeight, imageWidth 정보가 있는 경우,
    이를 기반으로 이진 마스크 생성

    Parameters:
        json_path (str): JSON 파일 경로

    Returns:
        mask (np.ndarray): polygon이 채워진 이진 마스크 (uint8, 0 또는 255)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    height = data['imageHeight']
    width = data['imageWidth']

    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=255)

    return mask

def find_image_json_pairs(directory, image_exts={'.png', '.jpg', '.jpeg', '.bmp'}):
    """
    디렉토리에서 이미지와 동일한 이름의 JSON 파일 쌍을 찾음

    Parameters:
        directory (str): 탐색할 디렉토리 경로
        image_exts (set): 허용된 이미지 확장자들 (기본: png, jpg, jpeg, bmp)

    Returns:
        list[list[str, str]]: [[image_path, json_path], ...] 형식의 리스트
    """
    files = os.listdir(directory)
    image_files = {}
    json_files = {}

    # 파일들을 분류
    for fname in files:
        name, ext = os.path.splitext(fname)
        ext = ext.lower()

        if ext in image_exts:
            image_files[name] = os.path.join(directory, fname)
        elif ext == '.json':
            json_files[name] = os.path.join(directory, fname)

    # 공통된 이름을 가진 쌍 찾기
    pairs = []
    for name in image_files.keys() & json_files.keys():
        pairs.append([image_files[name], json_files[name]])

    return pairs

def generate_masks_in_directory(directory):
    """
    디렉토리 내 이미지-JSON 쌍에서 마스크 생성 후 저장 (_mask.png 형식)

    Parameters:
        directory (str): 탐색할 디렉토리 경로
    """
    pairs = find_image_json_pairs(directory)

    for img_path, json_path in pairs:
        mask = json_to_binary_mask_anylabel(json_path)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(directory, mask_filename)

        cv2.imwrite(mask_path, mask)
        print(f"Saved mask: {mask_path}")

def remove_all_json_files(directory):
    """
    지정한 디렉토리 내 모든 .json 파일을 삭제

    Parameters:
        directory (str): 대상 디렉토리 경로
    """
    removed = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                removed += 1
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    print(f"총 삭제된 JSON 파일 수: {removed}")