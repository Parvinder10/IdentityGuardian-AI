import random
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from PIL import Image, ImageDraw, ImageFilter

class IdentityAugmenter:
    """
    Applies scan distortions, lighting shifts, and high-stakes identity tampering attacks
    (including face-swap splices and digit updates) to evaluate model robustness.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def add_noise(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image).astype(np.float32) / 255.0
        var_range = self.config["augmentations"]["noise"]["gaussian_variance_range"]
        variance = random.uniform(var_range[0], var_range[1])
        noise = np.random.normal(0, np.sqrt(variance), img_np.shape)
        noisy_img = np.clip(img_np + noise, 0.0, 1.0) * 255.0
        return Image.fromarray(noisy_img.astype(np.uint8))

    def add_blur(self, image: Image.Image) -> Image.Image:
        k_range = self.config["augmentations"]["blur"]["kernel_size_range"]
        k_size = random.choice([k for k in range(k_range[0], k_range[1] + 1) if k % 2 == 1])
        return image.filter(ImageFilter.GaussianBlur(radius=k_size // 2))

    def apply_perspective_warp(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        width, height = image.size
        img_np = np.array(image)
        scale = self.config["augmentations"]["distortion"]["perspective_warp_scale"]
        dx = width * scale
        dy = height * scale
        
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([
            [random.uniform(0, dx), random.uniform(0, dy)],
            [width - random.uniform(0, dx), random.uniform(0, dy)],
            [random.uniform(0, dx), height - random.uniform(0, dy)],
            [width - random.uniform(0, dx), height - random.uniform(0, dy)]
        ])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_np = cv2.warpPerspective(img_np, matrix, (width, height), borderValue=(255, 255, 255))
        warped_img = Image.fromarray(warped_np)
        
        adjusted_metadata = metadata.copy()
        new_boxes = []
        for box in metadata.get("layout_boxes", []):
            ymin, xmin, ymax, xmax = box["box_2d"]
            py_min, px_min = (ymin / 1000.0) * height, (xmin / 1000.0) * width
            py_max, px_max = (ymax / 1000.0) * height, (xmax / 1000.0) * width
            
            corners = np.array([[[px_min, py_min]], [[px_max, py_min]], [[px_min, py_max]], [[px_max, py_max]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(corners, matrix)
            xs = transformed[:, 0, 0]
            ys = transformed[:, 0, 1]
            
            ny_min = max(0, min(1000, int((np.min(ys) / height) * 1000)))
            nx_min = max(0, min(1000, int((np.min(xs) / width) * 1000)))
            ny_max = max(0, min(1000, int((np.max(ys) / height) * 1000)))
            nx_max = max(0, min(1000, int((np.max(xs) / width) * 1000)))
            
            new_box = box.copy()
            new_box["box_2d"] = [ny_min, nx_min, ny_max, nx_max]
            new_boxes.append(new_box)
            
        adjusted_metadata["layout_boxes"] = new_boxes
        return warped_img, adjusted_metadata

    def apply_face_swap_attack(self, card_image: Image.Image, metadata: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any], List[int]]:
        """
        Splicing an alternative face cartoon shape into the profile picture box of the ID.
        This simulates digital presentation attacks.
        """
        width, height = card_image.size
        # Find Profile_Photo box
        photo_box = next((b for b in metadata["layout_boxes"] if b["label"] == "Profile_Photo"), None)
        if not photo_box:
            return card_image, metadata, [0, 0, 0, 0]
            
        ymin, xmin, ymax, xmax = photo_box["box_2d"]
        py_min, px_min = int((ymin / 1000) * height), int((xmin / 1000) * width)
        py_max, px_max = int((ymax / 1000) * height), int((xmax / 1000) * width)
        
        # Draw a different alien face to paste on top
        alien_face = Image.new("RGB", (px_max - px_min, py_max - py_min), (230, 230, 230))
        draw_f = ImageDraw.Draw(alien_face)
        # Draw alien red face oval
        draw_f.ellipse([5, 5, alien_face.width-5, alien_face.height-5], fill=(220, 100, 100)) # alien skin
        draw_f.ellipse([15, 30, 25, 40], fill=(0,0,0)) # alien eyes
        draw_f.ellipse([alien_face.width-25, 30, alien_face.width-15, 40], fill=(0,0,0))
        
        card_copy = card_image.copy()
        card_copy.paste(alien_face, (px_min, py_min))
        
        metadata_copy = metadata.copy()
        metadata_copy["face_swapped"] = True
        
        return card_copy, metadata_copy, [py_min, px_min, py_max, px_max]

    def apply_text_replacement(self, card_image: Image.Image, metadata: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any], List[int]]:
        """Modifies numerical text in ID_Number slot to simulate credential manipulation."""
        width, height = card_image.size
        id_box = next((b for b in metadata["layout_boxes"] if b["label"] == "ID_Number"), None)
        if not id_box:
            return card_image, metadata, [0, 0, 0, 0]
            
        ymin, xmin, ymax, xmax = id_box["box_2d"]
        py_min, px_min = int((ymin / 1000) * height), int((xmin / 1000) * width)
        py_max, px_max = int((ymax / 1000) * height), int((xmax / 1000) * width)
        
        # Cover box with white pixels
        img_np = np.array(card_image).copy()
        cv2.rectangle(img_np, (px_min, py_min), (px_max, py_max), (220, 235, 245), -1) # card background color
        
        # Draw altered ID number
        fake_id = f"{random.randint(100, 999)}-99-9999" # Tampered digit structure
        cv2.putText(img_np, fake_id, (px_min + 5, py_max - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1) # Red text
        
        metadata_copy = metadata.copy()
        for box in metadata_copy["layout_boxes"]:
            if box["label"] == "ID_Number":
                box["text"] = fake_id
                
        metadata_copy["text_altered"] = True
        return Image.fromarray(img_np), metadata_copy, [py_min, px_min, py_max, px_max]

    def augment_pipeline(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
        img = image.copy()
        meta = metadata.copy()
        
        info = {
            "is_tampered": False,
            "tamper_box": [0, 0, 0, 0],
            "tamper_type": "none",
            "is_distorted": False
        }
        
        # Decide tampering type
        r = random.uniform(0, 1)
        if r < self.config["tampering"]["face_swap_probability"]:
            img, meta, t_box = self.apply_face_swap_attack(img, meta)
            info["is_tampered"] = True
            info["tamper_box"] = t_box
            info["tamper_type"] = "face_swap"
        elif r < (self.config["tampering"]["face_swap_probability"] + self.config["tampering"]["text_replacement_probability"]):
            img, meta, t_box = self.apply_text_replacement(img, meta)
            info["is_tampered"] = True
            info["tamper_box"] = t_box
            info["tamper_type"] = "text_alteration"
            
        # Geometric warps
        if random.uniform(0, 1) > 0.5:
            img, meta = self.apply_perspective_warp(img, meta)
            info["is_distorted"] = True
            
        # Degradations
        if random.uniform(0, 1) > 0.5:
            img = self.add_blur(img)
        if random.uniform(0, 1) > 0.5:
            img = self.add_noise(img)
            
        return img, meta, info
