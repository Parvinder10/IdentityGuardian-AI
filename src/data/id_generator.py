import random
import os
import json
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw

class IdentityCardGenerator:
    """
    Generates synthetic industrial-grade ID cards (driver licenses, national IDs)
    complete with profile photos, structural text fields, layout bounding boxes, and ground-truth metadata.
    """
    def __init__(self, width: int = 600, height: int = 400, seed: int = 42):
        self.width = width
        self.height = height
        random.seed(seed)

    def _create_profile_face(self, eye_color: Tuple[int, int, int] = None) -> Image.Image:
        """Generates a simple, procedurally drawn cartoon face to act as the profile picture."""
        face = Image.new("RGB", (150, 180), (240, 240, 240))
        draw = ImageDraw.Draw(face)
        
        if eye_color is None:
            eye_color = (random.randint(10, 100), random.randint(10, 100), random.randint(150, 255))
            
        # Draw face oval
        draw.ellipse([20, 20, 130, 160], fill=(255, 220, 180), outline=(0, 0, 0), width=2)
        # Eyes
        draw.ellipse([45, 65, 65, 85], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.ellipse([50, 70, 60, 80], fill=eye_color) # Pupil
        
        draw.ellipse([85, 65, 105, 85], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.ellipse([90, 70, 100, 80], fill=eye_color) # Pupil
        
        # Nose
        draw.polygon([(75, 90), (70, 115), (80, 115)], fill=(240, 180, 140), outline=(0,0,0))
        # Mouth
        draw.arc([50, 120, 100, 145], start=0, end=180, fill=(200, 50, 50), width=3)
        return face

    def generate_id_card(self) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
        """
        Creates an ID Card.
        Returns:
            - card_image (PIL.Image)
            - raw_profile_photo (PIL.Image) - used as the baseline for selfie matching
            - metadata (Dict) containing boxes and labels
        """
        # Create base canvas with card layout decoration
        card = Image.new("RGB", (self.width, self.height), (220, 235, 245)) # light blue card base
        draw = ImageDraw.Draw(card)
        
        # Draw header banner
        draw.rectangle([0, 0, self.width, 60], fill=(20, 80, 140))
        draw.text((20, 15), "IDENTITY CARD - GUARDIAN STATE", fill=(255, 255, 255))
        
        # Draw decorative background watermarks
        draw.arc([100, 100, 500, 500], start=0, end=360, fill=(200, 220, 235), width=4)
        
        layout_boxes = []
        ground_truth = {}
        
        # 1. Profile Photo Crop Slot
        profile_x, profile_y = 30, 100
        profile_w, profile_h = 130, 160
        raw_face = self._create_profile_face()
        resized_face = raw_face.resize((profile_w, profile_h))
        card.paste(resized_face, (profile_x, profile_y))
        
        layout_boxes.append({
            "box_2d": [profile_y, profile_x, profile_y + profile_h, profile_x + profile_w],
            "label": "Profile_Photo"
        })

        # 2. Text Fields
        fields_start_x = 200
        
        # Name Field
        name = f"{random.choice(['John', 'Alice', 'Michael', 'Emily', 'Sarah'])} {random.choice(['Smith', 'Doe', 'Johnson', 'Miller', 'Brown'])}"
        draw.text((fields_start_x, 100), "NAME:", fill=(100, 100, 100))
        draw.text((fields_start_x, 118), name, fill=(0, 0, 0))
        layout_boxes.append({
            "box_2d": [115, fields_start_x, 138, fields_start_x + 300],
            "label": "Name",
            "text": name
        })
        ground_truth["name"] = name

        # DOB Field
        dob = f"19{random.randint(60, 99)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        draw.text((fields_start_x, 170), "DATE OF BIRTH:", fill=(100, 100, 100))
        draw.text((fields_start_x, 188), dob, fill=(0, 0, 0))
        layout_boxes.append({
            "box_2d": [185, fields_start_x, 208, fields_start_x + 200],
            "label": "DOB",
            "text": dob
        })
        ground_truth["dob"] = dob

        # ID Number Field
        id_num = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        draw.text((fields_start_x, 240), "ID NUMBER:", fill=(100, 100, 100))
        draw.text((fields_start_x, 258), id_num, fill=(0, 0, 0))
        layout_boxes.append({
            "box_2d": [255, fields_start_x, 278, fields_start_x + 250],
            "label": "ID_Number",
            "text": id_num
        })
        ground_truth["id_number"] = id_num

        # Bottom validation signature seal
        draw.ellipse([480, 280, 560, 360], outline=(200, 50, 50), width=2)
        draw.text((495, 310), "ISSUED", fill=(200, 50, 50))
        
        # Convert layout boxes to normalized range [0, 1000]
        for item in layout_boxes:
            box = item["box_2d"]
            item["box_2d"] = [
                int(box[0] / self.height * 1000),
                int(box[1] / self.width * 1000),
                int(box[2] / self.height * 1000),
                int(box[3] / self.width * 1000)
            ]
            
        metadata = {
            "type": "national_id",
            "layout_boxes": layout_boxes,
            "ground_truth": ground_truth
        }
        
        return card, raw_face, metadata

    def generate_dataset(self, output_dir: str, num_samples: int = 50) -> List[Tuple[str, str, str]]:
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i in range(num_samples):
            card, face, meta = self.generate_id_card()
            card_path = os.path.join(output_dir, f"id_{i:04d}.png")
            face_path = os.path.join(output_dir, f"id_{i:04d}_face.png")
            meta_path = os.path.join(output_dir, f"id_{i:04d}.json")
            
            card.save(card_path)
            face.save(face_path)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            paths.append((card_path, face_path, meta_path))
        return paths
