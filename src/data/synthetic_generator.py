import random
import os
import json
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont

class DocumentGenerator:
    """
    Generates synthetic industrial-grade documents (Invoices, IDs, Certificates)
    with layout bounding boxes, OCR transcripts, and structural schemas.
    """
    def __init__(self, width: int = 800, height: int = 1000, seed: int = 42):
        self.width = width
        self.height = height
        random.seed(seed)
        
        # Load fallback fonts
        self.font_family = "Arial"
        
    def _create_base_canvas(self, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("RGB", (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(image)
        return image, draw

    def generate_invoice(self) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates a synthetic invoice with text elements, tables, signature boxes,
        returning the image and a JSON containing coordinates and labels.
        """
        image, draw = self._create_base_canvas()
        
        # Metadata storage
        layout_boxes = []
        ocr_tokens = []
        ground_truth = {}

        # 1. Header Area
        vendor_name = f"VENDOR: {random.choice(['TechCorp Solutions', 'Global Logistics Inc', 'Apex Financials', 'HyperVerge Systems'])}"
        draw.text((50, 50), vendor_name, fill=(0, 0, 0))
        # Bounding box format: [ymin, xmin, ymax, xmax] normalized to [0, 1000]
        layout_boxes.append({
            "box_2d": [50, 50, 80, 450],
            "label": "header",
            "text": vendor_name
        })
        ground_truth["vendor_name"] = vendor_name.replace("VENDOR: ", "")

        # Invoice ID and Date
        inv_id = f"INV-{random.randint(100000, 999999)}"
        inv_date = f"2026-07-{random.randint(10, 28):02d}"
        draw.text((550, 50), f"Invoice ID: {inv_id}", fill=(0, 0, 0))
        layout_boxes.append({
            "box_2d": [50, 550, 75, 750],
            "label": "text",
            "text": inv_id
        })
        draw.text((550, 75), f"Date: {inv_date}", fill=(0, 0, 0))
        layout_boxes.append({
            "box_2d": [75, 550, 100, 750],
            "label": "text",
            "text": inv_date
        })
        ground_truth["invoice_id"] = inv_id
        ground_truth["date"] = inv_date

        # 2. Table Area (Grid and contents)
        table_top = 200
        table_bottom = 500
        # Draw table border
        draw.rectangle([50, table_top, 750, table_bottom], outline=(0, 0, 0), width=2)
        # Column headers
        draw.line([50, table_top + 40, 750, table_top + 40], fill=(0, 0, 0), width=2)
        draw.text((70, table_top + 10), "Item Description", fill=(0, 0, 0))
        draw.text((450, table_top + 10), "Qty", fill=(0, 0, 0))
        draw.text((550, table_top + 10), "Unit Price", fill=(0, 0, 0))
        draw.text((660, table_top + 10), "Total", fill=(0, 0, 0))
        
        # Bounding box for entire table
        layout_boxes.append({
            "box_2d": [table_top, 50, table_bottom, 750],
            "label": "table",
            "text": "Table containing itemized charges"
        })

        # Draw line items
        items = [
            ("Neural Training Compute API", random.randint(1, 5), random.randint(100, 500)),
            ("Consulting Services (AI Strategy)", random.randint(5, 20), random.randint(150, 300)),
            ("Edge Deployment Unit License", random.randint(1, 2), random.randint(1000, 2000)),
        ]
        
        current_y = table_top + 50
        grand_total = 0.0
        for desc, qty, price in items:
            line_total = qty * price
            grand_total += line_total
            
            draw.text((70, current_y), desc, fill=(0, 0, 0))
            draw.text((450, current_y), str(qty), fill=(0, 0, 0))
            draw.text((550, current_y), f"${price:.2f}", fill=(0, 0, 0))
            draw.text((660, current_y), f"${line_total:.2f}", fill=(0, 0, 0))
            
            # Row tokens
            ocr_tokens.append({"text": desc, "box": [current_y, 70, current_y + 15, 400]})
            ocr_tokens.append({"text": str(qty), "box": [current_y, 450, current_y + 15, 480]})
            
            current_y += 40
            draw.line([50, current_y, 750, current_y], fill=(180, 180, 180), width=1)
            current_y += 10
            
        # Draw Grand Total text
        total_text = f"GRAND TOTAL: ${grand_total:.2f}"
        draw.text((520, table_bottom + 30), total_text, fill=(0, 0, 0))
        layout_boxes.append({
            "box_2d": [table_bottom + 25, 520, table_bottom + 55, 750],
            "label": "text",
            "text": total_text
        })
        ground_truth["grand_total"] = grand_total

        # 3. Signature & Stamp Area
        sig_top = 700
        draw.rectangle([50, sig_top, 300, sig_top + 100], outline=(100, 100, 100), width=1)
        draw.text((60, sig_top + 5), "Authorized Signature", fill=(100, 100, 100))
        # Draw a simulated messy signature path
        sig_points = [(80, sig_top + 70), (120, sig_top + 40), (150, sig_top + 80), (220, sig_top + 30), (270, sig_top + 60)]
        draw.line(sig_points, fill=(10, 20, 150), width=2) # blue ink
        
        layout_boxes.append({
            "box_2d": [sig_top, 50, sig_top + 100, 300],
            "label": "signature",
            "text": "Signature"
        })

        # Save metadata details
        # For simplicity, convert bounding boxes to normalized coordinates in range 0 - 1000
        for item in layout_boxes:
            box = item["box_2d"]
            item["box_2d"] = [
                int(box[0] / self.height * 1000),
                int(box[1] / self.width * 1000),
                int(box[2] / self.height * 1000),
                int(box[3] / self.width * 1000)
            ]

        metadata = {
            "type": "invoice",
            "layout_boxes": layout_boxes,
            "ocr_tokens": ocr_tokens,
            "ground_truth": ground_truth
        }
        
        return image, metadata

    def generate_certificate(self) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates a synthetic achievement certificate with layout and metadata.
        """
        image, draw = self._create_base_canvas()
        layout_boxes = []
        ground_truth = {}
        
        # Certificate border
        draw.rectangle([20, 20, 780, 980], outline=(180, 140, 20), width=10) # gold border
        
        # Center elements
        cert_title = "CERTIFICATE OF COMPLIANCE"
        draw.text((250, 150), cert_title, fill=(180, 140, 20))
        layout_boxes.append({
            "box_2d": [140, 240, 180, 560],
            "label": "header",
            "text": cert_title
        })
        
        sub_text = "This document confirms that the verified entity conforms to"
        draw.text((200, 250), sub_text, fill=(50, 50, 50))
        
        org_name = f"Recipient: {random.choice(['Alpha Analytics Ltd', 'Delta Cyber LLC', 'Gamma Biotech', 'Zephyr Robotics'])}"
        draw.text((220, 350), org_name, fill=(0, 0, 0))
        layout_boxes.append({
            "box_2d": [330, 200, 380, 600],
            "label": "text",
            "text": org_name
        })
        ground_truth["recipient"] = org_name.replace("Recipient: ", "")
        
        # Stamp representation
        stamp_center = (600, 750)
        draw.ellipse([stamp_center[0]-50, stamp_center[1]-50, stamp_center[0]+50, stamp_center[1]+50], outline=(200, 30, 30), width=3)
        draw.text((stamp_center[0]-35, stamp_center[1]-10), "VERIFIED", fill=(200, 30, 30))
        
        layout_boxes.append({
            "box_2d": [700, 550, 800, 650],
            "label": "signature", # treating stamps as stamp/signature category
            "text": "Official Stamp"
        })
        
        # Signature line
        draw.line([100, 780, 300, 780], fill=(0, 0, 0), width=2)
        draw.text((120, 790), "Auditor Signature", fill=(100, 100, 100))
        # Draw mock signature
        draw.line([(120, 770), (160, 740), (200, 775), (250, 750), (290, 765)], fill=(0, 0, 0), width=2)
        
        layout_boxes.append({
            "box_2d": [730, 100, 810, 300],
            "label": "signature",
            "text": "Auditor Signature"
        })

        # Normalize layout coordinates
        for item in layout_boxes:
            box = item["box_2d"]
            item["box_2d"] = [
                int(box[0] / self.height * 1000),
                int(box[1] / self.width * 1000),
                int(box[2] / self.height * 1000),
                int(box[3] / self.width * 1000)
            ]
            
        metadata = {
            "type": "certificate",
            "layout_boxes": layout_boxes,
            "ocr_tokens": [],
            "ground_truth": ground_truth
        }
        
        return image, metadata

    def generate_dataset(self, output_dir: str, num_samples: int = 100) -> List[Tuple[str, str]]:
        """
        Creates synthetic images and writes their labels to JSON in the specified output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i in range(num_samples):
            doc_type = random.choice(["invoice", "certificate"])
            if doc_type == "invoice":
                img, meta = self.generate_invoice()
            else:
                img, meta = self.generate_certificate()
                
            img_path = os.path.join(output_dir, f"doc_{i:04d}.png")
            json_path = os.path.join(output_dir, f"doc_{i:04d}.json")
            
            img.save(img_path)
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
                
            paths.append((img_path, json_path))
        return paths

if __name__ == "__main__":
    generator = DocumentGenerator(seed=42)
    img, meta = generator.generate_invoice()
    img.save("sample_invoice.png")
    print("Generated sample invoice and metadata:", json.dumps(meta, indent=2))
