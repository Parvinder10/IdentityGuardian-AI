from typing import List, Dict, Any, Tuple
import numpy as np

class LayoutAwareSemanticChunker:
    """
    Groups OCR/VLM layout blocks into semantically and spatially coherent chunks.
    Avoids splitting tables, headers, and signatures by analyzing 2D coordinates.
    """
    def __init__(self, vertical_threshold_px: int = 40, horizontal_threshold_px: int = 150):
        self.vertical_threshold = vertical_threshold_px
        self.horizontal_threshold = horizontal_threshold_px

    def _get_box_center(self, box: List[int]) -> Tuple[float, float]:
        # Bounding box: [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = box
        return (ymin + ymax) / 2.0, (xmin + xmax) / 2.0

    def chunk_document(self, layout_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups visual document layout elements into semantic paragraphs and tables.
        Uses visual distances to merge adjacent text/table coordinates.
        """
        if not layout_boxes:
            return []

        # Sort layout elements top-to-bottom, then left-to-right
        sorted_elements = sorted(layout_boxes, key=lambda x: (x["box_2d"][0], x["box_2d"][1]))
        
        chunks = []
        current_chunk = [sorted_elements[0]]
        
        for next_elem in sorted_elements[1:]:
            last_elem = current_chunk[-1]
            
            # Extract coordinates
            ly_min, lx_min, ly_max, lx_max = last_elem["box_2d"]
            ny_min, nx_min, ny_max, nx_max = next_elem["box_2d"]
            
            # Visual layout group rules:
            # 1. Merge if the elements are of table type and close vertically
            # 2. Merge if elements are of text type and lie within vertical line-height threshold
            is_same_class = (last_elem.get("label") == next_elem.get("label"))
            vertical_gap = ny_min - ly_max
            horizontal_overlap = not (nx_max < lx_min or nx_min > lx_max)
            
            if is_same_class and (vertical_gap < self.vertical_threshold) and horizontal_overlap:
                current_chunk.append(next_elem)
            elif last_elem.get("label") == "table" and next_elem.get("label") == "table" and (vertical_gap < self.vertical_threshold * 2):
                # Keep tables intact
                current_chunk.append(next_elem)
            else:
                # Flush current chunk
                chunks.append(self._compile_chunk(current_chunk))
                current_chunk = [next_elem]
                
        # Flush the final chunk
        chunks.append(self._compile_chunk(current_chunk))
        return chunks

    def _compile_chunk(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combines grouped elements into a single chunk structure with an outer bounding box."""
        texts = []
        ymins, xmins, ymaxs, xmaxs = [], [], [], []
        
        for el in elements:
            texts.append(el.get("text", ""))
            ymin, xmin, ymax, xmax = el["box_2d"]
            ymins.append(ymin)
            xmins.append(xmin)
            ymaxs.append(ymax)
            xmaxs.append(xmax)
            
        combined_text = "\n".join(texts)
        outer_box = [min(ymins), min(xmins), max(ymaxs), max(xmaxs)]
        
        # Primary label of the chunk (mode of element classes)
        labels = [el.get("label", "text") for el in elements]
        primary_label = max(set(labels), key=labels.count)
        
        return {
            "text": combined_text,
            "box_2d": outer_box,
            "label": primary_label,
            "element_count": len(elements)
        }
