from typing import List, Dict, Any, Tuple
import numpy as np

class IDLayoutSpatialChunker:
    """
    Groups visual elements on identity cards (Name labels, numbers, stamps)
    into unified logical chunks based on 2D box centers and overlap coordinates.
    """
    def __init__(self, x_merge_threshold: int = 200, y_merge_threshold: int = 30):
        self.x_merge_threshold = x_merge_threshold
        self.y_merge_threshold = y_merge_threshold

    def chunk_id_fields(self, layout_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups label cards and value strings based on visual layout bounds.
        """
        if not layout_boxes:
            return []
            
        # Sort layout elements top-to-bottom, then left-to-right
        sorted_elems = sorted(layout_boxes, key=lambda x: (x["box_2d"][0], x["box_2d"][1]))
        
        chunks = []
        visited = set()
        
        for i, elem_a in enumerate(sorted_elems):
            if i in visited:
                continue
            
            current_group = [elem_a]
            visited.add(i)
            
            box_a = elem_a["box_2d"]
            
            for j, elem_b in enumerate(sorted_elems):
                if j in visited:
                    continue
                box_b = elem_b["box_2d"]
                
                # Check horizontal matching (e.g. Label value placed side-by-side)
                y_overlap = not (box_b[2] < box_a[0] or box_b[0] > box_a[2])
                x_distance = box_b[1] - box_a[3]
                
                # Check vertical matching (e.g. Label value placed stacked)
                x_overlap = not (box_b[3] < box_a[1] or box_b[1] > box_a[3])
                y_distance = box_b[0] - box_a[2]
                
                if y_overlap and (0 <= x_distance < self.x_merge_threshold):
                    current_group.append(elem_b)
                    visited.add(j)
                elif x_overlap and (0 <= y_distance < self.y_merge_threshold):
                    current_group.append(elem_b)
                    visited.add(j)
                    
            chunks.append(self._compile_chunk(current_group))
            
        return chunks

    def _compile_chunk(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = []
        ymins, xmins, ymaxs, xmaxs = [], [], [], []
        
        for item in group:
            if "text" in item:
                texts.append(item["text"])
            else:
                texts.append(item.get("label", ""))
                
            box = item["box_2d"]
            ymins.append(box[0])
            xmins.append(box[1])
            ymaxs.append(box[2])
            xmaxs.append(box[3])
            
        combined_text = ": ".join(texts)
        outer_box = [min(ymins), min(xmins), max(ymaxs), max(xmaxs)]
        labels = [item.get("label", "text") for item in group]
        primary_label = max(set(labels), key=labels.count)
        
        return {
            "text": combined_text,
            "box_2d": outer_box,
            "label": primary_label,
            "group_size": len(group)
        }
