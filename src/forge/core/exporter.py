"""
3MF 导出模块 (Greedy Meshing Implementation)
Uses greedy meshing algorithm to merge adjacent voxels into larger rectangles,
dramatically reducing mesh complexity and improving print quality.
"""
import numpy as np
import zipfile
from xml.sax.saxutils import escape


class Exporter:
    """3MF 导出器 - 贪婪网格合并"""
    
    def __init__(self):
        pass
        
    def export(self, file_path: str, layer_data: np.ndarray, materials: list[dict], 
               pixel_size_mm: float = 0.4, layer_height_mm: float = 0.08,
               rgb_image: np.ndarray = None):
        """
        导出 3MF 文件
        
        Uses greedy meshing to merge adjacent same-material voxels into larger
        rectangles, reducing mesh complexity and creating continuous print paths.
        """
        # 1. Generate 3D Model XML with greedy meshing
        model_xml = self._generate_model_xml_greedy(
            layer_data, materials, pixel_size_mm, layer_height_mm, rgb_image
        )
        
        # 2. Generate [Content_Types].xml
        content_types_xml = self._generate_content_types()
        
        # 3. Generate _rels/.rels
        rels_xml = self._generate_rels()
        
        # 4. Write Zip
        with zipfile.ZipFile(file_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('[Content_Types].xml', content_types_xml)
            zf.writestr('_rels/.rels', rels_xml)
            zf.writestr('3D/3dmodel.model', model_xml)

    def _generate_content_types(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

    def _generate_rels(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""

    def _get_material_color(self, mat: dict) -> tuple:
        """Extract RGB tuple from material color definition"""
        if isinstance(mat['color'], str):
            hex_col = mat['color'].lstrip('#')
            r = int(hex_col[0:2], 16)
            g = int(hex_col[2:4], 16)
            b = int(hex_col[4:6], 16)
            return (r, g, b)
        else:
            return tuple(mat['color'][:3])

    def _greedy_mesh_2d(self, mask: np.ndarray) -> list:
        """
        Greedy meshing algorithm for 2D binary mask.
        
        Merges adjacent True pixels into maximal rectangles.
        Returns list of rectangles: [(x, y, width, height), ...]
        """
        H, W = mask.shape
        processed = np.zeros_like(mask, dtype=bool)
        rectangles = []
        
        for y in range(H):
            for x in range(W):
                if not mask[y, x] or processed[y, x]:
                    continue
                
                # Found an unprocessed pixel, expand to maximal rectangle
                # First, expand right as far as possible
                w = 1
                while x + w < W and mask[y, x + w] and not processed[y, x + w]:
                    w += 1
                
                # Then, expand down as far as possible while maintaining width
                h = 1
                while y + h < H:
                    # Check if entire row can be extended
                    can_extend = True
                    for dx in range(w):
                        if not mask[y + h, x + dx] or processed[y + h, x + dx]:
                            can_extend = False
                            break
                    if can_extend:
                        h += 1
                    else:
                        break
                
                # Mark all pixels in this rectangle as processed
                for dy in range(h):
                    for dx in range(w):
                        processed[y + dy, x + dx] = True
                
                rectangles.append((x, y, w, h))
        
        return rectangles

    def _generate_model_xml_greedy(self, layer_data: np.ndarray, materials: list[dict], 
                                    pixel_size_mm: float, layer_height_mm: float,
                                    rgb_image: np.ndarray = None) -> str:
        """
        Generate 3MF model XML using greedy meshing.
        
        For each material, merges adjacent voxels into larger rectangular blocks.
        """
        H, W, num_layers = layer_data.shape
        
        # Build color lookup from materials
        material_colors = [self._get_material_color(m) for m in materials]
        
        # If RGB image provided and matches dimensions, use it for vertex colors
        if rgb_image is not None and rgb_image.shape[0] == H and rgb_image.shape[1] == W:
            use_rgb_image = True
        else:
            use_rgb_image = False
        
        # Build XML header
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<model unit="millimeter">',
            '<resources>'
        ]
        
        object_ids = []
        current_id = 1
        
        # Create one object per material
        for m_idx, mat in enumerate(materials):
            coords, faces, colors = self._generate_greedy_mesh_for_material(
                layer_data, m_idx, material_colors, pixel_size_mm, layer_height_mm,
                rgb_image if use_rgb_image else None
            )
            
            if len(coords) == 0:
                continue
            
            obj_id = current_id
            current_id += 1
            object_ids.append(obj_id)
            
            mat_name = escape(mat['name'])
            
            xml_parts.append(f'<object id="{obj_id}" name="{mat_name}" type="model">')
            xml_parts.append('<mesh>')
            xml_parts.append('<vertices>')
            
            # Build vertex strings with RGB colors
            vx = coords[:, 0]
            vy = coords[:, 1]
            vz = coords[:, 2]
            vr = colors[:, 0].astype(int)
            vg = colors[:, 1].astype(int)
            vb = colors[:, 2].astype(int)
            
            s_vx = np.char.mod('%.3f', vx)
            s_vy = np.char.mod('%.3f', vy)
            s_vz = np.char.mod('%.3f', vz)
            s_vr = np.char.mod('%d', vr)
            s_vg = np.char.mod('%d', vg)
            s_vb = np.char.mod('%d', vb)
            
            v_lines = np.char.add('<vertex x="', s_vx)
            v_lines = np.char.add(v_lines, '" y="')
            v_lines = np.char.add(v_lines, s_vy)
            v_lines = np.char.add(v_lines, '" z="')
            v_lines = np.char.add(v_lines, s_vz)
            v_lines = np.char.add(v_lines, '" r="')
            v_lines = np.char.add(v_lines, s_vr)
            v_lines = np.char.add(v_lines, '" g="')
            v_lines = np.char.add(v_lines, s_vg)
            v_lines = np.char.add(v_lines, '" b="')
            v_lines = np.char.add(v_lines, s_vb)
            v_lines = np.char.add(v_lines, '" />')
            
            xml_parts.append("".join(v_lines))
            xml_parts.append('</vertices>')
            
            xml_parts.append('<triangles>')
            
            # Convert quads to triangles
            i0 = faces[:, 0]
            i1 = faces[:, 1]
            i2 = faces[:, 2]
            i3 = faces[:, 3]
            
            t1 = np.stack([i0, i2, i1], axis=1)
            t2 = np.stack([i0, i3, i2], axis=1)
            tris = np.vstack([t1, t2])
            
            t_v1 = np.char.mod('%d', tris[:, 0])
            t_v2 = np.char.mod('%d', tris[:, 1])
            t_v3 = np.char.mod('%d', tris[:, 2])
            
            t_str = np.char.add('<triangle v1="', t_v1)
            t_str = np.char.add(t_str, '" v2="')
            t_str = np.char.add(t_str, t_v2)
            t_str = np.char.add(t_str, '" v3="')
            t_str = np.char.add(t_str, t_v3)
            t_str = np.char.add(t_str, '" />')
            
            xml_parts.append("".join(t_str))
            
            xml_parts.append('</triangles>')
            xml_parts.append('</mesh>')
            xml_parts.append('</object>')
        
        xml_parts.append('</resources>')
        
        xml_parts.append('<build>')
        for oid in object_ids:
            xml_parts.append(f'<item objectid="{oid}"/>')
        xml_parts.append('</build>')
        
        xml_parts.append('</model>')
        
        return "".join(xml_parts)

    def _generate_greedy_mesh_for_material(self, layer_data: np.ndarray, 
                                            m_idx: int,
                                            material_colors: list,
                                            pixel_size_mm: float, 
                                            layer_height_mm: float,
                                            rgb_image: np.ndarray = None) -> tuple:
        """
        Generate mesh for a material using greedy meshing.
        
        For each layer, finds the 2D mask of pixels using this material,
        applies greedy meshing to merge into rectangles, then creates
        3D blocks for each merged rectangle.
        """
        H, W, num_layers = layer_data.shape
        
        all_coords = []
        all_faces = []
        all_colors = []
        vertex_offset = 0
        
        mat_r, mat_g, mat_b = material_colors[m_idx] if m_idx < len(material_colors) else (255, 255, 255)
        
        for z in range(num_layers):
            # Get 2D mask for this material at this layer
            mask = (layer_data[:, :, z] == m_idx)
            
            if not np.any(mask):
                continue
            
            # Apply greedy meshing to get merged rectangles
            rectangles = self._greedy_mesh_2d(mask)
            
            z_base = z * layer_height_mm
            z_top = (z + 1) * layer_height_mm
            
            for (rx, ry, rw, rh) in rectangles:
                # Convert pixel coordinates to mm
                x_min = rx * pixel_size_mm
                y_min = ry * pixel_size_mm
                x_max = (rx + rw) * pixel_size_mm
                y_max = (ry + rh) * pixel_size_mm
                
                # Determine color - use center pixel of rectangle
                center_y = ry + rh // 2
                center_x = rx + rw // 2
                if rgb_image is not None:
                    r, g, b = rgb_image[center_y, center_x]
                else:
                    r, g, b = mat_r, mat_g, mat_b
                
                # 8 vertices of the rectangular block
                block_verts = np.array([
                    [x_min, y_min, z_base],
                    [x_max, y_min, z_base],
                    [x_max, y_max, z_base],
                    [x_min, y_max, z_base],
                    [x_min, y_min, z_top],
                    [x_max, y_min, z_top],
                    [x_max, y_max, z_top],
                    [x_min, y_max, z_top],
                ], dtype=np.float64)
                
                # 6 faces of the block (as quads)
                block_faces = np.array([
                    [0, 3, 2, 1],  # Bottom (-Z)
                    [4, 5, 6, 7],  # Top (+Z)
                    [0, 1, 5, 4],  # Front (-Y)
                    [2, 3, 7, 6],  # Back (+Y)
                    [0, 4, 7, 3],  # Left (-X)
                    [1, 2, 6, 5],  # Right (+X)
                ], dtype=np.int32) + vertex_offset
                
                block_colors = np.tile([[r, g, b]], (8, 1)).astype(np.uint8)
                
                all_coords.append(block_verts)
                all_faces.append(block_faces)
                all_colors.append(block_colors)
                vertex_offset += 8
        
        if not all_coords:
            return np.array([]), np.array([]), np.array([])
        
        coords = np.vstack(all_coords)
        faces = np.vstack(all_faces)
        colors = np.vstack(all_colors)
        
        return coords, faces, colors
