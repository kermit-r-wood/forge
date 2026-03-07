"""
3MF 导出模块
"""
import numpy as np
import zipfile
from xml.sax.saxutils import escape


class Exporter:
    """3MF 导出器"""
    
    def __init__(self):
        pass
        
    def export(self, file_path: str, layer_data: np.ndarray, materials: list[dict], 
               pixel_size_mm: float = 0.4, layer_height_mm: float = 0.08,
               rgb_image: np.ndarray = None, base_thickness_mm: float = 0.0,
               invert_z: bool = False):
        """
        导出 3MF 文件
        
        Args:
            file_path: Output 3MF file path
            layer_data: (H, W, Layers) material index array
            materials: List of material definitions
            pixel_size_mm: Horizontal pixel size in mm
            layer_height_mm: Height per color layer in mm
            rgb_image: Optional RGB image for vertex colors
            base_thickness_mm: Thickness of solid base layer (0 = no base)
            invert_z: If True, reverses Z-axis orientation
        """
        # 1. Generate 3D Model XML
        model_xml, object_ids = self._generate_model_xml(
            layer_data, materials, pixel_size_mm, layer_height_mm, rgb_image,
            base_thickness_mm, invert_z
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

    def _generate_base_layer(self, H: int, W: int, pixel_size_mm: float, 
                             base_thickness_mm: float, base_color: tuple,
                             start_z: float = 0.0) -> tuple:
        """
        Generate a solid rectangular base layer.
        
        Args:
            start_z: Starting Z height for the base
        """
        x_max = W * pixel_size_mm
        y_max = H * pixel_size_mm
        z_bottom = start_z
        z_top = start_z + base_thickness_mm
        
        r, g, b = base_color
        
        # Single rectangular block covering the entire base
        coords = np.array([
            [0, 0, z_bottom],
            [x_max, 0, z_bottom],
            [x_max, y_max, z_bottom],
            [0, y_max, z_bottom],
            [0, 0, z_top],
            [x_max, 0, z_top],
            [x_max, y_max, z_top],
            [0, y_max, z_top],
        ], dtype=np.float64)

        
        # 6 faces as quads
        faces = np.array([
            [0, 3, 2, 1],  # Bottom (-Z)
            [4, 5, 6, 7],  # Top (+Z)
            [0, 1, 5, 4],  # Front (-Y)
            [2, 3, 7, 6],  # Back (+Y)
            [0, 4, 7, 3],  # Left (-X)
            [1, 2, 6, 5],  # Right (+X)
        ], dtype=np.int32)
        
        colors = np.tile([[r, g, b]], (8, 1)).astype(np.uint8)
        
        return coords, faces, colors
    
    def _vertices_to_xml(self, coords: np.ndarray, colors: np.ndarray) -> str:
        """Convert vertex coordinates and colors to XML string."""
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
        
        return "".join(v_lines)
    
    def _faces_to_xml(self, faces: np.ndarray) -> str:
        """Convert quad faces to triangle XML string."""
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
        
        return "".join(t_str)

    def _generate_model_xml(self, layer_data: np.ndarray, materials: list[dict], 
                                       pixel_size_mm: float, layer_height_mm: float,
                                       rgb_image: np.ndarray = None,
                                       base_thickness_mm: float = 0.0,
                                       invert_z: bool = False) -> tuple[str, list[int]]:
        """
        Generate 3MF model XML with per-pixel cubes.
        
        Creates individual cubes for each pixel at each layer, without merging.
        This produces larger file sizes but may be more compatible with slicers.
        
        Returns:
            Tuple of (model_xml_string, object_ids_list)
        """
        H, W, num_layers = layer_data.shape
        
        # Build color lookup from materials
        material_colors = [self._get_material_color(m) for m in materials]
        num_materials = len(materials)
        
        # Prepare per-material vertex/face storage
        mat_vertices = [[] for _ in range(num_materials)]
        mat_faces = [[] for _ in range(num_materials)]
        mat_colors = [[] for _ in range(num_materials)]
        
        # Calculate base offset
        if invert_z:
            color_layer_z_start = 0.0
        else:
            color_layer_z_start = base_thickness_mm
        
        # Standard cube definition (unit cube at origin)
        cube_verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ], dtype=float)
        
        # Faces as quads (v0, v1, v2, v3)
        cube_faces = np.array([
            [0, 1, 2, 3],  # bottom (-Z)
            [4, 7, 6, 5],  # top (+Z)
            [0, 4, 5, 1],  # front (-Y)
            [2, 6, 7, 3],  # back (+Y)
            [0, 3, 7, 4],  # left (-X)
            [1, 5, 6, 2]   # right (+X)
        ])
        
        # Iterate through each pixel and layer
        for y in range(H):
            for x in range(W):
                # Pixel position in mm
                # Flip both X and Y to correct image orientation
                px = (W - 1 - x) * pixel_size_mm  # Flip X axis
                py = (H - 1 - y) * pixel_size_mm  # Flip Y axis
                
                for z in range(num_layers):
                    m_idx = int(layer_data[y, x, z])
                    if m_idx < 0 or m_idx >= num_materials:
                        continue
                    
                    # Calculate Z position
                    # Fix: Match physical print layers to optical model assumption
                    # K-M optics assume combo[0] is closest to the viewer.
                    if invert_z:
                        # Face Down: Viewer sees Z=0 first -> combo[0] must be at Z=0
                        pz = z * layer_height_mm + color_layer_z_start
                    else:
                        # Face Up: Viewer sees highest Z first -> combo[0] must be at highest Z
                        pz = (num_layers - 1 - z) * layer_height_mm + color_layer_z_start
                    
                    # Scale and translate cube vertices
                    scaled_verts = cube_verts.copy()
                    scaled_verts[:, 0] = scaled_verts[:, 0] * pixel_size_mm + px
                    scaled_verts[:, 1] = scaled_verts[:, 1] * pixel_size_mm + py
                    scaled_verts[:, 2] = scaled_verts[:, 2] * layer_height_mm + pz
                    
                    # Get vertex offset for this material
                    v_offset = len(mat_vertices[m_idx])
                    
                    # Add vertices
                    mat_vertices[m_idx].extend(scaled_verts.tolist())
                    
                    # Add faces with offset
                    for face in cube_faces:
                        mat_faces[m_idx].append([f + v_offset for f in face])
                    
                    # Add colors (material color for each vertex)
                    color = material_colors[m_idx]
                    for _ in range(8):
                        mat_colors[m_idx].append(color)
        
        # Build XML
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<model unit="millimeter">',
            '<resources>'
        ]
        
        object_ids = []
        current_id = 1
        
        # Generate base layer as separate object
        if base_thickness_mm > 0:
            base_z_start = (num_layers * layer_height_mm) if invert_z else 0.0
            
            base_coords, base_faces_arr, base_colors_arr = self._generate_base_layer(
                H, W, pixel_size_mm, base_thickness_mm, material_colors[0] if materials else (255, 255, 255),
                start_z=base_z_start
            )
            if len(base_coords) > 0:
                obj_id = current_id
                current_id += 1
                object_ids.append(obj_id)
                
                xml_parts.append(f'<object id="{obj_id}" name="base(白)" type="model">')
                xml_parts.append('<mesh>')
                xml_parts.append('<vertices>')
                xml_parts.append(self._vertices_to_xml(base_coords, base_colors_arr))
                xml_parts.append('</vertices>')
                xml_parts.append('<triangles>')
                xml_parts.append(self._faces_to_xml(base_faces_arr))
                xml_parts.append('</triangles>')
                xml_parts.append('</mesh>')
                xml_parts.append('</object>')
        
        # Generate objects for each material
        for m_idx in range(num_materials):
            if len(mat_vertices[m_idx]) == 0:
                continue
            
            obj_id = current_id
            current_id += 1
            object_ids.append(obj_id)
            
            mat_name = materials[m_idx].get('name', f'Material_{m_idx}')
            
            xml_parts.append(f'<object id="{obj_id}" name="{mat_name}" type="model">')
            xml_parts.append('<mesh>')
            xml_parts.append('<vertices>')
            
            # Convert to numpy for efficient processing
            coords = np.array(mat_vertices[m_idx])
            colors = np.array(mat_colors[m_idx])
            faces = np.array(mat_faces[m_idx])
            
            xml_parts.append(self._vertices_to_xml(coords, colors))
            xml_parts.append('</vertices>')
            xml_parts.append('<triangles>')
            xml_parts.append(self._faces_to_xml(faces))
            xml_parts.append('</triangles>')
            xml_parts.append('</mesh>')
            xml_parts.append('</object>')
        
        # Close resources and build
        xml_parts.append('</resources>')
        xml_parts.append('<build>')
        for oid in object_ids:
            xml_parts.append(f'<item objectid="{oid}" />')
        xml_parts.append('</build>')
        xml_parts.append('</model>')
        
        return "".join(xml_parts), object_ids

