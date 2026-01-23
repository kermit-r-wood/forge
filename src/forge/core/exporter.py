"""
3MF 导出模块 (Native Implementation)
"""
import numpy as np
import zipfile
import uuid
import io
from xml.sax.saxutils import escape

class Exporter:
    """3MF 导出器"""
    
    def __init__(self):
        pass
        
    def export(self, file_path: str, layer_data: np.ndarray, materials: list[dict], 
               pixel_size_mm: float = 0.4, layer_height_mm: float = 0.08):
        """
        导出 3MF 文件
        :param layer_data: (H, W, Layers) uint8 矩阵，即 material_index
        :param materials: 材料定义列表
        """
        # 3MF Structure:
        # [Content_Types].xml
        # _rels/.rels
        # 3D/3dmodel.model
        
        # 1. Generate 3D Model XML
        model_xml = self._generate_model_xml(layer_data, materials, pixel_size_mm, layer_height_mm)
        
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

    def _generate_model_xml(self, layer_data: np.ndarray, materials: list[dict], 
                            pixel_size_mm: float, layer_height_mm: float) -> str:
        
        # Header
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel">',
            ' <metadata name="Application">Forge 3D</metadata>',
            ' <resources>'
        ]
        
        # --- Materials ---
        # Using BaseMaterials for simple color assignment
        mat_group_id = 1
        xml_parts.append(f'  <basematerials id="{mat_group_id}">')
        
        for mat in materials:
            if isinstance(mat['color'], str):
                hex_col = mat['color'].lstrip('#')
                if len(hex_col) == 6:
                    hex_col += "FF" # Add Alpha
            else:
                r, g, b = mat['color']
                hex_col = f"{r:02X}{g:02X}{b:02X}FF"
                
            xml_parts.append(f'   <base name="{mat["name"]}" displaycolor="#{hex_col}"/>')
            
        xml_parts.append('  </basematerials>')
        
        # --- Meshes ---
        # Generate one mesh object per material
        # Format:
        # <object id="..." type="model">
        #  <mesh>
        #   <vertices> ... </vertices>
        #   <triangles> ... </triangles>
        #  </mesh>
        # </object>
        
        object_ids = []
        current_id = 2
        
        # Precompute Cube Geometry
        # 8 vertices, 12 triangles
        # Vertices relative to (0,0,0)
        v_offsets = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), # Bottom 0-3
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)  # Top 4-7
        ]
        
        # Triangles (v1, v2, v3, p1) - p1 is property index (material index inside group)
        # However, for BaseMaterials, we assign the property at the Object level or Triangle level.
        # Since we are splitting objects by material, we can assign the property at the Object level?
        # 3MF Specification: <object pid="1" pindex="0"> applies to the whole object.
        
        tris_indices = [
            (0, 2, 1), (0, 3, 2), # Bottom
            (4, 5, 6), (4, 6, 7), # Top
            (0, 1, 5), (0, 5, 4), # Front
            (1, 2, 6), (1, 6, 5), # Right
            (2, 3, 7), (2, 7, 6), # Back
            (3, 0, 4), (3, 4, 7)  # Left
        ]
        
        for m_idx in range(len(materials)):
            positions = np.argwhere(layer_data == m_idx)
            if len(positions) == 0:
                continue
                
            obj_id = current_id
            current_id += 1
            object_ids.append(obj_id)
            
            # Object with Material Reference
            # pid = Material Group ID
            # pindex = Material Index (0-based)
            mat_name = escape(materials[m_idx]['name'])
            xml_parts.append(f'  <object id="{obj_id}" pid="{mat_group_id}" pindex="{m_idx}" name="{mat_name}" type="model">')
            xml_parts.append('   <mesh>')
            
            # -- Vertices --
            xml_parts.append('    <vertices>')
            
            # Optimization: Build larget string chunks
            # We have N cubes. Total vertices = 8 * N.
            # We simply list them all. Merging vertices is better but slower to implement now.
            
            vertex_buffer = io.StringIO()
            triangle_buffer = io.StringIO()
            
            vertex_count = 0
            
            for y, x, z in positions:
                bx, by, bz = x * pixel_size_mm, y * pixel_size_mm, z * layer_height_mm
                
                # Add 8 vertices
                for ox, oy, oz in v_offsets:
                    # x, y, z
                    vx = bx + ox * pixel_size_mm
                    vy = by + oy * pixel_size_mm
                    vz = bz + oz * layer_height_mm
                    vertex_buffer.write(f'<vertex x="{vx:.4f}" y="{vy:.4f}" z="{vz:.4f}" />')
                
                # Add 12 triangles
                base_v = vertex_count
                for v1, v2, v3 in tris_indices:
                    triangle_buffer.write(f'<triangle v1="{base_v+v1}" v2="{base_v+v2}" v3="{base_v+v3}" />')
                
                vertex_count += 8
            
            xml_parts.append(vertex_buffer.getvalue())
            xml_parts.append('    </vertices>')
            
            # -- Triangles --
            xml_parts.append('    <triangles>')
            xml_parts.append(triangle_buffer.getvalue())
            xml_parts.append('    </triangles>')
            
            xml_parts.append('   </mesh>')
            xml_parts.append('  </object>')
            
        xml_parts.append(' </resources>')
        
        # --- Build ---
        xml_parts.append(' <build>')
        for oid in object_ids:
            xml_parts.append(f'  <item objectid="{oid}"/>')
        xml_parts.append(' </build>')
        
        xml_parts.append('</model>')
        
        return "".join(xml_parts)
