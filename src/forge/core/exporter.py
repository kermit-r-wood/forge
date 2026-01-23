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
        mat_group_id = 1
        xml_parts.append(f'  <basematerials id="{mat_group_id}">')
        
        for mat in materials:
            if isinstance(mat['color'], str):
                hex_col = mat['color'].lstrip('#')
                if len(hex_col) == 6:
                    hex_col += "FF"
            else:
                r, g, b = mat['color']
                hex_col = f"{r:02X}{g:02X}{b:02X}FF"
                
            xml_parts.append(f'   <base name="{mat["name"]}" displaycolor="#{hex_col}"/>')
            
        xml_parts.append('  </basematerials>')
        
        # --- Meshes ---
        object_ids = []
        current_id = 2
        
        # Iterate over each material
        for m_idx in range(len(materials)):
            # Find all voxels of this material
            # Using vectorized face culling
            mesh_data = self._generate_optimized_mesh(layer_data, m_idx, pixel_size_mm, layer_height_mm)
            
            if not mesh_data['vertices']:
                continue
                
            obj_id = current_id
            current_id += 1
            object_ids.append(obj_id)
            
            mat_name = escape(materials[m_idx]['name'])
            xml_parts.append(f'  <object id="{obj_id}" pid="{mat_group_id}" pindex="{m_idx}" name="{mat_name}" type="model">')
            xml_parts.append('   <mesh>')
            
            # Vertices
            xml_parts.append('    <vertices>')
            xml_parts.append(mesh_data['vertices'])
            xml_parts.append('    </vertices>')
            
            # Triangles
            xml_parts.append('    <triangles>')
            xml_parts.append(mesh_data['triangles'])
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

    def _generate_optimized_mesh(self, layer_data: np.ndarray, m_idx: int, 
                               pixel_size_mm: float, layer_height_mm: float) -> dict:
        """
        Generates optimized mesh data for a specific material index.
        Returns dict with 'vertices' and 'triangles' strings.
        """
        # 1. Create a boolean mask for the current material
        # Pad with False to handle boundaries easily
        mask = np.pad(layer_data == m_idx, 1, mode='constant', constant_values=False)
        
        # 2. Identify exposed faces using boolean logic (Vectorized)
        # mask[y, x, z]
        # X axis: width (1st dim in our layer_data? No, layer_data is (H, W, Layers) -> (Y, X, Z))
        # Wait, previous code: for y, x, z in positions. positions = argwhere(layer_data == m_idx).
        # layer_data shape is (Height, Width, Layers).
        # So mask indices are [y+1, x+1, z+1] due to padding.
        
        # internal = padded[1:-1, 1:-1, 1:-1]
        
        # Faces pointing -X (Left)
        # Exists if Center is True AND Left Neighbor is False
        f_left = mask[1:-1, 1:-1, 1:-1] & ~mask[1:-1, :-2, 1:-1]
        
        # Faces pointing +X (Right)
        f_right = mask[1:-1, 1:-1, 1:-1] & ~mask[1:-1, 2:, 1:-1]
        
        # Faces pointing -Y (Top visually, but smaller Y coordinate)
        f_top = mask[1:-1, 1:-1, 1:-1] & ~mask[:-2, 1:-1, 1:-1]
        
        # Faces pointing +Y (Bottom)
        f_bottom = mask[1:-1, 1:-1, 1:-1] & ~mask[2:, 1:-1, 1:-1]
        
        # Faces pointing -Z (Lower layer)
        f_down = mask[1:-1, 1:-1, 1:-1] & ~mask[1:-1, 1:-1, :-2]
        
        # Faces pointing +Z (Upper layer)
        f_up = mask[1:-1, 1:-1, 1:-1] & ~mask[1:-1, 1:-1, 2:]
        
        # Get coordinates of faces
        # y, x, z
        
        # We need to construct vertices.
        # Strategy: To minimize vertex count, we could merge shared vertices, but that's complex (hashing).
        # Simpler optimization: Face Culling (what we did above).
        # We will generate 4 unique vertices per face. NO vertex sharing between faces to keep it simple and fast.
        # This is strictly better than 36 vertices per voxel (12 tris * 3).
        # Now it is 4 verts * N_exposed_faces.
        
        vertices_list = []
        triangles_list = []
        vertex_count = 0
        
        def add_faces(faces_mask, v_offsets, normal_axis):
            nonlocal vertex_count
            # faces_mask: (H, W, L) boolean
            # v_offsets: list of 4 tuples (dx, dy, dz) for the face corners relative to voxel origin (0,0,0)
            
            # Get indices (y, x, z)
            ys, xs, zs = np.nonzero(faces_mask)
            
            if len(ys) == 0:
                return
            
            n_faces = len(ys)
            
            # Calculate base coordinates
            # x -> pixel_size
            # y -> pixel_size
            # z -> layer_height
            
            # Vectorized coordinate calculation
            # shape (N, 1)
            x_base = xs * pixel_size_mm
            y_base = ys * pixel_size_mm
            z_base = zs * layer_height_mm
            
            # Generate 4 vertices for each face
            # Order: 0, 1, 2, 3 (quad) -> Tris: 0-1-2, 0-2-3 (or similar)
            
            verts_str_parts = []
            
            # For each of the 4 corners of the quad
            for i, (ox, oy, oz) in enumerate(v_offsets):
                vx = x_base + ox * pixel_size_mm
                vy = y_base + oy * pixel_size_mm
                vz = z_base + oz * layer_height_mm
                
                # We need to format these as XML strings efficiently.
                # Doing it in python loop is slow.
                # Use numpy string operations?
                # " <vertex x="{:.4f}" y="{:.4f}" z="{:.4f}" />"
                
                # Creating a structured array or simply iterating might be fast enough if we batch content
                # But python loop for 1M faces is still slow.
                
                # Much faster approach:
                # Create a big string array
                
                s_vx = np.char.mod('%.4f', vx)
                s_vy = np.char.mod('%.4f', vy)
                s_vz = np.char.mod('%.4f', vz)
                
                # Join columns
                # This is vectorized string formatting
                lines = np.char.add('<vertex x="', s_vx)
                lines = np.char.add(lines, '" y="')
                lines = np.char.add(lines, s_vy)
                lines = np.char.add(lines, '" z="')
                lines = np.char.add(lines, s_vz)
                lines = np.char.add(lines, '" />')
                
                vertices_list.append(lines)
                
            # Generate Triangles
            # 2 triangles per face.
            # Vertices indices:
            # v0 = base, v1 = base+N, v2 = base+2N, v3 = base+3N 
            # (Because we appended all corner 0s, then all corner 1s...)
            # WAIT: simpler to append v0..v3 per face?  No, bulk append is better.
            
            # Let's say we appended:
            # All P0s (0..N-1)
            # All P1s (N..2N-1)
            # All P2s (2N..3N-1)
            # All P3s (3N..4N-1)
            
            # Current base index
            base = vertex_count
            indices = np.arange(n_faces)
            
            i0 = base + indices
            i1 = base + n_faces + indices
            i2 = base + 2 * n_faces + indices
            i3 = base + 3 * n_faces + indices
            
            # Triangles: (0, 1, 2) and (0, 2, 3) 
            # Or (0, 2, 1) to flip normal? Need to check CCW winding.
            # Standard 3MF/STL is CCW for outside.
            
            # We need to construct string: <triangle v1="..." v2="..." v3="..." />
            
            def make_tris_str(idx_a, idx_b, idx_c):
                sa = np.char.mod('%d', idx_a)
                sb = np.char.mod('%d', idx_b)
                sc = np.char.mod('%d', idx_c)
                
                t = np.char.add('<triangle v1="', sa)
                t = np.char.add(t, '" v2="')
                t = np.char.add(t, sb)
                t = np.char.add(t, '" v3="')
                t = np.char.add(t, sc)
                t = np.char.add(t, '" />')
                return t
            
            # Triangle 1
            t1 = make_tris_str(i0, i2, i1) # 0-2-1
            # Triangle 2
            t2 = make_tris_str(i0, i3, i2) # 0-3-2
            
            triangles_list.append(t1)
            triangles_list.append(t2)
            
            vertex_count += 4 * n_faces
            
        # Define face geometries (Vertices order: BL, TL, TR, BR ? )
        # Coordinate system: X=Right, Y=Down(Image), Z=Depth
        # Let's use standard cube corners Logic
        # 0:000, 1:100, 2:110, 3:010 (Bottom Z=0)
        # 4:001, 5:101, 6:111, 7:011 (Top Z=1)
        
        # Left Face (-X): 0, 3, 7, 4 (CCW looking from left) -> (0,0,0) (0,1,0) (0,1,1) (0,0,1)
        add_faces(f_left,  [(0,0,0), (0,1,0), (0,1,1), (0,0,1)], 'left')
        
        # Right Face (+X): 1, 5, 6, 2 -> (1,0,0) (1,0,1) (1,1,1) (1,1,0)
        add_faces(f_right, [(1,0,0), (1,0,1), (1,1,1), (1,1,0)], 'right')
        
        # Top Face (-Y): 0, 4, 5, 1 -> (0,0,0) (0,0,1) (1,0,1) (1,0,0)
        add_faces(f_top,   [(0,0,0), (0,0,1), (1,0,1), (1,0,0)], 'top')
        
        # Bottom Face (+Y): 3, 2, 6, 7 -> (0,1,0) (1,1,0) (1,1,1) (0,1,1)
        add_faces(f_bottom, [(0,1,0), (1,1,0), (1,1,1), (0,1,1)], 'bottom')
        
        # Down Face (-Z): 0, 1, 2, 3 -> (0,0,0) (1,0,0) (1,1,0) (0,1,0)
        # (Looking from bottom up, standard winding 0-2-1 is down?)
        # Let's stick to CCW outside.
        # -Z normal: 0-1-2-3 is CW if looking from top. CCW if looking from bottom.
        add_faces(f_down,  [(0,0,0), (1,0,0), (1,1,0), (0,1,0)], 'down')
        
        # Up Face (+Z): 4, 7, 6, 5 -> (0,0,1) (0,1,1) (1,1,1) (1,0,1)
        add_faces(f_up,    [(0,0,1), (0,1,1), (1,1,1), (1,0,1)], 'up')
        
        # Concatenate all strings
        if not vertices_list:
            return {'vertices': '', 'triangles': ''}
            
        # Flatten lists
        v_final = np.concatenate(vertices_list)
        t_final = np.concatenate(triangles_list)
        
        return {
            'vertices': "".join(v_final),
            'triangles': "".join(t_final)
        }
