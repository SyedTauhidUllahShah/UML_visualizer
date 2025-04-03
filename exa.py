import uuid
import ast
import os
from enum import Enum
from typing import List, Dict, Any, Set, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path
import networkx as nx

class Visibility(Enum):
    PUBLIC = "+"
    PRIVATE = "-"
    PROTECTED = "#"
    PACKAGE = "~"

class RelationType(Enum):
    ASSOCIATION = "association"
    INHERITANCE = "inheritance"
    IMPLEMENTATION = "implementation"
    DEPENDENCY = "dependency"
    AGGREGATION = "aggregation"
    COMPOSITION = "composition"

class AttributeType(Enum):
    FIELD = "field"
    METHOD = "method"

class UMLElement:
    def __init__(self, name):
        self.id = str(uuid.uuid4())
        self.name = name
        self.x = 0
        self.y = 0
        self.width = 200
        self.height = 100

class UMLAttribute:
    def __init__(self, name, type_name="", visibility=Visibility.PUBLIC, static=False, attribute_type=AttributeType.FIELD, params=None, return_type=None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.type_name = type_name
        self.visibility = visibility
        self.static = static
        self.attribute_type = attribute_type
        self.params = params or []
        self.return_type = return_type

class UMLClass(UMLElement):
    def __init__(self, name, abstract=False):
        super().__init__(name)
        self.attributes = []
        self.methods = []
        self.abstract = abstract
    
    def add_attribute(self, name, type_name="", visibility=Visibility.PUBLIC, static=False):
        attr = UMLAttribute(name, type_name, visibility, static, AttributeType.FIELD)
        self.attributes.append(attr)
        return attr
    
    def add_method(self, name, return_type=None, params=None, visibility=Visibility.PUBLIC, static=False):
        method = UMLAttribute(name, "", visibility, static, AttributeType.METHOD, params, return_type)
        self.methods.append(method)
        return method

class UMLInterface(UMLElement):
    def __init__(self, name):
        super().__init__(name)
        self.methods = []
    
    def add_method(self, name, return_type=None, params=None, visibility=Visibility.PUBLIC):
        method = UMLAttribute(name, "", visibility, False, AttributeType.METHOD, params, return_type)
        self.methods.append(method)
        return method

class UMLRelationship:
    def __init__(self, source_id, target_id, relationship_type=RelationType.ASSOCIATION, 
                 source_multiplicity="", target_multiplicity="", label=""):
        self.id = str(uuid.uuid4())
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.source_multiplicity = source_multiplicity
        self.target_multiplicity = target_multiplicity
        self.label = label

class UMLDiagram:
    def __init__(self, name):
        self.name = name
        self.elements = {}
        self.relationships = []
    
    def add_element(self, element):
        self.elements[element.id] = element
        return element
    
    def add_relationship(self, source_id, target_id, relationship_type=RelationType.ASSOCIATION, 
                         source_multiplicity="", target_multiplicity="", label=""):
        rel = UMLRelationship(source_id, target_id, relationship_type, source_multiplicity, target_multiplicity, label)
        self.relationships.append(rel)
        return rel
    
    def auto_layout(self):
        # Create initial graph for layout
        G = nx.Graph()
        
        # Build inheritance and implementation hierarchy
        inheritance_hierarchy = self._build_inheritance_hierarchy()
        
        # Calculate proper element sizes based on content
        self._calculate_element_sizes()
        
        # Add nodes to graph with size attributes
        for element_id, element in self.elements.items():
            G.add_node(element_id, width=element.width, height=element.height)
        
        # Add relationships as edges with type attributes
        for rel in self.relationships:
            rel_type = 1 if rel.relationship_type in [RelationType.INHERITANCE, RelationType.IMPLEMENTATION] else 2
            G.add_edge(rel.source_id, rel.target_id, type=rel_type)
        
        # Assign hierarchy layers
        layers = self._assign_hierarchy_layers(inheritance_hierarchy)
        
        # Position elements by layer with increased spacing
        self._position_by_layer(layers)

        # Apply enhanced force-directed adjustment to separate elements
        self._force_directed_adjustment(50)
        
        # Final passes to resolve overlaps
        for _ in range(5):
            self._resolve_overlaps()
    
    def _calculate_element_sizes(self):
        """Calculate proper element sizes based on content"""
        for element_id, element in self.elements.items():
            # Base height for name section
            name_height = 30
            
            if isinstance(element, UMLClass):
                # Calculate attribute section height
                attr_count = len(element.attributes)
                attr_height = max(22 * attr_count, 15) if attr_count > 0 else 10
                
                # Calculate method section height
                method_count = len(element.methods)
                method_height = max(22 * method_count, 15) if method_count > 0 else 10
                
                # Adjust width based on content length
                max_width = 200  # Base width
                for attr in element.attributes:
                    attr_text = f"{attr.visibility.value}{attr.name}: {attr.type_name}"
                    text_width = len(attr_text) * 6.5  # Approximate width based on text length
                    max_width = max(max_width, text_width + 20)  # Add padding
                
                for method in element.methods:
                    method_text = format_method_text(method, 100)  # Get method text
                    text_width = len(method_text) * 6.5  # Approximate width based on text length
                    max_width = max(max_width, text_width + 20)  # Add padding
                
                element.width = min(max_width, 350)  # Cap max width
                element.height = name_height + attr_height + method_height
                
            elif isinstance(element, UMLInterface):
                # Calculate method section height
                method_count = len(element.methods)
                method_height = max(22 * method_count, 15) if method_count > 0 else 10
                
                # Adjust width based on content length
                max_width = 200  # Base width
                for method in element.methods:
                    method_text = format_method_text(method, 100)  # Get method text
                    text_width = len(method_text) * 6.5  # Approximate width based on text length
                    max_width = max(max_width, text_width + 20)  # Add padding
                
                element.width = min(max_width, 300)  # Cap max width
                element.height = name_height + method_height
    
    def _build_inheritance_hierarchy(self):
        """Build inheritance/implementation hierarchy"""
        hierarchy = {}
        for rel in self.relationships:
            if rel.relationship_type in [RelationType.INHERITANCE, RelationType.IMPLEMENTATION]:
                parent_id = rel.target_id
                child_id = rel.source_id
                
                if parent_id not in hierarchy:
                    hierarchy[parent_id] = []
                
                hierarchy[parent_id].append(child_id)
        return hierarchy
    
    def _assign_hierarchy_layers(self, hierarchy):
        """Assign elements to layers based on inheritance hierarchy"""
        layers = {}
        visited = set()
        
        def assign_layer(element_id, layer=0):
            if element_id in visited:
                return
            
            visited.add(element_id)
            
            if layer not in layers:
                layers[layer] = []
            
            layers[layer].append(element_id)
            
            # Process children
            children = hierarchy.get(element_id, [])
            for child_id in children:
                assign_layer(child_id, layer + 1)
        
        # Find root elements (interfaces and parents with no parents)
        roots = []
        for element_id, element in self.elements.items():
            is_child = False
            for rel in self.relationships:
                if rel.relationship_type in [RelationType.INHERITANCE, RelationType.IMPLEMENTATION]:
                    if rel.source_id == element_id:
                        is_child = True
                        break
            
            if not is_child:
                roots.append(element_id)
        
        # Assign layers starting from roots
        for root_id in roots:
            assign_layer(root_id)
        
        # Handle any remaining elements
        layer_count = len(layers)
        remaining_layer = layer_count
        for element_id in self.elements:
            if element_id not in visited:
                if remaining_layer not in layers:
                    layers[remaining_layer] = []
                layers[remaining_layer].append(element_id)
        
        return layers
    
    def _position_by_layer(self, layers):
        """Position elements by layer with improved spacing"""
        if not layers:
            # Fallback for no layers - use basic grid layout
            self._grid_layout()
            return
            
        max_layer = max(layers.keys()) if layers else 0
        
        # Increased spacing between layers and elements
        layer_spacing = 300  # Vertical spacing between layers
        min_element_spacing = 80  # Minimum horizontal spacing between elements
        
        # First pass: position interfaces at the top
        interface_ids = [eid for eid, e in self.elements.items() if isinstance(e, UMLInterface)]
        if interface_ids:
            interface_layer = -1  # Place interfaces above top layer
            if interface_layer not in layers:
                layers[interface_layer] = []
            for eid in interface_ids:
                # Remove from other layers if present
                for layer_num in list(layers.keys()):
                    if layer_num != interface_layer and eid in layers[layer_num]:
                        layers[layer_num].remove(eid)
                # Add to interface layer
                if eid not in layers[interface_layer]:
                    layers[interface_layer].append(eid)
        
        # Second pass: position by layer with better distribution
        for layer, element_ids in sorted(layers.items()):
            if not element_ids:
                continue
                
            # Sort elements to keep related elements close
            element_ids.sort(key=lambda eid: self.elements[eid].name)
            
            # Calculate total width needed for this layer
            total_width = sum(self.elements[eid].width for eid in element_ids)
            total_width += min_element_spacing * (len(element_ids) - 1)
            
            # Center the layer
            start_x = -total_width / 2
            
            # Position each element in the layer
            current_x = start_x
            for element_id in element_ids:
                element = self.elements[element_id]
                element.x = current_x
                element.y = layer * layer_spacing
                current_x += element.width + min_element_spacing
    
    def _grid_layout(self):
        """Simple grid layout as fallback"""
        elements = list(self.elements.values())
        cols = max(int(np.sqrt(len(elements))), 1)
        row, col = 0, 0
        spacing_x, spacing_y = 300, 300
        
        for element in elements:
            element.x = col * spacing_x
            element.y = row * spacing_y
            col += 1
            if col >= cols:
                col = 0
                row += 1
    
    def _force_directed_adjustment(self, iterations=20):
        """Apply force-directed layout to separate elements"""
        elements = list(self.elements.values())
        
        # Define forces
        repulsion_strength = 5000  # Strength of repulsion between elements
        attraction_strength = 0.01  # Strength of attraction for related elements
        damping = 0.9  # Damping factor to prevent oscillation
        
        # Track velocity for each element
        velocities = {e.id: [0, 0] for e in elements}
        
        for _ in range(iterations):
            # Reset forces
            forces = {e.id: [0, 0] for e in elements}
            
            # Calculate repulsion forces between all elements
            for i, elem1 in enumerate(elements):
                for j, elem2 in enumerate(elements[i+1:], i+1):
                    # Calculate distance between centers
                    dx = elem2.x + elem2.width/2 - (elem1.x + elem1.width/2)
                    dy = elem2.y + elem2.height/2 - (elem1.y + elem1.height/2)
                    distance = max(1, np.sqrt(dx**2 + dy**2))
                    
                    # Calculate repulsion force (stronger when closer)
                    force = repulsion_strength / (distance**2)
                    
                    # Normalize direction
                    if distance > 0:
                        dx /= distance
                        dy /= distance
                    
                    # Apply force to both elements in opposite directions
                    forces[elem1.id][0] -= force * dx
                    forces[elem1.id][1] -= force * dy
                    forces[elem2.id][0] += force * dx
                    forces[elem2.id][1] += force * dy
            
            # Calculate attraction forces for related elements
            for rel in self.relationships:
                if rel.source_id in self.elements and rel.target_id in self.elements:
                    source = self.elements[rel.source_id]
                    target = self.elements[rel.target_id]
                    
                    # Calculate distance between centers
                    dx = target.x + target.width/2 - (source.x + source.width/2)
                    dy = target.y + target.height/2 - (source.y + source.height/2)
                    distance = max(1, np.sqrt(dx**2 + dy**2))
                    
                    # Calculate attraction force (stronger when further)
                    force = attraction_strength * distance
                    
                    # Normalize direction
                    if distance > 0:
                        dx /= distance
                        dy /= distance
                    
                    # Apply force to both elements to bring them closer
                    forces[source.id][0] += force * dx
                    forces[source.id][1] += force * dy
                    forces[target.id][0] -= force * dx
                    forces[target.id][1] -= force * dy
            
            # Update positions based on forces
            for element in elements:
                # Update velocity with damping
                velocities[element.id][0] = velocities[element.id][0] * damping + forces[element.id][0]
                velocities[element.id][1] = velocities[element.id][1] * damping + forces[element.id][1]
                
                # Apply velocity to position
                element.x += velocities[element.id][0]
                element.y += velocities[element.id][1]
    
    def _resolve_overlaps(self):
        """Resolve overlaps between elements"""
        elements = list(self.elements.values())
        min_spacing = 50  # Minimum spacing between elements
        
        # Sort elements to prioritize maintaining vertical hierarchy
        elements.sort(key=lambda e: e.y)
        
        # Check for overlaps and resolve them
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i == j:
                    continue
                    
                # Get element rectangles with spacing
                rect1 = (elem1.x - min_spacing/2, elem1.y - min_spacing/2, 
                         elem1.width + min_spacing, elem1.height + min_spacing)
                rect2 = (elem2.x - min_spacing/2, elem2.y - min_spacing/2, 
                         elem2.width + min_spacing, elem2.height + min_spacing)
                
                # Check for overlap
                overlap_x = (rect1[0] < rect2[0] + rect2[2] and rect1[0] + rect1[2] > rect2[0])
                overlap_y = (rect1[1] < rect2[1] + rect2[3] and rect1[1] + rect1[3] > rect2[1])
                
                if overlap_x and overlap_y:
                    # Calculate overlap amounts
                    overlap_x_amount = min(rect1[0] + rect1[2] - rect2[0], rect2[0] + rect2[2] - rect1[0])
                    overlap_y_amount = min(rect1[1] + rect1[3] - rect2[1], rect2[1] + rect2[3] - rect1[1])
                    
                    # Determine if elements are in the same layer (similar y-coordinate)
                    same_layer = abs(elem1.y - elem2.y) < min_spacing
                    
                    if same_layer or overlap_x_amount < overlap_y_amount:
                        # Resolve horizontally
                        mid1_x = elem1.x + elem1.width/2
                        mid2_x = elem2.x + elem2.width/2
                        
                        if mid1_x < mid2_x:
                            move_x = overlap_x_amount / 2 + min_spacing/4
                            elem1.x -= move_x
                            elem2.x += move_x
                        else:
                            move_x = overlap_x_amount / 2 + min_spacing/4
                            elem1.x += move_x
                            elem2.x -= move_x
                    else:
                        # Resolve vertically
                        mid1_y = elem1.y + elem1.height/2
                        mid2_y = elem2.y + elem2.height/2
                        
                        if mid1_y < mid2_y:
                            move_y = overlap_y_amount / 2 + min_spacing/4
                            elem1.y -= move_y
                            elem2.y += move_y
                        else:
                            move_y = overlap_y_amount / 2 + min_spacing/4
                            elem1.y += move_y
                            elem2.y -= move_y

# Added the missing UMLParser class
class UMLParser:
    def __init__(self):
        self.diagram = UMLDiagram("Generated Diagram")
        self.classes = {}
        self.interfaces = {}
        self.inheritance_relationships = []
    
    def parse_file(self, file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()
        self._parse_code(code)
    
    def parse_directory(self, directory_path: str) -> None:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self.parse_file(file_path)
    
    def parse_text(self, code: str) -> None:
        self._parse_code(code)
    
    def _parse_code(self, code: str) -> None:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._process_class(node)
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
    
    def _process_class(self, node: ast.ClassDef) -> None:
        is_interface = self._is_interface(node)
        is_abstract = self._is_abstract(node)
        if is_interface:
            uml_element = UMLInterface(node.name)
            self.interfaces[node.name] = uml_element
        else:
            uml_element = UMLClass(node.name, abstract=is_abstract)
            self.classes[node.name] = uml_element
        self.diagram.add_element(uml_element)
        for base in node.bases:
            base_name = self._get_name_from_node(base)
            if base_name:
                self.inheritance_relationships.append((node.name, base_name))
        for child_node in node.body:
            if isinstance(child_node, ast.FunctionDef):
                self._process_method(child_node, uml_element)
            elif isinstance(child_node, ast.Assign):
                self._process_attribute(child_node, uml_element)
            elif isinstance(child_node, ast.AnnAssign):
                self._process_annotated_attribute(child_node, uml_element)
    
    def _is_interface(self, node: ast.ClassDef) -> bool:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'interface':
                return True
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id.startswith('Interface'):
                return True
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if not methods:
            return False
        return all(self._is_abstract_method(m) for m in methods)
    
    def _is_abstract(self, node: ast.ClassDef) -> bool:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                return True
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and self._is_abstract_method(child):
                return True
        return False
    
    def _is_abstract_method(self, node: ast.FunctionDef) -> bool:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                return True
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name):
                    if stmt.exc.func.id == 'NotImplementedError':
                        return True
        return False
    
    def _process_method(self, node: ast.FunctionDef, uml_element: Union[UMLClass, UMLInterface]) -> None:
        method_name = node.name
        visibility = self._get_visibility_from_name(method_name)
        is_static = any(isinstance(d, ast.Name) and d.id == 'staticmethod' for d in node.decorator_list)
        params = []
        for arg in node.args.args:
            if arg.arg != 'self' and arg.arg != 'cls':
                param_type = ""
                if hasattr(arg, 'annotation') and arg.annotation:
                    param_type = self._get_type_annotation(arg.annotation)
                params.append({"name": arg.arg, "type": param_type})
        return_type = None
        if node.returns:
            return_type = self._get_type_annotation(node.returns)
        if isinstance(uml_element, UMLClass):
            uml_element.add_method(method_name, return_type, params, visibility, is_static)
        elif isinstance(uml_element, UMLInterface):
            uml_element.add_method(method_name, return_type, params, visibility)
    
    def _process_attribute(self, node: ast.Assign, uml_element: UMLClass) -> None:
        if not isinstance(uml_element, UMLClass):
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                attr_name = target.id
                visibility = self._get_visibility_from_name(attr_name)
                type_name = ""
                if isinstance(node.value, ast.Num):
                    type_name = "int" if isinstance(node.value.n, int) else "float"
                elif isinstance(node.value, ast.Str):
                    type_name = "str"
                elif isinstance(node.value, ast.List):
                    type_name = "list"
                elif isinstance(node.value, ast.Dict):
                    type_name = "dict"
                uml_element.add_attribute(attr_name, type_name, visibility)
    
    def _process_annotated_attribute(self, node: ast.AnnAssign, uml_element: UMLClass) -> None:
        if not isinstance(uml_element, UMLClass) or not isinstance(node.target, ast.Name):
            return
        attr_name = node.target.id
        visibility = self._get_visibility_from_name(attr_name)
        type_name = self._get_type_annotation(node.annotation)
        uml_element.add_attribute(attr_name, type_name, visibility)
    
    def _get_visibility_from_name(self, name: str) -> Visibility:
        if name.startswith("__"):
            return Visibility.PRIVATE
        elif name.startswith("_"):
            return Visibility.PROTECTED
        return Visibility.PUBLIC
    
    def _get_type_annotation(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                container = node.value.id
                if isinstance(node.slice, ast.Index):
                    if hasattr(node.slice, 'value'):
                        if isinstance(node.slice.value, ast.Name):
                            return f"{container}[{node.slice.value.id}]"
                        elif isinstance(node.slice.value, ast.Tuple):
                            elts = []
                            for elt in node.slice.value.elts:
                                if isinstance(elt, ast.Name):
                                    elts.append(elt.id)
                            return f"{container}[{', '.join(elts)}]"
                elif hasattr(node, 'slice') and isinstance(node.slice, ast.Name):
                    return f"{container}[{node.slice.id}]"
        return "Any"
    
    def _get_name_from_node(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None
    
    def create_relationships(self) -> None:
        for child_name, parent_name in self.inheritance_relationships:
            child_id = None
            parent_id = None
            relationship_type = RelationType.INHERITANCE
            if child_name in self.classes:
                child_id = self.classes[child_name].id
            elif child_name in self.interfaces:
                child_id = self.interfaces[child_name].id
            if parent_name in self.classes:
                parent_id = self.classes[parent_name].id
            elif parent_name in self.interfaces:
                parent_id = self.interfaces[parent_name].id
                if child_id and child_name in self.classes:
                    relationship_type = RelationType.IMPLEMENTATION
            if child_id and parent_id:
                self.diagram.add_relationship(child_id, parent_id, relationship_type)
    
    def get_diagram(self) -> UMLDiagram:
        self.create_relationships()
        self.diagram.auto_layout()
        return self.diagram

def calculate_connection_points(source_rect, target_rect):
    """Calculate optimal connection points between two rectangles"""
    sx, sy, sw, sh = source_rect
    tx, ty, tw, th = target_rect
    
    # Calculate centers
    scx, scy = sx + sw/2, sy + sh/2
    tcx, tcy = tx + tw/2, ty + th/2
    
    # Calculate direction vector
    dx = tcx - scx
    dy = tcy - scy
    
    # Get angle between centers
    angle = np.arctan2(dy, dx)
    angle_degrees = np.degrees(angle)
    
    # Determine connecting sides based on angle
    if -45 <= angle_degrees <= 45:  # Right of source to left of target
        sp = (sx + sw, sy + sh/2)
        tp = (tx, ty + th/2)
    elif 45 < angle_degrees <= 135:  # Bottom of source to top of target
        sp = (sx + sw/2, sy + sh)
        tp = (tx + tw/2, ty)
    elif angle_degrees > 135 or angle_degrees <= -135:  # Left of source to right of target
        sp = (sx, sy + sh/2)
        tp = (tx + tw, ty + th/2)
    else:  # Top of source to bottom of target
        sp = (sx + sw/2, sy)
        tp = (tx + tw/2, ty + th)
    
    return {'source': sp, 'target': tp}

def route_path(source_point, target_point, source_rect, target_rect, elements_dict):
    """Create a smart path between two points that avoids other elements"""
    # Extract source and target coordinates
    sx, sy = source_point
    tx, ty = target_point
    
    # Get source and target element dimensions
    sx_rect, sy_rect, sw_rect, sh_rect = source_rect
    tx_rect, ty_rect, tw_rect, th_rect = target_rect
    
    # Get centers
    scx, scy = sx_rect + sw_rect/2, sy_rect + sh_rect/2
    tcx, tcy = tx_rect + tw_rect/2, ty_rect + th_rect/2
    
    # Calculate path points based on the relationship direction
    path_points = [(sx, sy)]  # Start with source point
    
    # Calculate the straight-line distance
    direct_distance = np.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
    
    # Determine the general relationship direction
    dx = abs(tx - sx)
    dy = abs(ty - sy)
    
    # For inheritance/implementation (usually vertical relationships), prefer straight lines
    # For other types, use orthogonal paths
    
    if dy > 2 * dx:  # Mostly vertical
        # Add a vertical segment first, then horizontal
        path_points.append((sx, ty))
    elif dx > 2 * dy:  # Mostly horizontal
        # Add a horizontal segment first, then vertical
        path_points.append((tx, sy))
    else:  # Mixed direction
        # Calculate midpoints
        mid_x = (sx + tx) / 2
        mid_y = (sy + ty) / 2
        
        # Create a path with two segments via midpoint
        path_points.append((mid_x, sy))
        path_points.append((mid_x, ty))
    
    # Add target point
    path_points.append((tx, ty))
    
    # Simplify path if possible - remove redundant points
    simplified_path = [path_points[0]]
    for i in range(1, len(path_points) - 1):
        prev = simplified_path[-1]
        current = path_points[i]
        next_point = path_points[i + 1]
        
        # Only add point if it changes direction
        if not (prev[0] == current[0] == next_point[0] or prev[1] == current[1] == next_point[1]):
            simplified_path.append(current)
    
    simplified_path.append(path_points[-1])
    
    return simplified_path

def format_method_text(method, max_length=40):
    """Format method text with better handling of long signatures"""
    # Handle parameters
    param_str = ""
    if method.params:
        param_texts = []
        for param in method.params:
            param_text = param["name"]
            if param.get("type"):
                param_text += f": {param['type']}"
            param_texts.append(param_text)
        
        # Show up to first 2 params, then use ellipsis if too many
        if len(param_texts) > 2:
            param_str = ", ".join(param_texts[:2]) + ", ..."
        else:
            param_str = ", ".join(param_texts)
    
    # Add return type if available
    return_str = f": {method.return_type}" if method.return_type else ""
    
    # Create method signature
    method_text = f"{method.visibility.value}{method.name}({param_str}){return_str}"
    
    # Truncate if too long
    if len(method_text) > max_length:
        method_text = method_text[:max_length-3] + "..."
    
    return method_text

def draw_arrow_head(ax, x, y, angle, arrow_type, style):
    """Draw the appropriate arrow head for the relationship type"""
    # Arrow properties
    arrow_size = 14
    arrow_width = 10
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    if arrow_type == 'inheritance' or arrow_type == 'implementation':
        # Triangle for inheritance/implementation
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        point1 = (x, y)  # Point
        point2 = (x - arrow_size * dx + arrow_width/2 * dy, 
                 y - arrow_size * dy - arrow_width/2 * dx)  # Base left
        point3 = (x - arrow_size * dx - arrow_width/2 * dy,
                 y - arrow_size * dy + arrow_width/2 * dx)  # Base right
        
        triangle = patches.Polygon([point1, point2, point3], 
                                   closed=True, 
                                   fill=True, 
                                   facecolor='white',
                                   edgecolor='black', 
                                   linewidth=1.5,
                                   zorder=3)
        ax.add_patch(triangle)
        
    elif arrow_type == 'aggregation' or arrow_type == 'composition':
        # Diamond for aggregation/composition
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        point1 = (x, y)  # Front point
        point2 = (x - arrow_size/2 * dx + arrow_width/2 * dy, 
                 y - arrow_size/2 * dy - arrow_width/2 * dx)  # Right point
        point3 = (x - arrow_size * dx, 
                 y - arrow_size * dy)  # Back point
        point4 = (x - arrow_size/2 * dx - arrow_width/2 * dy,
                 y - arrow_size/2 * dy + arrow_width/2 * dx)  # Left point
        
        fill_color = 'black' if arrow_type == 'composition' else 'white'
        diamond = patches.Polygon([point1, point2, point3, point4], 
                                  closed=True, 
                                  fill=True, 
                                  facecolor=fill_color,
                                  edgecolor='black', 
                                  linewidth=1.5,
                                  zorder=3)
        ax.add_patch(diamond)
        
    elif arrow_type == 'association' or arrow_type == 'dependency':
        # Open arrowhead for association/dependency
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        point1 = (x, y)  # Tip
        point2 = (x - arrow_size * dx + arrow_width/2 * dy, 
                 y - arrow_size * dy - arrow_width/2 * dx)  # Left wing
        point3 = (x - arrow_size * dx - arrow_width/2 * dy,
                 y - arrow_size * dy + arrow_width/2 * dx)  # Right wing
        
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
                color='black', linestyle=style, linewidth=1.5, zorder=3)
        ax.plot([point1[0], point3[0]], [point1[1], point3[1]], 
                color='black', linestyle=style, linewidth=1.5, zorder=3)

def draw_uml_diagram(diagram, filename=None, figsize=(24, 18)):
    """Draw UML diagram with improved formatting and layout"""
    fig = plt.figure(figsize=figsize, dpi=100, facecolor='white')
    ax = fig.add_subplot(111)
    
    # Colors
    header_color = '#FFFFAA'  # Yellow for headers
    interface_color = '#CCFFCC'  # Light green for interfaces
    abstract_color = '#FFCCCC'  # Light red for abstract classes
    class_color = '#FFFFFF'  # White for regular classes
    border_color = '#333333'  # Dark gray for borders
    
    # Calculate diagram boundaries with padding
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for element in diagram.elements.values():
        min_x = min(min_x, element.x)
        max_x = max(max_x, element.x + element.width)
        min_y = min(min_y, element.y)
        max_y = max(max_y, element.y + element.height)
    
    # Add padding
    padding = 300
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)  # Inverted for top-down view
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Store element information for relationship routing
    element_info = {}
    
    # Draw elements (first pass - classes and interfaces)
    for element_id, element in diagram.elements.items():
        # Calculate heights for each section
        name_height = 30
        attr_height = 0
        method_height = 0
        
        if isinstance(element, UMLClass):
            attr_height = 22 * max(1, len(element.attributes))
            method_height = 22 * max(1, len(element.methods))
        elif isinstance(element, UMLInterface):
            method_height = 22 * max(1, len(element.methods))
        
        total_height = name_height + attr_height + method_height
        
        # Determine background color based on element type
        if isinstance(element, UMLInterface):
            bg_color = interface_color
        elif isinstance(element, UMLClass) and element.abstract:
            bg_color = abstract_color
        else:
            bg_color = class_color
        
        # Store element information
        element_info[element_id] = {
            'x': element.x,
            'y': element.y,
            'width': element.width,
            'height': total_height,
            'name_height': name_height,
            'attr_height': attr_height,
            'type': 'interface' if isinstance(element, UMLInterface) else 'class'
        }
        
        # Draw main rectangle
        rect = patches.Rectangle(
            (element.x, element.y),
            element.width, total_height,
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Draw name section
        name_rect = patches.Rectangle(
            (element.x, element.y + total_height - name_height),
            element.width, name_height,
            facecolor=header_color,
            edgecolor=border_color,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(name_rect)
        
        # Format class/interface name
        if isinstance(element, UMLClass) and element.abstract:
            name_text = f"«abstract» {element.name}"
            font_style = 'italic'
        elif isinstance(element, UMLInterface):
            name_text = f"«interface» {element.name}"
            font_style = 'normal'
        else:
            name_text = element.name
            font_style = 'normal'
        
        # Draw name text
        ax.text(
            element.x + element.width/2,
            element.y + total_height - name_height/2,
            name_text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=11,
            fontweight='bold',
            fontstyle=font_style,
            family='sans-serif',
            zorder=3
        )
        
        # Draw dividers between sections
        if attr_height > 0:
            ax.plot(
                [element.x, element.x + element.width],
                [element.y + method_height, element.y + method_height],
                color=border_color,
                linewidth=1.0,
                zorder=2.5
            )
        
        # Draw attributes
        if isinstance(element, UMLClass) and element.attributes:
            for i, attr in enumerate(element.attributes):
                attr_type_str = f": {attr.type_name}" if attr.type_name else ""
                attr_text = f"{attr.visibility.value}{attr.name}{attr_type_str}"
                
                # Truncate if too long
                if len(attr_text) > 40:
                    attr_text = attr_text[:37] + "..."
                
                ax.text(
                    element.x + 10,  # Left margin
                    element.y + method_height + (i + 0.5) * (attr_height / max(1, len(element.attributes))),
                    attr_text,
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=9,
                    family='monospace',
                    zorder=3
                )
        
        # Draw methods
        methods = element.methods if isinstance(element, UMLInterface) else element.methods
        if methods:
            for i, method in enumerate(methods):
                # Format method text
                method_text = format_method_text(method)
                
                ax.text(
                    element.x + 10,  # Left margin
                    element.y + (i + 0.5) * (method_height / max(1, len(methods))),
                    method_text,
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=9,
                    family='monospace',
                    zorder=3
                )
    
    # Draw relationships (second pass)
    for rel in diagram.relationships:
        source_info = element_info.get(rel.source_id)
        target_info = element_info.get(rel.target_id)
        
        if not source_info or not target_info:
            continue
        
        # Calculate connection points
        source_rect = (source_info['x'], source_info['y'], source_info['width'], source_info['height'])
        target_rect = (target_info['x'], target_info['y'], target_info['width'], target_info['height'])
        
        connection = calculate_connection_points(source_rect, target_rect)
        source_point = connection['source']
        target_point = connection['target']
        
        # Get line style based on relationship type
        line_style = 'dashed' if rel.relationship_type in [RelationType.IMPLEMENTATION, RelationType.DEPENDENCY] else 'solid'
        
        # Calculate path points
        path_points = route_path(source_point, target_point, source_rect, target_rect, diagram.elements)
        
        # Draw path segments
        for i in range(len(path_points) - 1):
            x1, y1 = path_points[i]
            x2, y2 = path_points[i + 1]
            ax.plot([x1, x2], [y1, y2], 
                   color=border_color, 
                   linestyle=line_style, 
                   linewidth=1.5,
                   zorder=1)
        
        # Calculate endpoint angle for arrow
        if len(path_points) >= 2:
            pre_end_point = path_points[-2]
            end_point = path_points[-1]
            angle = np.degrees(np.arctan2(end_point[1] - pre_end_point[1], 
                                        end_point[0] - pre_end_point[0]))
        else:
            angle = np.degrees(np.arctan2(target_point[1] - source_point[1], 
                                        target_point[0] - source_point[0]))
        
        # Map relationship type to arrow style
        arrow_type_map = {
            RelationType.INHERITANCE: 'inheritance',
            RelationType.IMPLEMENTATION: 'inheritance',
            RelationType.ASSOCIATION: 'association',
            RelationType.DEPENDENCY: 'dependency',
            RelationType.AGGREGATION: 'aggregation',
            RelationType.COMPOSITION: 'composition'
        }
        
        # Draw arrow head
        draw_arrow_head(
            ax, 
            path_points[-1][0], path_points[-1][1], 
            angle, 
            arrow_type_map[rel.relationship_type],
            line_style
        )
        
        # Draw multiplicity labels
        if rel.source_multiplicity:
            # Position near source
            if len(path_points) >= 2:
                dx = path_points[1][0] - path_points[0][0]
                dy = path_points[1][1] - path_points[0][1]
                angle = np.arctan2(dy, dx)
                
                offset = 15
                mx = path_points[0][0] + offset * np.sin(angle)
                my = path_points[0][1] - offset * np.cos(angle)
            else:
                mx = source_point[0] + 15
                my = source_point[1] - 15
                
            ax.text(
                mx, my,
                rel.source_multiplicity,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=9,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'),
                zorder=3
            )
        
        if rel.target_multiplicity:
            # Position near target
            if len(path_points) >= 2:
                dx = path_points[-1][0] - path_points[-2][0]
                dy = path_points[-1][1] - path_points[-2][1]
                angle = np.arctan2(dy, dx)
                
                offset = 15
                mx = path_points[-1][0] + offset * np.sin(angle)
                my = path_points[-1][1] - offset * np.cos(angle)
            else:
                mx = target_point[0] + 15
                my = target_point[1] - 15
                
            ax.text(
                mx, my,
                rel.target_multiplicity,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=9,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'),
                zorder=3
            )
        
        # Draw relationship label
        if rel.label:
            # Find midpoint of the path
            if len(path_points) >= 2:
                mid_index = len(path_points) // 2
                if len(path_points) % 2 == 0:
                    mid_x = (path_points[mid_index-1][0] + path_points[mid_index][0]) / 2
                    mid_y = (path_points[mid_index-1][1] + path_points[mid_index][1]) / 2
                else:
                    mid_x, mid_y = path_points[mid_index]
            else:
                mid_x = (source_point[0] + target_point[0]) / 2
                mid_y = (source_point[1] + target_point[1]) / 2
            
            ax.text(
                mid_x, mid_y,
                rel.label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=9,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'),
                zorder=3
            )
    
    # Add diagram title
    if diagram.name:
        ax.text(
            0.5, 0.98,
            diagram.name,
            transform=fig.transFigure,
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=16,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def generate_uml_from_code(code=None, file_path=None, directory_path=None, save_path=None, diagram_name="Class Diagram", figsize=(24, 18)):
    """Generate UML diagram from Python code with improved layout"""
    parser = UMLParser()
    
    if diagram_name:
        parser.diagram.name = diagram_name
    
    if code:
        parser.parse_text(code)
    elif file_path:
        parser.parse_file(file_path)
    elif directory_path:
        parser.parse_directory(directory_path)
    else:
        return None
    
    diagram = parser.get_diagram()
    fig = draw_uml_diagram(diagram, save_path, figsize)
    plt.show()
    return diagram

# Example code remains the same
def complex_banking_example():
    code = """
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set
from datetime import datetime

class ILoggable:
    @abstractmethod
    def log_activity(self, message: str) -> None:
        pass

class IIdentifiable:
    @abstractmethod
    def get_id(self) -> str:
        pass

class Person(ABC, IIdentifiable):
    def __init__(self, id: str, name: str, address: str):
        self._id = id
        self._name = name
        self._address = address
    
    def get_id(self) -> str:
        return self._id
    
    def get_name(self) -> str:
        return self._name
    
    def set_address(self, address: str) -> None:
        self._address = address
    
    def get_address(self) -> str:
        return self._address

class Customer(Person, ILoggable):
    def __init__(self, id: str, name: str, address: str, phone: str):
        super().__init__(id, name, address)
        self.__phone = phone
        self.__accounts = []
        self.__activity_log = []
    
    def add_account(self, account) -> None:
        self.__accounts.append(account)
    
    def remove_account(self, account) -> bool:
        if account in self.__accounts:
            self.__accounts.remove(account)
            return True
        return False
    
    def get_accounts(self):
        return self.__accounts
    
    def log_activity(self, message: str) -> None:
        timestamp = datetime.now()
        self.__activity_log.append(f"{timestamp}: {message}")

class Employee(Person, ILoggable):
    def __init__(self, id: str, name: str, address: str, position: str, salary: float):
        super().__init__(id, name, address)
        self.__position = position
        self.__salary = salary
        self.__branch = None
        self.__activity_log = []
    
    def set_branch(self, branch) -> None:
        self.__branch = branch
    
    def get_branch(self):
        return self.__branch
    
    def log_activity(self, message: str) -> None:
        timestamp = datetime.now()
        self.__activity_log.append(f"{timestamp}: {message}")

class Branch(IIdentifiable, ILoggable):
    def __init__(self, id: str, name: str, address: str):
        self.__id = id
        self.__name = name
        self.__address = address
        self.__employees = []
        self.__activity_log = []
    
    def get_id(self) -> str:
        return self.__id
    
    def add_employee(self, employee) -> None:
        self.__employees.append(employee)
        employee.set_branch(self)
    
    def remove_employee(self, employee) -> bool:
        if employee in self.__employees:
            self.__employees.remove(employee)
            employee.set_branch(None)
            return True
        return False
    
    def log_activity(self, message: str) -> None:
        timestamp = datetime.now()
        self.__activity_log.append(f"{timestamp}: {message}")

class Transaction(ABC, IIdentifiable, ILoggable):
    def __init__(self, id: str, amount: float, timestamp: datetime):
        self.__id = id
        self.__amount = amount
        self.__timestamp = timestamp
        self.__status = "PENDING"
    
    def get_id(self) -> str:
        return self.__id
    
    def get_amount(self) -> float:
        return self.__amount
    
    def get_timestamp(self) -> datetime:
        return self.__timestamp
    
    def get_status(self) -> str:
        return self.__status
    
    def set_status(self, status: str) -> None:
        self.__status = status
    
    @abstractmethod
    def execute(self) -> bool:
        pass
    
    def log_activity(self, message: str) -> None:
        print(f"Transaction {self.__id}: {message}")

class DepositTransaction(Transaction):
    def __init__(self, id: str, amount: float, timestamp: datetime, account, customer):
        super().__init__(id, amount, timestamp)
        self.__account = account
        self.__customer = customer
    
    def execute(self) -> bool:
        if self.get_amount() <= 0:
            self.set_status("FAILED")
            return False
        
        self.__account.deposit(self.get_amount())
        self.set_status("COMPLETED")
        self.log_activity(f"Deposited {self.get_amount()} into account {self.__account.get_id()}")
        return True

class WithdrawalTransaction(Transaction):
    def __init__(self, id: str, amount: float, timestamp: datetime, account, customer):
        super().__init__(id, amount, timestamp)
        self.__account = account
        self.__customer = customer
    
    def execute(self) -> bool:
        if self.get_amount() <= 0 or self.__account.get_balance() < self.get_amount():
            self.set_status("FAILED")
            return False
        
        self.__account.withdraw(self.get_amount())
        self.set_status("COMPLETED")
        self.log_activity(f"Withdrew {self.get_amount()} from account {self.__account.get_id()}")
        return True

class TransferTransaction(Transaction):
    def __init__(self, id: str, amount: float, timestamp: datetime, from_account, to_account, customer):
        super().__init__(id, amount, timestamp)
        self.__from_account = from_account
        self.__to_account = to_account
        self.__customer = customer
    
    def execute(self) -> bool:
        if (self.get_amount() <= 0 or 
            self.__from_account.get_balance() < self.get_amount()):
            self.set_status("FAILED")
            return False
        
        self.__from_account.withdraw(self.get_amount())
        self.__to_account.deposit(self.get_amount())
        self.set_status("COMPLETED")
        self.log_activity(f"Transferred {self.get_amount()} from account {self.__from_account.get_id()} to {self.__to_account.get_id()}")
        return True

class Account(ABC, IIdentifiable, ILoggable):
    def __init__(self, id: str, customer, balance: float = 0.0):
        self.__id = id
        self.__customer = customer
        self.__balance = balance
        self.__transactions = []
        self.__activity_log = []
        customer.add_account(self)
    
    def get_id(self) -> str:
        return self.__id
    
    def get_customer(self):
        return self.__customer
    
    def get_balance(self) -> float:
        return self.__balance
    
    def deposit(self, amount: float) -> bool:
        if amount <= 0:
            return False
        self.__balance += amount
        self.log_activity(f"Deposited {amount}, new balance: {self.__balance}")
        return True
    
    def withdraw(self, amount: float) -> bool:
        if amount <= 0 or amount > self.__balance:
            return False
        self.__balance -= amount
        self.log_activity(f"Withdrew {amount}, new balance: {self.__balance}")
        return True
    
    def add_transaction(self, transaction) -> None:
        self.__transactions.append(transaction)
    
    def log_activity(self, message: str) -> None:
        timestamp = datetime.now()
        self.__activity_log.append(f"{timestamp}: {message}")

class CheckingAccount(Account):
    def __init__(self, id: str, customer, balance: float = 0.0, overdraft_limit: float = 0.0):
        super().__init__(id, customer, balance)
        self.__overdraft_limit = overdraft_limit
        self.__monthly_fee = 10.0
    
    def withdraw(self, amount: float) -> bool:
        if amount <= 0:
            return False
        
        if self.get_balance() >= amount:
            return super().withdraw(amount)
        elif self.get_balance() + self.__overdraft_limit >= amount:
            self.log_activity(f"Using overdraft for withdrawal of {amount}")
            return super().withdraw(amount)
        else:
            return False
    
    def apply_monthly_fee(self) -> None:
        super().withdraw(self.__monthly_fee)
        self.log_activity(f"Applied monthly fee of {self.__monthly_fee}")

class SavingsAccount(Account):
    def __init__(self, id: str, customer, balance: float = 0.0, interest_rate: float = 0.01):
        super().__init__(id, customer, balance)
        self.__interest_rate = interest_rate
        self.__withdrawal_limit = 6
        self.__withdrawals_this_month = 0
    
    def withdraw(self, amount: float) -> bool:
        if self.__withdrawals_this_month >= self.__withdrawal_limit:
            self.log_activity(f"Withdrawal limit exceeded")
            return False
        
        if super().withdraw(amount):
            self.__withdrawals_this_month += 1
            return True
        return False
    
    def apply_interest(self) -> None:
        interest = self.get_balance() * self.__interest_rate
        super().deposit(interest)
        self.log_activity(f"Applied interest: {interest}")
    
    def reset_withdrawal_count(self) -> None:
        self.__withdrawals_this_month = 0
        self.log_activity("Reset monthly withdrawal count")

class Bank(IIdentifiable, ILoggable):
    def __init__(self, id: str, name: str):
        self.__id = id
        self.__name = name
        self.__branches = []
        self.__customers = set()
        self.__accounts = []
        self.__activity_log = []
    
    def get_id(self) -> str:
        return self.__id
    
    def add_branch(self, branch) -> None:
        self.__branches.append(branch)
    
    def remove_branch(self, branch) -> bool:
        if branch in self.__branches:
            self.__branches.remove(branch)
            return True
        return False
    
    def add_customer(self, customer) -> None:
        self.__customers.add(customer)
    
    def remove_customer(self, customer) -> bool:
        if customer in self.__customers:
            self.__customers.remove(customer)
            return True
        return False
    
    def register_account(self, account) -> None:
        self.__accounts.append(account)
    
    def log_activity(self, message: str) -> None:
        timestamp = datetime.now()
        self.__activity_log.append(f"{timestamp}: {message}")
    """
    
    return generate_uml_from_code(code=code, diagram_name="Banking System", save_path="improved_banking_system.png", figsize=(24, 18))

# Generate UML diagram from the banking example
diagram = complex_banking_example()

