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
        """Use networkx to create a non-overlapping layout"""
        # Create a graph where nodes are UML elements and edges are relationships
        G = nx.Graph()
        
        # Calculate proper heights based on content
        for element_id, element in self.elements.items():
            name_height = 30
            
            if isinstance(element, UMLClass):
                attr_height = max(25 * len(element.attributes), 15)
                method_height = max(25 * len(element.methods), 15)
                element.height = name_height + attr_height + method_height
            elif isinstance(element, UMLInterface):
                method_height = max(25 * len(element.methods), 15)
                element.height = name_height + method_height
            
            # Add node with size attributes
            G.add_node(element_id, width=element.width, height=element.height)
        
        # Add edges based on relationships
        for rel in self.relationships:
            G.add_edge(rel.source_id, rel.target_id)
        
        # Handle disconnected components
        if not nx.is_connected(G) and len(G.nodes) > 0:
            # For each disconnected component, arrange it separately
            components = list(nx.connected_components(G))
            
            # Layout each component with spring layout
            for i, component in enumerate(components):
                subgraph = G.subgraph(component)
                
                # Get positions with spring layout
                pos = nx.spring_layout(subgraph, k=0.5, iterations=100)
                
                # Scale and position this component
                scale_factor = 600  # Increase spacing between nodes
                offset_x = i * 1000  # Horizontal spacing between components
                offset_y = 0
                
                # Apply positions to elements
                for node_id, (x, y) in pos.items():
                    element = self.elements[node_id]
                    element.x = x * scale_factor + offset_x
                    element.y = y * scale_factor + offset_y
        else:
            # Use spring layout for the entire graph
            pos = nx.spring_layout(G, k=0.5, iterations=100)
            
            # Apply layout to elements
            scale_factor = 600  # Increase spacing between nodes
            for node_id, (x, y) in pos.items():
                element = self.elements[node_id]
                element.x = x * scale_factor
                element.y = y * scale_factor
        
        # Adjust positions to ensure no overlaps
        self._resolve_overlaps()
        
        # Special layout adjustments for inheritance hierarchies
        self._adjust_inheritance_hierarchies()
    
    def _resolve_overlaps(self):
        """Resolve any overlapping elements"""
        overlap_resolved = False
        iteration = 0
        max_iterations = 50
        
        while not overlap_resolved and iteration < max_iterations:
            overlap_resolved = True
            iteration += 1
            
            elements = list(self.elements.values())
            for i, elem1 in enumerate(elements):
                for elem2 in elements[i+1:]:
                    # Check if elements overlap
                    overlap_x = (elem1.x < elem2.x + elem2.width and 
                                elem1.x + elem1.width > elem2.x)
                    overlap_y = (elem1.y < elem2.y + elem2.height and 
                                elem1.y + elem1.height > elem2.y)
                    
                    if overlap_x and overlap_y:
                        overlap_resolved = False
                        
                        # Calculate overlap amounts
                        overlap_x_amount = min(elem1.x + elem1.width - elem2.x, 
                                              elem2.x + elem2.width - elem1.x)
                        overlap_y_amount = min(elem1.y + elem1.height - elem2.y, 
                                              elem2.y + elem2.height - elem1.y)
                        
                        # Push apart based on minimum overlap
                        if overlap_x_amount < overlap_y_amount:
                            # Push horizontally
                            if elem1.x < elem2.x:
                                elem1.x -= overlap_x_amount / 2 + 20  # Add extra space
                                elem2.x += overlap_x_amount / 2 + 20
                            else:
                                elem1.x += overlap_x_amount / 2 + 20
                                elem2.x -= overlap_x_amount / 2 + 20
                        else:
                            # Push vertically
                            if elem1.y < elem2.y:
                                elem1.y -= overlap_y_amount / 2 + 20
                                elem2.y += overlap_y_amount / 2 + 20
                            else:
                                elem1.y += overlap_y_amount / 2 + 20
                                elem2.y -= overlap_y_amount / 2 + 20
    
    def _adjust_inheritance_hierarchies(self):
        """Special adjustments for inheritance relationships"""
        # Group elements by inheritance relationships
        hierarchies = {}
        
        # Find all inheritance relationships
        for rel in self.relationships:
            if rel.relationship_type in [RelationType.INHERITANCE, RelationType.IMPLEMENTATION]:
                parent_id = rel.target_id
                child_id = rel.source_id
                
                if parent_id not in hierarchies:
                    hierarchies[parent_id] = []
                
                hierarchies[parent_id].append(child_id)
        
        # Adjust positions for each hierarchy
        for parent_id, children_ids in hierarchies.items():
            if parent_id in self.elements and children_ids:
                parent = self.elements[parent_id]
                
                # Arrange children in a row below parent
                child_width_total = sum(self.elements[cid].width for cid in children_ids if cid in self.elements)
                spacing = 50  # Space between children
                total_width = child_width_total + spacing * (len(children_ids) - 1)
                
                # Center the row of children under the parent
                start_x = parent.x + parent.width/2 - total_width/2
                
                # Position children
                current_x = start_x
                for child_id in children_ids:
                    if child_id in self.elements:
                        child = self.elements[child_id]
                        child.x = current_x
                        child.y = parent.y + parent.height + 100  # Place below parent
                        current_x += child.width + spacing
                
                # Re-check for overlaps after this adjustment
                self._resolve_overlaps()

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

def draw_uml_diagram(diagram, filename=None):
    """Draw a professional UML diagram with proper UML notation"""
    # Colors
    header_color = '#FFF59D'  # Light yellow
    bg_color = '#FFFFFF'      # White
    border_color = '#333333'  # Dark gray
    grid_color = '#EEEEEE'    # Light gray
    
    # Setup figure with a grid background
    fig_width = 16
    fig_height = 12
    dpi = 100
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='white')
    ax = fig.add_subplot(111)
    
    # Draw grid background
    ax.set_axisbelow(True)
    ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Calculate diagram boundaries for proper scaling
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for element in diagram.elements.values():
        min_x = min(min_x, element.x)
        max_x = max(max_x, element.x + element.width)
        min_y = min(min_y, element.y)
        max_y = max(max_y, element.y + element.height)
    
    # Add padding
    padding = 100
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding
    
    # Set axis limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)  # Invert y-axis for top-down layout
    
    # Turn off axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Store element information for relationship drawing
    element_info = {}
    
    # Draw UML elements
    for element_id, element in diagram.elements.items():
        # Calculate compartment heights
        name_height = 30
        attr_height = 0
        method_height = 0
        
        if isinstance(element, UMLClass):
            attr_height = 25 * max(1, len(element.attributes))
            method_height = 25 * max(1, len(element.methods))
        elif isinstance(element, UMLInterface):
            method_height = 25 * max(1, len(element.methods))
        
        total_height = name_height + attr_height + method_height
        
        # Store element dimensions
        element_info[element_id] = {
            'x': element.x,
            'y': element.y,
            'width': element.width,
            'height': total_height,
            'name_height': name_height,
            'attr_height': attr_height,
            'type': 'interface' if isinstance(element, UMLInterface) else 'class'
        }
        
        # Draw class box
        # Main rectangle
        rect = patches.Rectangle(
            (element.x, element.y),
            element.width, total_height,
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Class name box
        name_rect = patches.Rectangle(
            (element.x, element.y + total_height - name_height),
            element.width, name_height,
            facecolor=header_color,
            edgecolor=border_color,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(name_rect)
        
        # Draw class/interface name
        if isinstance(element, UMLClass) and element.abstract:
            name_text = f"«abstract» {element.name}"
            font_style = 'italic'
        elif isinstance(element, UMLInterface):
            name_text = f"«interface» {element.name}"
            font_style = 'normal'
        else:
            name_text = element.name
            font_style = 'normal'
        
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
        
        # Draw compartment dividers
        if attr_height > 0:
            ax.plot(
                [element.x, element.x + element.width],
                [element.y + method_height, element.y + method_height],
                color=border_color,
                linewidth=1.0,
                zorder=3
            )
        
        # Draw attributes
        if isinstance(element, UMLClass) and element.attributes:
            for i, attr in enumerate(element.attributes):
                attr_type_str = f": {attr.type_name}" if attr.type_name else ""
                attr_text = f"{attr.visibility.value}{attr.name}{attr_type_str}"
                
                ax.text(
                    element.x + 10,
                    element.y + method_height + (i + 0.5) * (attr_height / max(1, len(element.attributes))),
                    attr_text,
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=10,
                    family='monospace',
                    zorder=3
                )
        
        # Draw methods
        methods = element.methods
        if methods:
            for i, method in enumerate(methods):
                param_str = ""
                if method.params and len(method.params) > 0:
                    param_parts = []
                    for param in method.params:
                        param_text = param["name"]
                        if param.get("type"):
                            param_text += f": {param['type']}"
                        param_parts.append(param_text)
                    param_str = ", ".join(param_parts)
                    
                return_str = f": {method.return_type}" if method.return_type else ""
                method_text = f"{method.visibility.value}{method.name}({param_str}){return_str}"
                
                ax.text(
                    element.x + 10,
                    element.y + (i + 0.5) * (method_height / max(1, len(methods))),
                    method_text,
                    verticalalignment='center',
                    horizontalalignment='left',
                    fontsize=10,
                    family='monospace',
                    zorder=3
                )
    
    # Draw relationships with proper UML notation
    for rel in diagram.relationships:
        source_info = element_info.get(rel.source_id)
        target_info = element_info.get(rel.target_id)
        
        if not source_info or not target_info:
            continue
        
        # Calculate connection points
        source_rect = (source_info['x'], source_info['y'], source_info['width'], source_info['height'])
        target_rect = (target_info['x'], target_info['y'], target_info['width'], target_info['height'])
        
        connection = calculate_connection_points(source_rect, target_rect)
        sx, sy = connection['source']
        tx, ty = connection['target']
        
        # Draw different line styles based on relationship type
        if rel.relationship_type == RelationType.IMPLEMENTATION:
            linestyle = 'dashed'
        else:
            linestyle = 'solid'
        
        # Create path for the relationship line
        path_points = calculate_smart_path(
            (sx, sy), (tx, ty), 
            source_rect, target_rect, 
            diagram.elements
        )
        
        # Draw line segments of the path
        for i in range(len(path_points) - 1):
            x1, y1 = path_points[i]
            x2, y2 = path_points[i + 1]
            ax.plot([x1, x2], [y1, y2], 
                   color=border_color, 
                   linestyle=linestyle, 
                   linewidth=1.5,
                   zorder=1)
        
        # Draw the appropriate UML notation for the relationship type
        # The endpoint is the last point in the path
        end_point = path_points[-1]
        # Calculate angle from the second last point to the last point
        if len(path_points) >= 2:
            pre_end_point = path_points[-2]
            angle = calculate_angle(pre_end_point[0], pre_end_point[1], end_point[0], end_point[1])
        else:
            angle = calculate_angle(sx, sy, tx, ty)
        
        draw_relationship_end(ax, rel.relationship_type, end_point[0], end_point[1], angle)
        
        # Draw multiplicity labels near the endpoints
        if rel.source_multiplicity:
            # Position near the source end
            if len(path_points) >= 2:
                mx, my = get_label_position(path_points[0], path_points[1], 15)
            else:
                mx, my = get_label_position((sx, sy), (tx, ty), 15)
                
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
            # Position near the target end
            if len(path_points) >= 2:
                mx, my = get_label_position(path_points[-1], path_points[-2], 15)
            else:
                mx, my = get_label_position((tx, ty), (sx, sy), 15)
                
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
        
        # Draw relationship name/label if present
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
                mid_x = (sx + tx) / 2
                mid_y = (sy + ty) / 2
            
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
    
    # Add title
    if diagram.name:
        ax.text(
            0.5, 0.02,
            f"Class Diagram for {diagram.name}",
            transform=fig.transFigure,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=14,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def calculate_connection_points(source_rect, target_rect):
    """Calculate the best connection points between two rectangles"""
    sx, sy, sw, sh = source_rect
    tx, ty, tw, th = target_rect
    
    # Find centers
    scx, scy = sx + sw/2, sy + sh/2
    tcx, tcy = tx + tw/2, ty + th/2
    
    # Determine the closest sides for connection
    dx = tcx - scx
    dy = tcy - scy
    
    # Determine connection side
    if abs(dx) > abs(dy):
        # Connect horizontally (left or right sides)
        if dx > 0:
            # Source on left, target on right
            source_x = sx + sw
            source_y = scy
            target_x = tx
            target_y = tcy
        else:
            # Source on right, target on left
            source_x = sx
            source_y = scy
            target_x = tx + tw
            target_y = tcy
    else:
        # Connect vertically (top or bottom sides)
        if dy > 0:
            # Source above target
            source_x = scx
            source_y = sy + sh
            target_x = tcx
            target_y = ty
        else:
            # Source below target
            source_x = scx
            source_y = sy
            target_x = tcx
            target_y = ty + th
    
    return {
        'source': (source_x, source_y),
        'target': (target_x, target_y)
    }

def calculate_smart_path(source_point, target_point, source_rect, target_rect, elements_dict):
    """Calculate a smart path that avoids obstacles"""
    sx, sy = source_point
    tx, ty = target_point
    
    # Direct line - simplest case
    if is_direct_path_clear(source_point, target_point, source_rect, target_rect, elements_dict):
        return [source_point, target_point]
    
    # Otherwise, create a simplified version of a Manhattan path
    # Determine if a horizontal-first or vertical-first path would be better
    sx_rect, sy_rect, sw_rect, sh_rect = source_rect
    tx_rect, ty_rect, tw_rect, th_rect = target_rect
    
    # Calculate centers
    scx = sx_rect + sw_rect/2
    scy = sy_rect + sh_rect/2
    tcx = tx_rect + tw_rect/2
    tcy = ty_rect + th_rect/2
    
    dx = abs(tcx - scx)
    dy = abs(tcy - scy)
    
    if dx > dy:
        # Horizontal-first path might be better
        mid_x = (sx + tx) / 2
        path = [source_point, (mid_x, sy), (mid_x, ty), target_point]
    else:
        # Vertical-first path might be better
        mid_y = (sy + ty) / 2
        path = [source_point, (sx, mid_y), (tx, mid_y), target_point]
    
    # Simplify path if possible
    return simplify_path(path, source_rect, target_rect, elements_dict)

def is_direct_path_clear(p1, p2, source_rect, target_rect, elements_dict):
    """Check if a direct path between points is clear of obstacles"""
    # Only need to check elements that might be between the source and target
    x1, y1 = p1
    x2, y2 = p2
    
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)
    
    sx_rect, sy_rect, sw_rect, sh_rect = source_rect
    tx_rect, ty_rect, tw_rect, th_rect = target_rect
    
    for element in elements_dict.values():
        # Skip source and target elements
        if (element.x == sx_rect and element.y == sy_rect) or \
           (element.x == tx_rect and element.y == ty_rect):
            continue
        
        # Check if element overlaps with the path bounding box
        if (element.x + element.width >= min_x and element.x <= max_x and
            element.y + element.height >= min_y and element.y <= max_y):
            
            # More detailed check for line-rectangle intersection
            # Simplified: If the rectangle contains any point on the line, there's an intersection
            for t in np.linspace(0, 1, 10):  # Check 10 points along the line
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                
                if (element.x <= px <= element.x + element.width and
                    element.y <= py <= element.y + element.height):
                    return False
    
    return True

def simplify_path(path, source_rect, target_rect, elements_dict):
    """Simplify a path by removing unnecessary points"""
    if len(path) <= 2:
        return path
    
    result = [path[0]]
    i = 0
    
    while i < len(path) - 1:
        # Check if we can directly connect to a further point
        for j in range(len(path) - 1, i, -1):
            if is_direct_path_clear(path[i], path[j], source_rect, target_rect, elements_dict):
                result.append(path[j])
                i = j
                break
        else:
            # If no direct path found, keep the next point
            result.append(path[i + 1])
            i += 1
    
    return result

def calculate_angle(x1, y1, x2, y2):
    """Calculate angle between two points in degrees"""
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def get_label_position(point1, point2, offset=10):
    """Calculate position for a label near a line endpoint"""
    x1, y1 = point1
    x2, y2 = point2
    
    angle = np.arctan2(y2 - y1, x2 - x1)
    
    # Position label perpendicular to the line
    perp_angle = angle + np.pi/2
    
    label_x = x1 + offset * np.cos(perp_angle)
    label_y = y1 + offset * np.sin(perp_angle)
    
    return label_x, label_y

def draw_relationship_end(ax, relationship_type, x, y, angle):
    """Draw the appropriate UML notation at the end of a relationship line"""
    if relationship_type == RelationType.INHERITANCE or relationship_type == RelationType.IMPLEMENTATION:
        # Open triangle arrow for inheritance/implementation
        draw_inheritance_arrow(ax, x, y, angle)
    elif relationship_type == RelationType.ASSOCIATION:
        # No decoration for basic association
        pass
    elif relationship_type == RelationType.DEPENDENCY:
        # Open arrow for dependency
        draw_dependency_arrow(ax, x, y, angle)
    elif relationship_type == RelationType.AGGREGATION:
        # Open diamond for aggregation
        draw_diamond(ax, x, y, angle, filled=False)
    elif relationship_type == RelationType.COMPOSITION:
        # Filled diamond for composition
        draw_diamond(ax, x, y, angle, filled=True)

def draw_inheritance_arrow(ax, x, y, angle):
    """Draw the standard UML inheritance/implementation arrow (open triangle)"""
    arrow_size = 15
    angle_rad = np.radians(angle)
    
    # Points for the arrow (triangle)
    # The tip is at (x,y)
    p1 = (x, y)
    
    # Calculate the two base points of the triangle
    p2 = (
        x - arrow_size * np.cos(angle_rad) + arrow_size/2 * np.sin(angle_rad),
        y - arrow_size * np.sin(angle_rad) - arrow_size/2 * np.cos(angle_rad)
    )
    
    p3 = (
        x - arrow_size * np.cos(angle_rad) - arrow_size/2 * np.sin(angle_rad),
        y - arrow_size * np.sin(angle_rad) + arrow_size/2 * np.cos(angle_rad)
    )
    
    # Draw hollow triangle
    triangle = patches.Polygon(
        [p1, p2, p3], 
        closed=True, 
        fill=True, 
        facecolor='white', 
        edgecolor='black', 
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(triangle)

def draw_dependency_arrow(ax, x, y, angle):
    """Draw the standard UML dependency arrow (open arrow)"""
    arrow_size = 15
    angle_rad = np.radians(angle)
    
    # Calculate the points for the arrow
    # The tip is at (x,y)
    p1 = (x, y)
    
    # Calculate the two ends of the arrow
    p2 = (
        x - arrow_size * np.cos(angle_rad) + arrow_size/3 * np.sin(angle_rad),
        y - arrow_size * np.sin(angle_rad) - arrow_size/3 * np.cos(angle_rad)
    )
    
    p3 = (
        x - arrow_size * np.cos(angle_rad) - arrow_size/3 * np.sin(angle_rad),
        y - arrow_size * np.sin(angle_rad) + arrow_size/3 * np.cos(angle_rad)
    )
    
    # Draw the two lines of the open arrow
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1.5, zorder=2)
    ax.plot([p1[0], p3[0]], [p1[1], p3[1]], color='black', linewidth=1.5, zorder=2)

def draw_diamond(ax, x, y, angle, filled=False):
    """Draw the standard UML diamond for aggregation/composition"""
    diamond_size = 15
    angle_rad = np.radians(angle)
    
    # Calculate the points for the diamond
    # The tip is at (x,y)
    p1 = (x, y)
    
    # Calculate the side points
    p2 = (
        x - diamond_size/2 * np.cos(angle_rad) + diamond_size/2 * np.sin(angle_rad),
        y - diamond_size/2 * np.sin(angle_rad) - diamond_size/2 * np.cos(angle_rad)
    )
    
    p3 = (
        x - diamond_size * np.cos(angle_rad),
        y - diamond_size * np.sin(angle_rad)
    )
    
    p4 = (
        x - diamond_size/2 * np.cos(angle_rad) - diamond_size/2 * np.sin(angle_rad),
        y - diamond_size/2 * np.sin(angle_rad) + diamond_size/2 * np.cos(angle_rad)
    )
    
    # Draw the diamond
    facecolor = 'black' if filled else 'white'
    diamond = patches.Polygon(
        [p1, p2, p3, p4], 
        closed=True, 
        fill=True, 
        facecolor=facecolor, 
        edgecolor='black', 
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(diamond)

def create_bank_example():
    """Create banking system diagram with proper layout"""
    diagram = UMLDiagram("Banking System")
    
    # Create classes with appropriate attributes
    bank = UMLClass("Bank")
    bank.add_attribute("BankId", "int", Visibility.PUBLIC)
    bank.add_attribute("Name", "string", Visibility.PUBLIC)
    bank.add_attribute("Location", "string", Visibility.PUBLIC)
    
    customer = UMLClass("Customer")
    customer.add_attribute("Id", "int", Visibility.PUBLIC)
    customer.add_attribute("Name", "string", Visibility.PUBLIC)
    customer.add_attribute("Adress", "string", Visibility.PUBLIC)
    customer.add_attribute("PhoneNo", "int", Visibility.PUBLIC)
    customer.add_attribute("AcctNo", "int", Visibility.PUBLIC)
    customer.add_method("GeneralInquiry", None, [], Visibility.PUBLIC)
    customer.add_method("DepositMoney", None, [], Visibility.PUBLIC)
    customer.add_method("WithdrawMoney", None, [], Visibility.PUBLIC)
    customer.add_method("OpenAccount", None, [], Visibility.PUBLIC)
    customer.add_method("CloseAccount", None, [], Visibility.PUBLIC)
    customer.add_method("ApplyForLoan", None, [], Visibility.PUBLIC)
    customer.add_method("RequestCard", None, [], Visibility.PUBLIC)
    
    teller = UMLClass("Teller")
    teller.add_attribute("Id", "int", Visibility.PUBLIC)
    teller.add_attribute("Name", "string", Visibility.PUBLIC)
    teller.add_method("CollectMoney", None, [], Visibility.PUBLIC)
    teller.add_method("OpenAccount", None, [], Visibility.PUBLIC)
    teller.add_method("CloseAccount", None, [], Visibility.PUBLIC)
    teller.add_method("LoanRequest", None, [], Visibility.PUBLIC)
    teller.add_method("ProvideInfo", None, [], Visibility.PUBLIC)
    teller.add_method("IssueCard", None, [], Visibility.PUBLIC)
    
    account = UMLClass("Account")
    account.add_attribute("Id", "int", Visibility.PUBLIC)
    account.add_attribute("CustomerId", "int", Visibility.PUBLIC)
    
    checking = UMLClass("Checking")
    checking.add_attribute("Id", "int", Visibility.PUBLIC)
    checking.add_attribute("CustomerId", "int", Visibility.PUBLIC)
    
    savings = UMLClass("Savings")
    savings.add_attribute("Id", "int", Visibility.PUBLIC)
    savings.add_attribute("CustomerId", "int", Visibility.PUBLIC)
    
    loan = UMLClass("Loan")
    loan.add_attribute("Id", "int", Visibility.PUBLIC)
    loan.add_attribute("Type", "string", Visibility.PUBLIC)
    loan.add_attribute("AccountId", "int", Visibility.PUBLIC)
    loan.add_attribute("CustomerId", "int", Visibility.PUBLIC)
    
    # Add elements to diagram
    diagram.add_element(bank)
    diagram.add_element(customer)
    diagram.add_element(teller)
    diagram.add_element(account)
    diagram.add_element(checking)
    diagram.add_element(savings)
    diagram.add_element(loan)
    
    # Add relationships with proper multiplicities
    diagram.add_relationship(customer.id, bank.id, RelationType.ASSOCIATION, "1..*", "1")
    diagram.add_relationship(teller.id, bank.id, RelationType.ASSOCIATION, "1..*", "1")
    diagram.add_relationship(account.id, customer.id, RelationType.ASSOCIATION, "1..*", "1")
    diagram.add_relationship(checking.id, account.id, RelationType.INHERITANCE)
    diagram.add_relationship(savings.id, account.id, RelationType.INHERITANCE)
    diagram.add_relationship(loan.id, customer.id, RelationType.ASSOCIATION, "0..*", "1")
    diagram.add_relationship(teller.id, account.id, RelationType.ASSOCIATION, "1..*", "1..*")
    
    # Auto layout
    diagram.auto_layout()
    
    return diagram

def generate_uml_from_code(code=None, file_path=None, directory_path=None, save_path=None):
    """Generate UML diagram from Python code"""
    parser = UMLParser()
    
    if code:
        parser.parse_text(code)
    elif file_path:
        parser.parse_file(file_path)
    elif directory_path:
        parser.parse_directory(directory_path)
    else:
        return None
    
    diagram = parser.get_diagram()
    fig = draw_uml_diagram(diagram, save_path)
    plt.show()
    return diagram

def show_bank_example():
    """Show the banking system example with improved layout"""
    diagram = create_bank_example()
    fig = draw_uml_diagram(diagram)
    plt.show()
    return diagram

def create_animal_example():
    """Create an example Animal hierarchy diagram"""
    diagram = UMLDiagram("Animal Hierarchy")
    
    # Interface
    pet_interface = UMLInterface("IPet")
    pet_interface.add_method("play", "None", [], Visibility.PUBLIC)
    
    # Abstract class
    animal = UMLClass("Animal", abstract=True)
    animal.add_attribute("name", "str", Visibility.PROTECTED)
    animal.add_method("make_sound", "str", [], Visibility.PUBLIC)
    animal.add_method("get_name", "str", [], Visibility.PUBLIC)
    
    # Concrete classes
    dog = UMLClass("Dog")
    dog.add_attribute("breed", "str", Visibility.PRIVATE)
    dog.add_method("make_sound", "str", [], Visibility.PUBLIC)
    dog.add_method("get_breed", "str", [], Visibility.PUBLIC)
    
    cat = UMLClass("Cat")
    cat.add_attribute("color", "str", Visibility.PRIVATE)
    cat.add_method("make_sound", "str", [], Visibility.PUBLIC)
    cat.add_method("get_color", "str", [], Visibility.PUBLIC)
    
    # Add to diagram
    diagram.add_element(pet_interface)
    diagram.add_element(animal)
    diagram.add_element(dog)
    diagram.add_element(cat)
    
    # Add relationships
    diagram.add_relationship(dog.id, animal.id, RelationType.INHERITANCE)
    diagram.add_relationship(cat.id, animal.id, RelationType.INHERITANCE)
    diagram.add_relationship(dog.id, pet_interface.id, RelationType.IMPLEMENTATION)
    diagram.add_relationship(cat.id, pet_interface.id, RelationType.IMPLEMENTATION)
    diagram.add_relationship(dog.id, cat.id, RelationType.ASSOCIATION, "1", "*", "chases")
    
    # Auto layout
    diagram.auto_layout()
    
    return diagram

def show_animal_example():
    """Show animal hierarchy example"""
    diagram = create_animal_example()
    fig = draw_uml_diagram(diagram)
    plt.show()
    return diagram


# Then you can show the banking system example
show_bank_example()

# Or try the animal hierarchy example
show_animal_example()
