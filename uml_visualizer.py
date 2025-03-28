import uuid
import ast
import os
import re
import tempfile
from enum import Enum
from typing import List, Dict, Any, Set, Optional, Union
from flask import Flask, render_template_string, request, jsonify

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
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }

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
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type_name": self.type_name,
            "visibility": self.visibility.value,
            "static": self.static,
            "attribute_type": self.attribute_type.value,
            "params": self.params,
            "return_type": self.return_type
        }

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
    
    def to_dict(self):
        result = super().to_dict()
        result.update({
            "type": "class",
            "abstract": self.abstract,
            "attributes": [attr.to_dict() for attr in self.attributes],
            "methods": [method.to_dict() for method in self.methods]
        })
        return result

class UMLInterface(UMLElement):
    def __init__(self, name):
        super().__init__(name)
        self.methods = []
    
    def add_method(self, name, return_type=None, params=None, visibility=Visibility.PUBLIC):
        method = UMLAttribute(name, "", visibility, False, AttributeType.METHOD, params, return_type)
        self.methods.append(method)
        return method
    
    def to_dict(self):
        result = super().to_dict()
        result.update({
            "type": "interface",
            "methods": [method.to_dict() for method in self.methods]
        })
        return result

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
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.relationship_type.value,
            "source_multiplicity": self.source_multiplicity,
            "target_multiplicity": self.target_multiplicity,
            "label": self.label
        }

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
    
    def to_dict(self):
        return {
            "name": self.name,
            "elements": [el.to_dict() for el in self.elements.values()],
            "relationships": [rel.to_dict() for rel in self.relationships]
        }
    
    def auto_layout(self):
        elements = list(self.elements.values())
        if not elements:
            return
        grid_size = int(len(elements) ** 0.5) + 1
        spacing_x, spacing_y = 300, 200
        for i, element in enumerate(elements):
            row, col = i // grid_size, i % grid_size
            element.x = col * spacing_x + 50
            element.y = row * spacing_y + 50

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

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UML Class Diagram Visualizer</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body { font-family: 'Arial', sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
        .header { background-color: #333; color: white; padding: 1rem; }
        .content { display: flex; flex: 1; overflow: hidden; }
        .sidebar { width: 300px; background-color: #f5f5f5; padding: 1rem; display: flex; flex-direction: column; overflow: auto; }
        .main { flex: 1; overflow: auto; background-color: #eee; position: relative; }
        .diagram-container { width: 100%; height: 100%; overflow: auto; }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; }
        .form-control { width: 100%; padding: 0.5rem; box-sizing: border-box; }
        .btn { background-color: #4CAF50; border: none; color: white; padding: 0.5rem 1rem; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 0.5rem 0; cursor: pointer; border-radius: 4px; }
        .btn:hover { background-color: #45a049; }
        .tab-container { display: flex; margin-bottom: 1rem; }
        .tab { padding: 0.5rem 1rem; cursor: pointer; background-color: #ddd; margin-right: 0.5rem; border-radius: 4px 4px 0 0; }
        .tab.active { background-color: #f5f5f5; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .uml-class, .uml-interface { fill: white; stroke: #333; stroke-width: 2; }
        .uml-class text, .uml-interface text { font-family: 'Arial', sans-serif; }
        .uml-class-name, .uml-interface-name { font-weight: bold; text-anchor: middle; }
        .uml-attribute, .uml-method { text-anchor: start; font-size: 12px; }
        .uml-divider { stroke: #333; stroke-width: 1; }
        .relationship { stroke: #333; stroke-width: 2; fill: none; }
        .relationship-text { font-size: 12px; text-anchor: middle; }
        .control-panel { position: absolute; top: 10px; right: 10px; background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0, 0, 0, 0.2); }
        .tooltip { position: absolute; background-color: rgba(0, 0, 0, 0.8); color: white; padding: 5px; border-radius: 3px; font-size: 12px; pointer-events: none; display: none; max-width: 300px; z-index: 1000; }
    </style>
</head>
<body>
    <div class="header">
        <h1>UML Class Diagram Visualizer</h1>
    </div>
    <div class="content">
        <div class="sidebar">
            <div class="tab-container">
                <div class="tab active" data-tab="code">Code Input</div>
                <div class="tab" data-tab="file">File Upload</div>
                <div class="tab" data-tab="example">Example</div>
            </div>
            <div class="tab-content active" id="code-tab">
                <div class="form-group">
                    <label for="code-input">Python Code:</label>
                    <textarea id="code-input" class="form-control" rows="15" placeholder="Paste your Python code here..."></textarea>
                </div>
                <button id="parse-code-btn" class="btn">Generate Diagram</button>
            </div>
            <div class="tab-content" id="file-tab">
                <div class="form-group">
                    <label for="file-input">Upload Python File:</label>
                    <input type="file" id="file-input" class="form-control" accept=".py">
                </div>
                <button id="parse-file-btn" class="btn">Generate Diagram</button>
            </div>
            <div class="tab-content" id="example-tab">
                <p>Click the button below to load an example UML diagram:</p>
                <button id="load-example-btn" class="btn">Load Example</button>
            </div>
        </div>
        <div class="main">
            <div id="diagram-container" class="diagram-container"></div>
            <div class="tooltip" id="tooltip"></div>
            <div class="control-panel">
                <button id="zoom-in-btn" class="btn"><i class="fas fa-search-plus"></i></button>
                <button id="zoom-out-btn" class="btn"><i class="fas fa-search-minus"></i></button>
                <button id="reset-zoom-btn" class="btn"><i class="fas fa-sync-alt"></i></button>
                <button id="download-btn" class="btn"><i class="fas fa-download"></i></button>
            </div>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.0.0/d3.min.js"></script>
    <script>
        class UMLRenderer {
            constructor(svgContainerId, tooltipId) {
                this.svgContainer = d3.select(`#${svgContainerId}`);
                this.tooltip = d3.select(`#${tooltipId}`);
                this.svg = null;
                this.diagram = null;
                this.zoom = d3.zoom().scaleExtent([0.1, 4]).on("zoom", this.handleZoom.bind(this));
                this.init();
            }
            init() {
                this.svg = this.svgContainer.append("svg")
                    .attr("width", "100%")
                    .attr("height", "100%")
                    .call(this.zoom);
                this.svgGroup = this.svg.append("g");
                this.relationshipGroup = this.svgGroup.append("g").attr("class", "relationships");
                this.elementGroup = this.svgGroup.append("g").attr("class", "elements");
            }
            handleZoom(event) {
                this.svgGroup.attr("transform", event.transform);
            }
            zoomIn() {
                this.svg.transition().duration(300).call(this.zoom.scaleBy, 1.2);
            }
            zoomOut() {
                this.svg.transition().duration(300).call(this.zoom.scaleBy, 0.8);
            }
            resetZoom() {
                this.svg.transition().duration(300).call(
                    this.zoom.transform,
                    d3.zoomIdentity
                );
            }
            render(diagram) {
                this.diagram = diagram;
                this.clear();
                this.renderElements();
                this.renderRelationships();
                this.fitDiagramToView();
            }
            clear() {
                this.elementGroup.selectAll("*").remove();
                this.relationshipGroup.selectAll("*").remove();
            }
            renderElements() {
                const self = this;
                this.diagram.elements.forEach(element => {
                    if (element.type === "class") {
                        this.renderClass(element);
                    } else if (element.type === "interface") {
                        this.renderInterface(element);
                    }
                });
            }
            renderClass(classElement) {
                const nameHeight = 30;
                const attributeHeight = 20 * Math.max(1, classElement.attributes.length);
                const methodHeight = 20 * Math.max(1, classElement.methods.length);
                const totalHeight = nameHeight + attributeHeight + methodHeight;
                const g = this.elementGroup.append("g")
                    .attr("transform", `translate(${classElement.x}, ${classElement.y})`)
                    .attr("class", "uml-element")
                    .attr("data-id", classElement.id);
                g.append("rect")
                    .attr("class", "uml-class")
                    .attr("width", classElement.width)
                    .attr("height", totalHeight)
                    .attr("rx", 3)
                    .attr("ry", 3);
                g.append("text")
                    .attr("class", "uml-class-name")
                    .attr("x", classElement.width / 2)
                    .attr("y", 20)
                    .text(() => {
                        let name = classElement.name;
                        if (classElement.abstract) {
                            name = `«abstract» ${name}`;
                        }
                        return name;
                    })
                    .attr("font-style", classElement.abstract ? "italic" : "normal");
                g.append("line")
                    .attr("class", "uml-divider")
                    .attr("x1", 0)
                    .attr("y1", nameHeight)
                    .attr("x2", classElement.width)
                    .attr("y2", nameHeight);
                const attributeGroup = g.append("g")
                    .attr("transform", `translate(10, ${nameHeight})`);
                classElement.attributes.forEach((attr, i) => {
                    attributeGroup.append("text")
                        .attr("class", "uml-attribute")
                        .attr("x", 0)
                        .attr("y", (i + 1) * 20)
                        .text(`${attr.visibility} ${attr.name}${attr.type_name ? ` : ${attr.type_name}` : ""}`);
                });
                g.append("line")
                    .attr("class", "uml-divider")
                    .attr("x1", 0)
                    .attr("y1", nameHeight + attributeHeight)
                    .attr("x2", classElement.width)
                    .attr("y2", nameHeight + attributeHeight);
                const methodGroup = g.append("g")
                    .attr("transform", `translate(10, ${nameHeight + attributeHeight})`);
                classElement.methods.forEach((method, i) => {
                    let methodText = `${method.visibility} ${method.name}(`;
                    if (method.params && method.params.length > 0) {
                        methodText += method.params.map(p => `${p.name}${p.type ? ` : ${p.type}` : ""}`).join(", ");
                    }
                    methodText += `)${method.return_type ? ` : ${method.return_type}` : ""}`;
                    methodGroup.append("text")
                        .attr("class", "uml-method")
                        .attr("x", 0)
                        .attr("y", (i + 1) * 20)
                        .text(methodText);
                });
                this.setupTooltip(g, classElement);
                this.setupDrag(g, classElement);
            }
            renderInterface(interfaceElement) {
                const nameHeight = 30;
                const methodHeight = 20 * Math.max(1, interfaceElement.methods.length);
                const totalHeight = nameHeight + methodHeight;
                const g = this.elementGroup.append("g")
                    .attr("transform", `translate(${interfaceElement.x}, ${interfaceElement.y})`)
                    .attr("class", "uml-element")
                    .attr("data-id", interfaceElement.id);
                g.append("rect")
                    .attr("class", "uml-interface")
                    .attr("width", interfaceElement.width)
                    .attr("height", totalHeight)
                    .attr("rx", 3)
                    .attr("ry", 3);
                g.append("text")
                    .attr("class", "uml-interface-name")
                    .attr("x", interfaceElement.width / 2)
                    .attr("y", 20)
                    .text(`«interface» ${interfaceElement.name}`);
                g.append("line")
                    .attr("class", "uml-divider")
                    .attr("x1", 0)
                    .attr("y1", nameHeight)
                    .attr("x2", interfaceElement.width)
                    .attr("y2", nameHeight);
                const methodGroup = g.append("g")
                    .attr("transform", `translate(10, ${nameHeight})`);
                interfaceElement.methods.forEach((method, i) => {
                    let methodText = `${method.visibility} ${method.name}(`;
                    if (method.params && method.params.length > 0) {
                        methodText += method.params.map(p => `${p.name}${p.type ? ` : ${p.type}` : ""}`).join(", ");
                    }
                    methodText += `)${method.return_type ? ` : ${method.return_type}` : ""}`;
                    methodGroup.append("text")
                        .attr("class", "uml-method")
                        .attr("x", 0)
                        .attr("y", (i + 1) * 20)
                        .text(methodText);
                });
                this.setupTooltip(g, interfaceElement);
                this.setupDrag(g, interfaceElement);
            }
            renderRelationships() {
                const self = this;
                this.diagram.relationships.forEach(relationship => {
                    const source = this.findElementById(relationship.source_id);
                    const target = this.findElementById(relationship.target_id);
                    if (!source || !target) return;
                    const sourceRect = this.getElementRectById(relationship.source_id);
                    const targetRect = this.getElementRectById(relationship.target_id);
                    if (!sourceRect || !targetRect) return;
                    const { sourcePoint, targetPoint } = this.calculateConnectionPoints(sourceRect, targetRect);
                    this.drawRelationship(relationship, sourcePoint, targetPoint, sourceRect, targetRect);
                });
            }
            setupTooltip(element, data) {
                const self = this;
                element
                    .on("mouseover", function(event) {
                        let tooltipContent = `<strong>${data.type === "interface" ? "Interface" : "Class"}: ${data.name}</strong><br>`;
                        if (data.type === "class" && data.abstract) {
                            tooltipContent += `<em>Abstract</em><br>`;
                        }
                        if (data.type === "class" && data.attributes.length > 0) {
                            tooltipContent += `<strong>Attributes:</strong><br>`;
                            data.attributes.forEach(attr => {
                                tooltipContent += `${attr.visibility} ${attr.name}${attr.type_name ? ` : ${attr.type_name}` : ""}<br>`;
                            });
                        }
                        tooltipContent += `<strong>Methods:</strong><br>`;
                        const methods = data.type === "class" ? data.methods : data.methods;
                        methods.forEach(method => {
                            let methodText = `${method.visibility} ${method.name}(`;
                            if (method.params && method.params.length > 0) {
                                methodText += method.params.map(p => `${p.name}${p.type ? ` : ${p.type}` : ""}`).join(", ");
                            }
                            methodText += `)${method.return_type ? ` : ${method.return_type}` : ""}`;
                            tooltipContent += `${methodText}<br>`;
                        });
                        self.tooltip
                            .html(tooltipContent)
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY + 10) + "px")
                            .style("display", "block");
                    })
                    .on("mouseout", function() {
                        self.tooltip.style("display", "none");
                    })
                    .on("mousemove", function(event) {
                        self.tooltip
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY + 10) + "px");
                    });
            }
            setupDrag(element, data) {
                const self = this;
                const dragHandler = d3.drag()
                    .on("start", function(event) {
                        d3.select(this).raise().attr("stroke", "black");
                    })
                    .on("drag", function(event) {
                        const transform = d3.select(this).attr("transform");
                        const translate = transform.substring(transform.indexOf("(") + 1, transform.indexOf(")")).split(",");
                        const x = parseFloat(translate[0]) + event.dx;
                        const y = parseFloat(translate[1]) + event.dy;
                        d3.select(this).attr("transform", `translate(${x}, ${y})`);
                        data.x = x;
                        data.y = y;
                        self.renderRelationships();
                    })
                    .on("end", function(event) {
                        d3.select(this).attr("stroke", null);
                    });
                element.call(dragHandler);
            }
            findElementById(id) {
                return this.diagram.elements.find(el => el.id === id);
            }
            getElementRectById(id) {
                const element = this.findElementById(id);
                if (!element) return null;
                const node = this.elementGroup.select(`[data-id="${id}"]`).node();
                if (!node) return null;
                const bounds = node.getBBox();
                return {
                    x: element.x,
                    y: element.y,
                    width: bounds.width,
                    height: bounds.height,
                    element: element
                };
            }
            calculateConnectionPoints(sourceRect, targetRect) {
                const sourceCenter = {
                    x: sourceRect.x + sourceRect.width / 2,
                    y: sourceRect.y + sourceRect.height / 2
                };
                const targetCenter = {
                    x: targetRect.x + targetRect.width / 2,
                    y: targetRect.y + targetRect.height / 2
                };
                const dx = targetCenter.x - sourceCenter.x;
                const dy = targetCenter.y - sourceCenter.y;
                const angle = Math.atan2(dy, dx);
                let sourcePoint, targetPoint;
                if (Math.abs(Math.cos(angle)) > Math.abs(Math.sin(angle))) {
                    if (dx > 0) {
                        sourcePoint = {
                            x: sourceRect.x + sourceRect.width,
                            y: sourceRect.y + sourceRect.height / 2
                        };
                        targetPoint = {
                            x: targetRect.x,
                            y: targetRect.y + targetRect.height / 2
                        };
                    } else {
                        sourcePoint = {
                            x: sourceRect.x,
                            y: sourceRect.y + sourceRect.height / 2
                        };
                        targetPoint = {
                            x: targetRect.x + targetRect.width,
                            y: targetRect.y + targetRect.height / 2
                        };
                    }
                } else {
                    if (dy > 0) {
                        sourcePoint = {
                            x: sourceRect.x + sourceRect.width / 2,
                            y: sourceRect.y + sourceRect.height
                        };
                        targetPoint = {
                            x: targetRect.x + targetRect.width / 2,
                            y: targetRect.y
                        };
                    } else {
                        sourcePoint = {
                            x: sourceRect.x + sourceRect.width / 2,
                            y: sourceRect.y
                        };
                        targetPoint = {
                            x: targetRect.x + targetRect.width / 2,
                            y: targetRect.y + targetRect.height
                        };
                    }
                }
                return { sourcePoint, targetPoint };
            }
            drawRelationship(relationship, sourcePoint, targetPoint, sourceRect, targetRect) {
                const relationshipGroup = this.relationshipGroup.append("g")
                    .attr("class", "relationship-group");
                const pathData = this.generatePathData(relationship, sourcePoint, targetPoint);
                const relationshipPath = relationshipGroup.append("path")
                    .attr("class", "relationship")
                    .attr("d", pathData);
                this.drawRelationshipEnd(relationshipGroup, relationship, sourcePoint, targetPoint);
                if (relationship.label) {
                    const midPoint = this.getMidPoint(sourcePoint, targetPoint);
                    relationshipGroup.append("text")
                        .attr("class", "relationship-text")
                        .attr("x", midPoint.x)
                        .attr("y", midPoint.y - 10)
                        .text(relationship.label);
                }
                if (relationship.source_multiplicity) {
                    const quarterPoint = this.getPointAlongLine(sourcePoint, targetPoint, 0.1);
                    relationshipGroup.append("text")
                        .attr("class", "relationship-text")
                        .attr("x", quarterPoint.x + 10)
                        .attr("y", quarterPoint.y - 10)
                        .text(relationship.source_multiplicity);
                }
                if (relationship.target_multiplicity) {
                    const threeQuarterPoint = this.getPointAlongLine(sourcePoint, targetPoint, 0.9);
                    relationshipGroup.append("text")
                        .attr("class", "relationship-text")
                        .attr("x", threeQuarterPoint.x + 10)
                        .attr("y", threeQuarterPoint.y - 10)
                        .text(relationship.target_multiplicity);
                }
            }
            generatePathData(relationship, sourcePoint, targetPoint) {
                return `M${sourcePoint.x},${sourcePoint.y} L${targetPoint.x},${targetPoint.y}`;
            }
            drawRelationshipEnd(relationshipGroup, relationship, sourcePoint, targetPoint) {
                const dx = targetPoint.x - sourcePoint.x;
                const dy = targetPoint.y - sourcePoint.y;
                const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                const arrowSize = 10;
                switch (relationship.type) {
                    case "inheritance":
                        this.drawInheritanceArrow(relationshipGroup, targetPoint, angle, arrowSize);
                        break;
                    case "implementation":
                        this.drawImplementationArrow(relationshipGroup, targetPoint, angle, arrowSize);
                        break;
                    case "aggregation":
                        this.drawAggregationDiamond(relationshipGroup, sourcePoint, angle, arrowSize);
                        break;
                    case "composition":
                        this.drawCompositionDiamond(relationshipGroup, sourcePoint, angle, arrowSize);
                        break;
                    case "dependency":
                        this.drawDependencyArrow(relationshipGroup, targetPoint, angle, arrowSize);
                        break;
                    case "association":
                        break;
                }
            }
            drawInheritanceArrow(group, point, angle, size) {
                const arrowPoints = this.calculateArrowPoints(point, angle, size);
                group.append("polygon")
                    .attr("points", arrowPoints.join(" "))
                    .attr("transform", `rotate(${angle + 90}, ${point.x}, ${point.y})`)
                    .attr("fill", "white")
                    .attr("stroke", "#333")
                    .attr("stroke-width", 2);
            }
            drawImplementationArrow(group, point, angle, size) {
                const arrowPoints = this.calculateArrowPoints(point, angle, size);
                group.append("polygon")
                    .attr("points", arrowPoints.join(" "))
                    .attr("transform", `rotate(${angle + 90}, ${point.x}, ${point.y})`)
                    .attr("fill", "white")
                    .attr("stroke", "#333")
                    .attr("stroke-width", 2);
                const lineDashOffset = size * 2;
                const lineLength = 100;
                const lineX1 = point.x + lineDashOffset * Math.cos((angle - 180) * Math.PI / 180);
                const lineY1 = point.y + lineDashOffset * Math.sin((angle - 180) * Math.PI / 180);
                const lineX2 = lineX1 + lineLength * Math.cos((angle - 180) * Math.PI / 180);
                const lineY2 = lineY1 + lineLength * Math.sin((angle - 180) * Math.PI / 180);
                group.select("path")
                    .attr("stroke-dasharray", "5,5");
            }
            drawAggregationDiamond(group, point, angle, size) {
                const diamondPoints = this.calculateDiamondPoints(point, angle, size);
                group.append("polygon")
                    .attr("points", diamondPoints.join(" "))
                    .attr("transform", `rotate(${angle}, ${point.x}, ${point.y})`)
                    .attr("fill", "white")
                    .attr("stroke", "#333")
                    .attr("stroke-width", 2);
            }
            drawCompositionDiamond(group, point, angle, size) {
                const diamondPoints = this.calculateDiamondPoints(point, angle, size);
                group.append("polygon")
                    .attr("points", diamondPoints.join(" "))
                    .attr("transform", `rotate(${angle}, ${point.x}, ${point.y})`)
                    .attr("fill", "black")
                    .attr("stroke", "#333")
                    .attr("stroke-width", 2);
            }
            drawDependencyArrow(group, point, angle, size) {
                const arrowPoints = this.calculateArrowPoints(point, angle, size);
                group.append("polygon")
                    .attr("points", arrowPoints.join(" "))
                    .attr("transform", `rotate(${angle + 90}, ${point.x}, ${point.y})`)
                    .attr("fill", "white")
                    .attr("stroke", "#333")
                    .attr("stroke-width", 2);
                group.select("path")
                    .attr("stroke-dasharray", "5,5");
            }
            calculateArrowPoints(point, angle, size) {
                return [
                    `${point.x},${point.y}`,
                    `${point.x - size},${point.y + size}`,
                    `${point.x - size},${point.y - size}`
                ];
            }
            calculateDiamondPoints(point, angle, size) {
                return [
                    `${point.x},${point.y}`,
                    `${point.x - size},${point.y + size / 2}`,
                    `${point.x - 2 * size},${point.y}`,
                    `${point.x - size},${point.y - size / 2}`
                ];
            }
            getMidPoint(point1, point2) {
                return {
                    x: (point1.x + point2.x) / 2,
                    y: (point1.y + point2.y) / 2
                };
            }
            getPointAlongLine(point1, point2, ratio) {
                return {
                    x: point1.x + (point2.x - point1.x) * ratio,
                    y: point1.y + (point2.y - point1.y) * ratio
                };
            }
            fitDiagramToView() {
                const diagramBounds = this.svgGroup.node().getBBox();
                const padding = 50;
                const containerWidth = this.svgContainer.node().getBoundingClientRect().width;
                const containerHeight = this.svgContainer.node().getBoundingClientRect().height;
                const scale = Math.min(
                    containerWidth / (diagramBounds.width + padding * 2),
                    containerHeight / (diagramBounds.height + padding * 2)
                );
                const translateX = containerWidth / 2 - (diagramBounds.x + diagramBounds.width / 2) * scale;
                const translateY = containerHeight / 2 - (diagramBounds.y + diagramBounds.height / 2) * scale;
                this.svg.call(
                    this.zoom.transform,
                    d3.zoomIdentity.translate(translateX, translateY).scale(scale)
                );
            }
            downloadSVG() {
                const svgData = new XMLSerializer().serializeToString(this.svg.node());
                const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
                const svgUrl = URL.createObjectURL(svgBlob);
                const downloadLink = document.createElement("a");
                downloadLink.href = svgUrl;
                downloadLink.download = `${this.diagram.name || "uml-diagram"}.svg`;
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            const umlRenderer = new UMLRenderer('diagram-container', 'tooltip');
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            const codeTextarea = document.getElementById('code-input');
            const fileInput = document.getElementById('file-input');
            const parseCodeBtn = document.getElementById('parse-code-btn');
            const parseFileBtn = document.getElementById('parse-file-btn');
            const loadExampleBtn = document.getElementById('load-example-btn');
            const zoomInBtn = document.getElementById('zoom-in-btn');
            const zoomOutBtn = document.getElementById('zoom-out-btn');
            const resetZoomBtn = document.getElementById('reset-zoom-btn');
            const downloadBtn = document.getElementById('download-btn');
            
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    const tabId = tab.getAttribute('data-tab');
                    tabs.forEach(t => t.classList.remove('active'));
                    tabContents.forEach(tc => tc.classList.remove('active'));
                    tab.classList.add('active');
                    document.getElementById(`${tabId}-tab`).classList.add('active');
                });
            });
            
            parseCodeBtn.addEventListener('click', () => {
                const code = codeTextarea.value;
                if (!code) {
                    alert('Please enter some Python code.');
                    return;
                }
                fetch('/api/parse-code', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ code })
                })
                .then(response => response.json())
                .then(data => {
                    umlRenderer.render(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while parsing the code.');
                });
            });
            
            parseFileBtn.addEventListener('click', () => {
                const file = fileInput.files[0];
                if (!file) {
                    alert('Please select a file.');
                    return;
                }
                const formData = new FormData();
                formData.append('file', file);
                fetch('/api/parse-file', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    umlRenderer.render(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while parsing the file.');
                });
            });
            
            loadExampleBtn.addEventListener('click', () => {
                fetch('/api/example-diagram')
                .then(response => response.json())
                .then(data => {
                    umlRenderer.render(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while loading the example.');
                });
            });
            
            zoomInBtn.addEventListener('click', () => {
                umlRenderer.zoomIn();
            });
            
            zoomOutBtn.addEventListener('click', () => {
                umlRenderer.zoomOut();
            });
            
            resetZoomBtn.addEventListener('click', () => {
                umlRenderer.resetZoom();
            });
            
            downloadBtn.addEventListener('click', () => {
                umlRenderer.downloadSVG();
            });
            
            window.addEventListener('resize', () => {
                if (umlRenderer.diagram) {
                    umlRenderer.fitDiagramToView();
                }
            });
            
            function handleDragOver(event) {
                event.preventDefault();
                event.stopPropagation();
                event.dataTransfer.dropEffect = 'copy';
            }
            
            function handleDrop(event) {
                event.preventDefault();
                event.stopPropagation();
                const files = event.dataTransfer.files;
                if (files.length > 0) {
                    const file = files[0];
                    if (file.name.endsWith('.py')) {
                        fileInput.files = files;
                        tabs.forEach(t => t.classList.remove('active'));
                        tabContents.forEach(tc => tc.classList.remove('active'));
                        document.querySelector('[data-tab="file"]').classList.add('active');
                        document.getElementById('file-tab').classList.add('active');
                        parseFileBtn.click();
                    } else {
                        alert('Please drop a Python (.py) file.');
                    }
                }
            }
            
            const dropZone = document.querySelector('.main');
            dropZone.addEventListener('dragover', handleDragOver);
            dropZone.addEventListener('drop', handleDrop);
        });
    </script>
</body>
</html>
'''

def create_example_diagram():
    diagram = UMLDiagram("Example Diagram")
    animal = UMLClass("Animal", abstract=True)
    animal.add_attribute("name", "str", Visibility.PROTECTED)
    animal.add_method("make_sound", "str", [], Visibility.PUBLIC)
    animal.add_method("get_name", "str", [], Visibility.PUBLIC)
    diagram.add_element(animal)
    dog = UMLClass("Dog")
    dog.add_attribute("breed", "str", Visibility.PRIVATE)
    dog.add_method("make_sound", "str", [], Visibility.PUBLIC)
    dog.add_method("get_breed", "str", [], Visibility.PUBLIC)
    diagram.add_element(dog)
    cat = UMLClass("Cat")
    cat.add_attribute("color", "str", Visibility.PRIVATE)
    cat.add_method("make_sound", "str", [], Visibility.PUBLIC)
    cat.add_method("get_color", "str", [], Visibility.PUBLIC)
    diagram.add_element(cat)
    pet_interface = UMLInterface("IPet")
    pet_interface.add_method("play", "None", [], Visibility.PUBLIC)
    diagram.add_element(pet_interface)
    diagram.add_relationship(dog.id, animal.id, RelationType.INHERITANCE)
    diagram.add_relationship(cat.id, animal.id, RelationType.INHERITANCE)
    diagram.add_relationship(dog.id, pet_interface.id, RelationType.IMPLEMENTATION)
    diagram.add_relationship(cat.id, pet_interface.id, RelationType.IMPLEMENTATION)
    diagram.add_relationship(dog.id, cat.id, RelationType.ASSOCIATION, "1", "*", "chases")
    diagram.auto_layout()
    return diagram

def start_server(port=5000, debug=False):
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/api/parse-code', methods=['POST'])
    def parse_code():
        data = request.json
        code = data.get('code', '')
        parser = UMLParser()
        parser.parse_text(code)
        diagram = parser.get_diagram()
        return jsonify(diagram.to_dict())
    
    @app.route('/api/parse-file', methods=['POST'])
    def parse_file():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp:
            file.save(temp.name)
            parser = UMLParser()
            parser.parse_file(temp.name)
            os.unlink(temp.name)
        diagram = parser.get_diagram()
        return jsonify(diagram.to_dict())
    
    @app.route('/api/example-diagram', methods=['GET'])
    def example_diagram():
        return jsonify(create_example_diagram().to_dict())
    
    app.run(debug=debug, port=port)

if __name__ == '__main__':
    start_server(debug=True)
