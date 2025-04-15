from typing import Dict, Any, Optional
import os
import yaml
from loguru import logger

from .prompt_template import PromptTemplate

class PromptManager:
    """提示词管理器，负责加载、管理和提供提示词模板"""
    
    def __init__(self, templates_dir: str = None):
        """
        初始化提示词管理器
        
        Args:
            templates_dir: 提示词模板目录
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_data: Dict[str, Dict[str, Any]] = {}  # 存储模板的原始配置数据
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )
        # 确保模板目录存在
        os.makedirs(self.templates_dir, exist_ok=True)
        # 加载所有模板
        self.load_all_templates()
    
    def load_all_templates(self):
        """加载所有提示词模板文件"""
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                file_path = os.path.join(self.templates_dir, filename)
                self.load_templates_from_file(file_path)
        
        logger.info(f"已加载 {len(self.templates)} 个提示词模板")
    
    def load_templates_from_file(self, file_path: str):
        """从文件加载提示词模板"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates_data = yaml.safe_load(f)
            
            for template_name, template_data in templates_data.items():
                # 保存原始配置数据
                self.templates_data[template_name] = template_data.copy()
                
                # 处理schema中的required字段
                if "schema" in template_data:
                    schema = template_data["schema"]
                    # 处理输入和输出schema
                    if "input" in schema:
                        self._process_schema_section(schema["input"])
                    if "output" in schema:
                        self._process_schema_section(schema["output"])
                    # 兼容处理老格式（没有input/output区分的schema）
                    elif "properties" in schema:
                        self._process_schema_section(schema)
                
                # 初始化模板内容为None
                template_content = None
                
                # 优先从template_path加载内容
                if 'template_path' in template_data:
                    template_path = template_data['template_path']
                    # 如果路径是相对路径，则相对于YAML文件目录
                    if not os.path.isabs(template_path):
                        yaml_dir = os.path.dirname(os.path.abspath(file_path))
                        template_path = os.path.join(yaml_dir, template_path)
                    
                    if os.path.exists(template_path):
                        try:
                            with open(template_path, 'r', encoding='utf-8') as md_file:
                                template_content = md_file.read()
                            logger.info(f"从 {template_path} 加载了提示词模板内容")
                        except Exception as md_error:
                            logger.error(f"读取提示词Markdown文件 {template_path} 时出错: {str(md_error)}")
                    else:
                        logger.warning(f"提示词Markdown文件 {template_path} 不存在，尝试使用YAML中的模板内容")
                
                # 如果没有从template_path加载到内容，则使用template字段
                if template_content is None:
                    if 'template' in template_data:
                        template_content = template_data['template']
                    else:
                        logger.warning(f"模板 {template_name} 既没有有效的template_path也没有template字段，跳过加载")
                        continue
                
                template = PromptTemplate(
                    template=template_content,
                    template_name=template_name,
                    template_type=template_data.get('type', 'generic'),
                    role=template_data.get('role', 'user'),
                    max_tokens=template_data.get('max_tokens'),
                    min_tokens=template_data.get('min_tokens'),
                    schema=template_data.get('schema')
                )
                self.register_template(template)
                
            logger.info(f"从 {file_path} 加载了 {len(templates_data)} 个提示词模板")
        except Exception as e:
            logger.error(f"加载提示词模板文件 {file_path} 时出错: {str(e)}")
    
    def _process_schema_section(self, schema: Dict[str, Any]):
        """处理schema的required字段，将其移至各属性中"""
        required_fields = schema.pop("required", [])
        
        # 将required标记移至各属性中
        if "properties" in schema:
            for field_name, field_props in schema["properties"].items():
                field_props["required"] = field_name in required_fields
    
    def register_template(self, template: PromptTemplate):
        """注册提示词模板"""
        if template.template_name:
            self.templates[template.template_name] = template
        else:
            logger.warning("尝试注册没有名称的提示词模板")
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """通过名称获取提示词模板"""
        return self.templates.get(template_name)
    