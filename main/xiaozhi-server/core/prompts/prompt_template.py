from typing import Dict, Any, Type
import json
from loguru import logger

from pydantic import BaseModel, Field, create_model


def _get_field_type(type_str: str) -> Type:
    """从Schema类型字符串获取Python类型"""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }
    return type_map.get(type_str, str)

# 用以验证大模型返回的结果是否符合要求
def _create_model_from_schema(schema: Dict[str, Any], output: bool) -> Type[BaseModel] | None | Any:
    """从JSON Schema创建Pydantic模型"""
    # 如果schema包含input和output区分，则只使用output部分创建模型
    if schema is None:
        return None
    if "output" in schema and output:
        properties = schema.get("output", {}).get("properties", {})
    elif "input" in schema and not output:
        properties = schema.get("input", {}).get("properties", {})
    else:
        properties = schema.get("properties", {})
    if properties is None:
        return None
    
    fields = {}

    for field_name, field_schema in properties.items():
        field_type = _get_field_type(field_schema.get("type", "string"))
        is_required = field_schema.get("required", False)

        # 使用Pydantic的Field创建字段，包含描述和必填标记
        field_default = ... if is_required else None
        field_description = field_schema.get("description", "")

        fields[field_name] = (
            field_type,
            Field(default=field_default, description=field_description)
        )

    model_name = schema.get("title", "DynamicModel")        
    return create_model(model_name, **fields)


class PromptTemplate:
    """提示词模板类，支持变量替换和格式化"""
    
    def __init__(
        self, 
        template: str, 
        template_name: str = None,
        template_type: str = "generic",
        role: str = "user",
        max_tokens: int = None,
        min_tokens: int = None,
        schema: Dict[str, Any] = None
    ):
        """
        初始化提示词模板
        
        Args:
            template: 提示词模板字符串
            template_name: 模板名称
            template_type: 模板类型
            role: 角色，可以是"user"或"assistant"
            max_tokens: 最大token数
            min_tokens: 最小token数
            schema: JSON Schema定义，可以包含input和output部分
        """
        self.template = template
        self.template_name = template_name
        self.template_type = template_type
        self.role = role
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.schema = schema
        self.schema_formatted = self.format_schema()
        self.max_min_token_formatted = self.get_max_min_token_formatted()
        if self.schema_formatted is not None:
            self.formatted_prompt = self.template + self.max_min_token_formatted + self.schema_formatted
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            self.output_schema_model = schema
        else:
            self.output_schema_model = _create_model_from_schema(self.schema, True)
            self.input_schema_model = _create_model_from_schema(self.schema, False)

    def format_schema(self) -> str:
        """生成JSON Schema的格式化指令"""
        if not self.schema:
            return ""
        
        # 创建用于展示的schema
        display_schema = self._create_display_schema()
        schema_str = json.dumps(display_schema, ensure_ascii=False, indent=2)
        
        return f"""您的响应必须是一个符合以下JSON Schema的JSON对象:
```json
        {schema_str}
```
请确保您的响应可以被JSON解析，并且匹配上述的Schema结构。
        """
    
    def _create_display_schema(self) -> Dict[str, Any]:
        """创建用于展示的schema，标记每个字段是否必需"""
        # 初始化display_schema
        display_schema = {}
        
        # 确定要处理的schema部分
        target_schema = self.schema

        # 如果schema包含output部分，则使用output部分
        if isinstance(target_schema, dict) and "output" in target_schema:
            target_schema = target_schema.get("output", {})
            
        # 处理属性
        if isinstance(target_schema, dict):
            for field_name, field_schema in target_schema.get("properties", {}).items():
                field_info = field_schema.copy()
                is_required = field_info.pop("required", False)
                if is_required:
                    field_info["required"] = is_required
                
                display_schema[field_name] = field_info
        
        return display_schema

    def get_max_min_token_formatted(self) -> str:
        """获取最大最小token的格式化字符串"""
        formatted_str = ""
        if self.max_tokens is not None or self.min_tokens is not None:
            formatted_str += "注意，你的本次响应必须符合以下token数量的限制："
        if self.max_tokens is not None:
            formatted_str += f"最大token: {self.max_tokens}\n"
        if self.min_tokens is not None:
            if formatted_str:
                formatted_str += ", "
            formatted_str += f"最小token: {self.min_tokens}\n"
        return formatted_str

    def get_input_prompt(self, val: Dict[str, Any]) -> str:
        """根据输入schema验证并转换输入值为JSON字符串
        
        Args:
            val: 输入的字典值
            
        Returns:
            str: JSON格式的字符串
            
        Raises:
            ValueError: 当必需字段缺失或值类型不匹配时抛出
        """
        if self.input_schema_model is None:
            return json.dumps(val, ensure_ascii=False)
            
        try:
            # 使用Pydantic模型验证输入
            validated_data = self.input_schema_model(**val)
            # 转换为JSON字符串
            return json.dumps(validated_data.model_dump(), ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"输入值验证失败: {str(e)}")

    def parse(self, text: str) -> Any:
        """从文本中解析JSON对象"""
        try:
            # 尝试提取JSON部分
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if 0 <= start_idx < end_idx:
                json_str = text[start_idx:end_idx+1]
                data = json.loads(json_str)
                if self.output_schema_model is not None:
                    # 使用Pydantic模型验证
                    return self.output_schema_model(**data)
            else:
                raise ValueError("未找到JSON对象")
        except Exception as e:
            logger.error(f"解析JSON输出时出错: {str(e)}")
            raise
