from typing import Dict, Any, Type
import json
from config.logger import setup_logging

from pydantic import BaseModel, Field, create_model

TAG = __name__
logger = setup_logging()


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
        # display_schema = self._create_display_schema()
        schema_str = json.dumps(self.schema, ensure_ascii=False, indent=2)
        
        return f"""你的响应必须是一个符合以下JSON Schema的JSON对象，注意在required列表中的字段是必须填有意义的内容，不能填写空值或未知等内容:
```json
        {schema_str}
```
请确保你的响应可以被JSON解析，并且匹配上述的Schema结构。
        """

    def get_max_min_token_formatted(self) -> str:
        """获取最大最小token的格式化字符串"""
        formatted_str = ""
        if self.max_tokens is not None or self.min_tokens is not None:
            formatted_str += "注意，你的本次响应必须符合以下token数量的限制："
        if self.max_tokens is not None:
            formatted_str += f"最大token: {self.max_tokens}\n"
            formatted_str += "但是请注意很重要的一点，如果要求回答为json格式，则可以忽略max_tokens的限制，而必须首先确保json schema结构完整，能够解析json\n"
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
                    try:
                        # 尝试使用Pydantic模型验证
                        return self.output_schema_model(**data)
                    except Exception as e:
                        logger.bind(tag=TAG).warning(f"Pydantic模型验证失败，将使用自定义对象访问模式: {str(e)}")
                        return AttrDict(data)
                else:
                    # 如果没有模型，则返回带属性访问的字典
                    return AttrDict(data)
            else:
                raise ValueError("未找到JSON对象")
        except Exception as e:
            logger.bind(tag=TAG).error(f"解析JSON输出时出错: {str(e)}")
            raise

# 添加一个辅助类，允许同时以属性和字典方式访问数据
class AttrDict(dict):
    """允许通过属性访问的字典类"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
        # 递归处理嵌套的字典和列表
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            elif isinstance(value, list):
                self[key] = [
                    AttrDict(item) if isinstance(item, dict) else item
                    for item in value
                ]
    
    # 提供属性访问的兜底方法
    def __getattr__(self, name):
        # 尝试从字典获取，如果不存在则返回None而不是抛出异常
        return self.get(name)
