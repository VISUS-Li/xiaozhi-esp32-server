from typing import Dict, Any, Type, List, Optional
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

def _create_model_from_schema(
    schema: Optional[Dict[str, Any]], 
    output: bool = True
) -> Optional[Type[BaseModel]]:
    """从JSON Schema创建Pydantic模型，支持递归解析嵌套结构
    
    Args:
        schema: JSON Schema定义
        output: 是否处理output部分，True处理output，False处理input
        
    Returns:
        创建的Pydantic模型类，如果无法创建则返回None
    """
    # 检查schema是否为None
    if schema is None:
        return None
    
    # 根据output参数确定处理的schema部分
    schema_part = None
    if "output" in schema and output:
        schema_part = schema.get("output", {})
    elif "input" in schema and not output:
        schema_part = schema.get("input", {})
    else:
        schema_part = schema
    
    if schema_part is None or not isinstance(schema_part, dict):
        return None
    
    # 获取properties和required字段
    properties = schema_part.get("properties", {})
    if not properties:
        return None
    
    required_fields = schema_part.get("required", [])
    

    model_name = schema_part.get("title", schema.get("title", f"DynamicModel_{id(schema)}"))
    
    # 处理所有字段
    fields = {}
    for field_name, field_schema in properties.items():
        # 获取默认值和描述
        is_required = field_name in required_fields
        # 非必需字段明确设置为可接受None值，对于字符串类型的字段也允许None
        if is_required:
            field_default = ...
        else:
            # 对于非必需字段，明确设置默认值为None并允许接受None值
            field_default = None
        field_description = field_schema.get("description", "")
        
        # 获取字段类型并处理特殊类型
        field_type = _process_field_type(field_schema)
        
        # 添加到字段字典，对于非必需字段设置allow_none=True
        if is_required:
            fields[field_name] = (
                field_type,
                Field(default=field_default, description=field_description)
            )
        else:
            # 允许非必需字段为None值
            fields[field_name] = (
                Optional[field_type],
                Field(default=field_default, description=field_description)
            )
    
    # 创建并返回模型
    return create_model(model_name, **fields)

def _process_field_type(field_schema: Dict[str, Any]) -> Type:
    """处理单个字段的类型，支持嵌套对象和数组
    
    Args:
        field_schema: 字段的schema定义
        
    Returns:
        字段的Python类型
    """
    field_type = _get_field_type(field_schema.get("type", "string"))
    
    # 处理嵌套对象
    if field_type == dict and "properties" in field_schema:
        # 使用唯一名称避免命名冲突
        # 递归创建嵌套模型
        nested_model = _create_model_from_schema(field_schema, True)
        field_type = nested_model if nested_model else dict
    
    # 处理数组类型
    elif field_type == list and "items" in field_schema:
        items_schema = field_schema.get("items", {})
        # 处理数组元素
        item_type = _get_field_type(items_schema.get("type", "string"))
        if item_type == dict and "properties" in items_schema:
            # 为数组元素创建模型
            item_model = _create_model_from_schema(items_schema, True)
            field_type = List[item_model] if item_model else List[dict]
        else:
            # 处理基本类型的数组
            field_type = List[item_type]
    
    return field_type


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
        
        # 只提取output部分的schema
        output_schema = {}
        if "output" in self.schema:
            # 如果schema中有output字段，就只使用output部分
            output_schema = self.schema.get("output", {})
        else:
            # 如果没有明确的output字段，则假设整个schema就是输出结构
            output_schema = self.schema
        
        schema_str = json.dumps(output_schema, ensure_ascii=False, indent=2)
        
        return f"""你的响应必须是一个符合以下JSON Schema的JSON对象，注意在required列表中的字段是必须填有意义的内容，不能填写空值或未知等内容:
```json
        {schema_str}
```
请确保你的响应可以被JSON解析，并且匹配上述的Schema结构。不管什么情况下，都必须保证返回的是一个完整的JSON结构。
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

    def parse(self, text: str, validate_required: bool = False) -> Any:
        """从文本中解析JSON对象
        
        Args:
            text: 包含JSON数据的文本或纯JSON字符串
            validate_required: 是否验证required字段，True则验证失败时抛出异常
            
        Returns:
            解析后的对象，如果有schema模型则返回模型实例，否则返回AttrDict
        
        Raises:
            ValueError: 当JSON解析失败时或当validate_required=True且必填字段缺失时
        """
        try:
            # 首先尝试直接解析，处理纯JSON字符串的情况
            try:
                data = json.loads(text)
                logger.bind(tag=TAG).debug("成功直接解析JSON字符串")
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试从文本中提取JSON对象
                logger.bind(tag=TAG).debug("直接解析失败，尝试从文本中提取JSON")
                
                # 尝试寻找最外层的花括号对
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                
                if 0 <= start_idx < end_idx:
                    json_str = text[start_idx:end_idx+1]
                    try:
                        data = json.loads(json_str)
                        logger.bind(tag=TAG).debug("成功通过简单提取解析JSON")
                    except json.JSONDecodeError:
                        # 如果简单提取失败，使用正则表达式进行更复杂的匹配
                        logger.bind(tag=TAG).debug("简单提取解析失败，尝试正则匹配")
                        import re
                        # 匹配可能包含嵌套结构的JSON对象
                        json_pattern = re.compile(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}')
                        matches = list(json_pattern.finditer(text))
                        
                        if not matches:
                            raise ValueError("未找到有效的JSON对象")
                        
                        # 优先选择最长匹配，通常是最完整的JSON
                        longest_match = max(matches, key=lambda m: len(m.group(0)))
                        json_str = longest_match.group(0)
                        try:
                            data = json.loads(json_str)
                            logger.bind(tag=TAG).debug("成功通过正则匹配解析JSON")
                        except json.JSONDecodeError:
                            # 尝试其他可能的匹配
                            for match in sorted(matches, key=lambda m: len(m.group(0)), reverse=True):
                                if match == longest_match:
                                    continue
                                try:
                                    json_str = match.group(0)
                                    data = json.loads(json_str)
                                    logger.bind(tag=TAG).debug("成功通过备选正则匹配解析JSON")
                                    break
                                except json.JSONDecodeError:
                                    continue
                            else:
                                raise ValueError("所有可能的JSON匹配都解析失败")
                else:
                    raise ValueError("文本中未找到JSON对象的起始和结束标记")
            
            # 使用schema模型验证解析结果
            if self.output_schema_model is not None:
                try:
                    # 尝试使用Pydantic模型验证
                    model_instance = self.output_schema_model(**data)
                    logger.bind(tag=TAG).debug(f"成功解析并验证JSON对象为 {self.output_schema_model.__name__} 类型")
                    return model_instance
                except Exception as e:
                    logger.bind(tag=TAG).warning(f"Pydantic模型验证失败: {str(e)}")
                    # 如果需要验证required字段，则在验证失败时抛出异常
                    if validate_required:
                        raise ValueError(f"JSON数据验证失败，缺少必填字段或字段类型不匹配: {str(e)}")
                    # 否则退回到兼容模式，返回AttrDict
                    return AttrDict(data)
            else:
                # 没有模型，使用AttrDict提供属性访问
                return AttrDict(data)
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"解析JSON输出时出错: {str(e)}")
            raise ValueError(f"JSON解析失败: {str(e)}")


# 重构AttrDict类，改进嵌套字典和列表的处理
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
                self[key] = self._process_list(value)
    
    def _process_list(self, lst):
        """递归处理列表中的字典和嵌套列表"""
        result = []
        for item in lst:
            if isinstance(item, dict):
                result.append(AttrDict(item))
            elif isinstance(item, list):
                result.append(self._process_list(item))
            else:
                result.append(item)
        return result
    
    # 提供属性访问的兜底方法
    def __getattr__(self, name):
        # 尝试从字典获取，如果不存在则返回None而不是抛出异常
        return self.get(name)
    
    # 增加字符串表示方法，方便调试
    def __str__(self):
        return f"AttrDict({super().__str__()})"
    
    def __repr__(self):
        return self.__str__()
