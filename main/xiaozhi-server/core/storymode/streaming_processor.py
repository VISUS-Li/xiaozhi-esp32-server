from loguru import logger
import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from core.utils.util import get_string_no_punctuation_or_emoji
from core.prompts import PromptManager
from config.module_config import get_module_config
import random

TAG = __name__

class StreamingTextProcessor:
    """流式文本处理器，将流式返回的大语言模型文本按照标点符号分段并进行TTS处理"""
    
    def __init__(self, connection_handler):
        """
        初始化流式文本处理器
        
        Args:
            connection_handler: 连接处理器实例，包含speak_and_play等方法
        """
        self.conn = connection_handler
        self.punctuations = ("。", "？", "！", "；", "：", ",", "、", ":", "：")  # 可用于分割的标点符号
        self.prompt_manager = PromptManager()  # 初始化提示词管理器，用于解析JSON
        
        # 获取故事模式配置
        self.story_config = {}
        try:
            if hasattr(self.conn, 'config'):
                self.story_config = get_module_config(self.conn.config, "story-module")
                logger.bind(tag=TAG).info(f"已加载故事模式配置，包含 {len(self.story_config.get('voices', {}).get('narrators', []))} 个旁白音色和 {len(self.story_config.get('voices', {}).get('characters', []))} 个角色音色")
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载故事模式配置失败: {e}")
        
        # 检查是否为JSON响应模式
        self.is_json_mode = False
        self.json_buffer = []
        self.current_stage = ""
        # 角色音色映射，用于保持角色音色一致性
        self.role_voice_mapping = {}
    
    async def process_streaming_text(self, llm_responses, dialogue_callback=None, abort_check=None):
        """
        处理流式文本并转换为语音
        
        Args:
            llm_responses: 大模型流式返回的文本生成器
            dialogue_callback: 处理完成后的对话回调函数，接收完整的响应文本作为参数
            abort_check: 检查是否中止处理的函数，返回True表示中止处理
            
        Returns:
            完整的响应文本
        """
        response_message = []    # 存储完整响应
        processed_chars = 0      # 已处理的字符数
        text_index = 0           # 文本索引
        json_processed = False   # 是否已处理JSON
        
        # 如果没有提供中止检查函数，使用默认实现
        if abort_check is None:
            abort_check = lambda: self.conn.client_abort if hasattr(self.conn, 'client_abort') else False
        
        # 检查当前阶段，决定是否启用JSON处理模式
        if hasattr(self.conn, 'story_session') and hasattr(self.conn.story_session, 'stage'):
            self.current_stage = self.conn.story_session.stage
            # 启用JSON模式的条件: story_continuation阶段或outline_generation阶段
            self.is_json_mode = (self.current_stage == "story_continuation" or 
                                self.current_stage == "outline_generation")
            logger.bind(tag=TAG).info(f"当前阶段: {self.current_stage}, 是否启用JSON模式: {self.is_json_mode}")
        
        try:
            # 开始处理流式响应
            for content in llm_responses:
                # 将新内容添加到响应中
                response_message.append(content)
                
                # 检查是否需要中止处理
                if abort_check():
                    logger.bind(tag=TAG).info("流式处理被中止")
                    break
                
                # 合并当前全部文本
                full_text = "".join(response_message)
                current_text = full_text[processed_chars:]  # 从未处理的位置开始
                
                # 在流式处理期间，JSON模式下不处理JSON（等待完整响应）
                # 只处理常规文本
                if not self.is_json_mode:
                    # 常规文本模式: 查找最后一个有效标点
                    last_punct_pos = self._find_last_punctuation(current_text)
                    
                    # 找到分割点则处理
                    if last_punct_pos != -1:
                        segment_text_raw = current_text[:last_punct_pos + 1]
                        segment_text = get_string_no_punctuation_or_emoji(segment_text_raw)
                        if segment_text:
                            # 处理该文本段
                            text_index += 1
                            await self._process_text_segment(segment_text, text_index)
                            processed_chars += len(segment_text_raw)  # 更新已处理字符位置
            
            # 处理最后剩余的文本
            full_text = "".join(response_message)
            remaining_text = full_text[processed_chars:]
            
            if self.is_json_mode:
                # JSON模式下，只在完整响应后尝试处理JSON
                logger.bind(tag=TAG).debug("尝试处理完整的JSON响应")
                json_processed = await self._process_complete_json_response(full_text, text_index)
                
                # 如果JSON处理失败，尝试按常规文本处理
                if not json_processed and remaining_text:
                    logger.bind(tag=TAG).info("未找到有效JSON，按常规文本处理")
                    segment_text = get_string_no_punctuation_or_emoji(remaining_text)
                    if segment_text:
                        text_index += 1
                        await self._process_text_segment(segment_text, text_index)
            elif remaining_text:
                # 常规模式下处理剩余文本
                segment_text = get_string_no_punctuation_or_emoji(remaining_text)
                if segment_text:
                    text_index += 1
                    await self._process_text_segment(segment_text, text_index)
            
            # 完整响应文本
            complete_response = "".join(response_message)
            
            # 如果有回调函数，调用它
            if dialogue_callback:
                dialogue_callback(complete_response)
                
            return complete_response
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"流式文本处理出错: {e}")
            return "".join(response_message)  # 返回已处理的内容
    
    def _find_last_punctuation(self, text):
        """查找文本中最后一个有效标点的位置"""
        last_punct_pos = -1
        for punct in self.punctuations:
            pos = text.rfind(punct)
            if pos > last_punct_pos:
                last_punct_pos = pos
        return last_punct_pos
    
    async def _process_text_segment(self, text, text_index):
        """处理单个文本段"""
        if hasattr(self.conn, 'recode_first_last_text'):
            self.conn.recode_first_last_text(text, text_index)
        
        # 使用连接处理器的executor提交speak_and_play任务
        future = self.conn.executor.submit(
            self.conn.speak_and_play, text, text_index
        )
        
        # 将任务加入TTS队列
        self.conn.tts_queue.put(future)
    
    async def _process_complete_json_response(self, text, text_index):
        """处理完整的JSON响应"""
        try:
            # 尝试从文本中提取JSON对象
            matches = re.finditer(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', text)
            json_objects = [match.group(0) for match in matches]
            
            if not json_objects:
                logger.bind(tag=TAG).info("未在响应中找到JSON对象")
                return False
            
            # 找到最后一个（最完整的）JSON对象
            json_text = json_objects[-1]
            
            try:
                # 尝试解析为JSON
                json_data = json.loads(json_text)
                
                # 检查JSON是否有效且完整
                is_complete = self._is_json_complete(json_data)
                if not is_complete:
                    logger.bind(tag=TAG).warning("找到的JSON数据不完整")
                    return False
                
                # 如果JSON有效且完整，处理它
                await self._process_json_content(json_text, text_index)
                return True
                
            except json.JSONDecodeError as e:
                logger.bind(tag=TAG).error(f"JSON解析失败: {e}")
                return False
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理完整JSON响应时出错: {e}")
            return False
    
    def _is_json_complete(self, json_data):
        """检查JSON数据是否完整，根据当前阶段从配置中获取必需字段"""
        try:
            # 从prompt_manager获取当前阶段的模板
            template_name = self.current_stage
            template = self.prompt_manager.get_template(template_name)
            
            if not template or not template.schema:
                logger.bind(tag=TAG).warning(f"未找到阶段 {template_name} 的模板或schema")
                return False
                
            # 获取模板中定义的必需字段
            schema = template.schema
            required_fields = []
            
            # 从output schema中提取必需字段
            if "output" in schema and "required" in schema["output"]:
                required_fields = schema["output"]["required"]
            elif "required" in schema:
                required_fields = schema["required"]
                
            if not required_fields:
                logger.bind(tag=TAG).warning(f"未在模板 {template_name} 中找到必需字段")
                return False
                
            # 检查所有必需字段是否存在
            for field in required_fields:
                if field not in json_data:
                    logger.bind(tag=TAG).debug(f"缺少必需字段: {field}")
                    return False
                    
            # 特殊检查: 如果有数组类型字段，确保其非空且类型正确
            if "output" in schema and "properties" in schema["output"]:
                properties = schema["output"]["properties"]
                for field, prop in properties.items():
                    if field in json_data and prop.get("type") == "array":
                        if not isinstance(json_data[field], list):
                            logger.bind(tag=TAG).debug(f"字段 {field} 应为数组类型")
                            return False
                        if required_fields and field in required_fields and len(json_data[field]) == 0:
                            logger.bind(tag=TAG).debug(f"必需数组 {field} 为空")
                            return False
                            
                        # 如果数组有items定义，检查每个元素
                        if "items" in prop and prop["items"].get("type") == "object" and json_data[field]:
                            item_required = prop["items"].get("required", [])
                            for item in json_data[field]:
                                if not isinstance(item, dict):
                                    return False
                                for req in item_required:
                                    if req not in item:
                                        return False
            
            # 通过所有检查
            return True
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"检查JSON完整性时出错: {e}")
            return False
    
    async def _process_json_content(self, json_text, text_index):
        """处理JSON内容，根据不同的阶段调用不同的处理方法"""
        if self.current_stage == "story_continuation":
            await self._process_story_continuation(json_text, text_index)
        elif self.current_stage == "outline_generation":
            await self._process_story_outline(json_text, text_index)
        else:
            # 对于其他JSON内容，暂时按普通文本处理
            segment_text = get_string_no_punctuation_or_emoji(json_text)
            if segment_text:
                await self._process_text_segment(segment_text, text_index)
    
    async def _process_story_continuation(self, json_text, text_index):
        """处理故事续写阶段的JSON内容"""
        try:
            # 使用提示词模板中的解析方法解析JSON
            template = self.prompt_manager.get_template("story_continuation")
            if not template:
                logger.bind(tag=TAG).error("未找到story_continuation模板，无法解析JSON")
                return
            
            story_data = template.parse(json_text)
            
            # 检查是否有story_seg字段
            story_seg = getattr(story_data, "story_seg", None) if hasattr(story_data, "story_seg") else story_data.get("story_seg")
            if not story_seg:
                logger.bind(tag=TAG).error(f"解析的故事数据缺少story_seg字段: {story_data}")
                return
            
            # 检查故事是否已结束
            ended = getattr(story_data, "ended", False) if hasattr(story_data, "ended") else story_data.get("ended", False)
            if ended:
                logger.bind(tag=TAG).info("故事已标记为结束状态")
            
            # 处理故事片段
            for i, segment in enumerate(story_seg):
                text_index += 1
                
                # 支持两种访问方式：属性访问或字典访问
                is_role = getattr(segment, "is_role", None) if hasattr(segment, "is_role") else segment.get("is_role")
                
                if is_role:
                    # 处理角色对话
                    await self._process_role_dialogue(segment, text_index)
                else:
                    # 处理旁白
                    await self._process_narrator(segment, text_index)
        
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理故事续写JSON时出错: {e}")
            # 如果解析失败，尝试作为普通文本处理
            segment_text = get_string_no_punctuation_or_emoji(json_text)
            if segment_text:
                await self._process_text_segment(segment_text, text_index)
    
    async def _process_role_dialogue(self, segment, text_index):
        """处理角色对话"""
        try:
            # 支持两种访问方式
            role_name = getattr(segment, "role_name", None) if hasattr(segment, "role_name") else segment.get("role_name", "")
            content = getattr(segment, "content", None) if hasattr(segment, "content") else segment.get("content", "")
            mood = getattr(segment, "mood", "") if hasattr(segment, "mood") else segment.get("mood", "")
            mood_location = getattr(segment, "mood_location", 2) if hasattr(segment, "mood_location") else segment.get("mood_location", 2)
            
            # 获取性别，用于选择合适的语音
            role_gender = None
            if hasattr(segment, "role_gender"):
                role_gender = segment.role_gender
            elif "role_gender" in segment:
                role_gender = segment["role_gender"]
            
            # 根据mood_location组装完整句子
            if mood_location == 1:
                # 情绪描述在前
                full_text = f"{role_name}{mood}说道：{content}"
            else:
                # 情绪描述在后
                full_text = f"{role_name}说道：{content}{mood}"
            
            # 获取角色的语音配置
            tts_config = self._get_role_voice(role_name, role_gender)
            
            # 记录文本位置
            if hasattr(self.conn, 'recode_first_last_text'):
                self.conn.recode_first_last_text(full_text, text_index)
            
            # 使用特定音色生成语音
            if tts_config:
                # 自定义的TTS处理
                future = self.conn.executor.submit(
                    self._custom_speak_and_play, full_text, text_index, tts_config
                )
            else:
                # 使用默认TTS处理
                future = self.conn.executor.submit(
                    self.conn.speak_and_play, full_text, text_index
                )
            
            # 加入TTS队列
            self.conn.tts_queue.put(future)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理角色对话时出错: {e}")
            logger.bind(tag=TAG).error(f"错误详情: {str(e)}")
            # 记录segment内容以便调试
            try:
                logger.bind(tag=TAG).debug(f"段落内容: {segment}")
            except:
                pass
    
    async def _process_narrator(self, segment, text_index):
        """处理旁白"""
        try:
            # 支持两种访问方式
            content = getattr(segment, "content", None) if hasattr(segment, "content") else segment.get("content", "")
            
            # 记录文本位置
            if hasattr(self.conn, 'recode_first_last_text'):
                self.conn.recode_first_last_text(content, text_index)
            
            # 获取旁白的语音配置
            tts_config = self._get_narrator_voice()
            
            # 使用特定音色生成语音
            if tts_config:
                # 自定义的TTS处理
                future = self.conn.executor.submit(
                    self._custom_speak_and_play, content, text_index, tts_config
                )
            else:
                # 使用默认TTS处理
                future = self.conn.executor.submit(
                    self.conn.speak_and_play, content, text_index
                )
            
            # 加入TTS队列
            self.conn.tts_queue.put(future)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理旁白时出错: {e}")
            logger.bind(tag=TAG).error(f"错误详情: {str(e)}")
            # 记录segment内容以便调试
            try:
                logger.bind(tag=TAG).debug(f"段落内容: {segment}")
            except:
                pass
    
    def _get_role_voice(self, role_name, gender=None):
        """获取角色的语音配置，保持一致性"""
        # 检查是否已有该角色的音色映射
        if role_name in self.role_voice_mapping:
            return self.role_voice_mapping[role_name]
        
        # 从配置中获取可用的角色音色
        characters = self.story_config.get("voices", {}).get("characters", [])
        if not characters:
            return None
        
        # 根据性别筛选音色
        if gender:
            suitable_voices = [char for char in characters 
                            if char.get("gender", "") == gender]
        else:
            suitable_voices = characters
        
        # 如果没有合适的音色，使用所有角色音色
        if not suitable_voices:
            suitable_voices = characters
        
        # 随机选择一个音色
        if suitable_voices:
            selected_voice = random.choice(suitable_voices)
            # 确保选择的音色配置至少包含type属性
            tts_config = selected_voice.get("tts", {})
            if tts_config and "type" in tts_config:
                # 存储映射关系，保持一致性
                self.role_voice_mapping[role_name] = tts_config
                return tts_config
        
        return None
    
    def _get_narrator_voice(self):
        """获取旁白的语音配置"""
        # 从配置中获取可用的旁白音色
        narrators = self.story_config.get("voices", {}).get("narrators", [])
        if narrators:
            # 随机选择一个旁白音色
            selected_voice = random.choice(narrators)
            # 确保选择的音色配置至少包含type属性
            tts_config = selected_voice.get("tts", {})
            if tts_config and "type" in tts_config:
                return tts_config
        return None
    
    def _custom_speak_and_play(self, text, text_index, tts_config):
        """使用指定的TTS配置生成语音并播放"""
        try:
            # 导入必要的TTS模块
            from core.utils import tts
            
            # 创建TTS实例
            tts_type = tts_config.get("type")
            if not tts_type:
                # 如果没有指定TTS类型，使用默认方法
                return self.conn.speak_and_play(text, text_index)
            
            # 合并TTS配置
            # 首先获取全局配置中对应TTS类型的配置
            global_tts_config = {}
            if hasattr(self.conn, 'config') and 'TTS' in self.conn.config:
                for tts_name, tts_cfg in self.conn.config['TTS'].items():
                    if tts_cfg.get('type') == tts_type:
                        global_tts_config = tts_cfg.copy()  # 创建一个副本，避免修改原始配置
                        break
            
            # 创建一个新字典，先复制全局配置，然后再覆盖故事模式的配置
            # 这样故事模式的配置优先级更高
            merged_config = {}
            
            # 1. 首先添加全局配置中的内容
            if global_tts_config:
                merged_config.update(global_tts_config)
            
            # 2. 然后用故事模式的配置覆盖全局配置
            merged_config.update(tts_config)
            
            # 3. 确保必要参数存在
            # 确保输出目录被包含
            if 'output_dir' not in merged_config and global_tts_config and 'output_dir' in global_tts_config:
                merged_config['output_dir'] = global_tts_config['output_dir']
            elif 'output_dir' not in merged_config and hasattr(self.conn, 'config') and 'TTS' in self.conn.config:
                # 尝试从任何TTS配置中获取输出目录
                for tts_name, tts_cfg in self.conn.config['TTS'].items():
                    if 'output_dir' in tts_cfg:
                        merged_config['output_dir'] = tts_cfg['output_dir']
                        break
            
            # 确保最终的配置中有type字段
            if 'type' not in merged_config:
                merged_config['type'] = tts_type
            
            logger.bind(tag=TAG).debug(f"使用TTS类型: {tts_type}, 参数: {merged_config}")
            
            # 创建TTS实例
            tts_instance = tts.create_instance(
                tts_type, 
                merged_config,
                self.conn.config.get("delete_audio_file", True)
            )
            
            if not tts_instance:
                # 如果创建失败，使用默认方法
                return self.conn.speak_and_play(text, text_index)
            
            # 生成语音文件
            audio_file = tts_instance.to_tts(text)
            if not audio_file:
                # 如果生成失败，使用默认方法
                return self.conn.speak_and_play(text, text_index)
            
            # 获取Opus编码数据
            opus_data, duration = tts_instance.audio_to_opus_data(audio_file)
            
            # 播放音频 - 使用连接处理器的音频播放队列
            if hasattr(self.conn, 'audio_play_queue'):
                # 将音频数据加入播放队列
                self.conn.audio_play_queue.put((opus_data, text, text_index))
            
            return audio_file, text, text_index
        except Exception as e:
            logger.bind(tag=TAG).error(f"自定义语音生成出错: {e}")
            # 如果失败，使用默认方法
            return self.conn.speak_and_play(text, text_index)
    
    async def _process_story_outline(self, json_text, text_index):
        """处理故事大纲阶段的JSON内容"""
        try:
            # 使用提示词模板中的解析方法解析JSON
            template = self.prompt_manager.get_template("outline_generation")
            if not template:
                logger.bind(tag=TAG).error("未找到outline_generation模板，无法解析JSON")
                return
            
            # 解析故事大纲数据
            outline_data = template.parse(json_text)
            
            # 按顺序处理各个部分
            outline_parts = []
            
            # 辅助函数，尝试获取属性或字典值
            def get_value(obj, key):
                if hasattr(obj, key):
                    return getattr(obj, key)
                elif isinstance(obj, dict) and key in obj:
                    return obj[key]
                return None
            
            # 1. 标题
            title = get_value(outline_data, "title")
            if title:
                outline_parts.append(f"故事的标题是：{title}")
            
            # 2. 角色
            roles = get_value(outline_data, "roles")
            if roles:
                roles_text = "，".join(roles) if isinstance(roles, list) else str(roles)
                outline_parts.append(f"故事中的角色有：{roles_text}")
            
            # 3. 场景
            scene = get_value(outline_data, "scene")
            if scene:
                outline_parts.append(f"故事发生在：{scene}")
            
            # 4. 大纲
            outline = get_value(outline_data, "outline")
            if outline:
                outline_parts.append(f"故事的情节是：{outline}")
            
            # 5. 寓意
            meaning = get_value(outline_data, "meaning")
            if meaning:
                outline_parts.append(f"故事的寓意是：{meaning}")
                        
            # 依次生成语音
            for i, part in enumerate(outline_parts):
                curr_text_index = text_index + i
                
                # 记录文本位置
                if hasattr(self.conn, 'recode_first_last_text'):
                    self.conn.recode_first_last_text(part, curr_text_index)
                
                # 获取旁白的语音配置
                tts_config = self._get_narrator_voice()
                
                # 使用特定音色生成语音
                if tts_config:
                    # 自定义的TTS处理
                    future = self.conn.executor.submit(
                        self._custom_speak_and_play, part, curr_text_index, tts_config
                    )
                else:
                    # 使用默认TTS处理
                    future = self.conn.executor.submit(
                        self.conn.speak_and_play, part, curr_text_index
                    )
                
                # 加入TTS队列
                self.conn.tts_queue.put(future)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理故事大纲JSON时出错: {e}")
            logger.bind(tag=TAG).error(f"错误详情: {str(e)}")
            # 如果解析失败，尝试作为普通文本处理
            segment_text = get_string_no_punctuation_or_emoji(json_text)
            if segment_text:
                await self._process_text_segment(segment_text, text_index)

def process_streaming_text(connection_handler, llm_responses, dialogue_callback=None, abort_check=None):
    """
    处理流式文本的便捷函数
    
    Args:
        connection_handler: 连接处理器实例
        llm_responses: 大模型流式返回的文本生成器
        dialogue_callback: 处理完成后的对话回调函数
        abort_check: 检查是否中止处理的函数
        
    Returns:
        协程对象，需要用await调用
    """
    processor = StreamingTextProcessor(connection_handler)
    return processor.process_streaming_text(llm_responses, dialogue_callback, abort_check) 