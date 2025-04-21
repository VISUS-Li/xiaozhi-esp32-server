import json
import random
import threading  # 添加threading模块导入
import asyncio
import time

from loguru import logger

from config.module_config import get_module_config
# 导入必要的TTS模块
from core.utils import tts
from core.utils.util import get_string_no_punctuation_or_emoji
from concurrent.futures import Future

TAG = __name__


def _get_voice_unique_id(tts_config):
    """获取声音的唯一标识符，用于避免重复使用声音"""
    if not tts_config:
        return None

    tts_type = tts_config.get("type")
    if not tts_type:
        return None

    if tts_type == "edge":
        # 对于Edge TTS，使用voice作为标识符
        voice = tts_config.get("voice")
        if voice:
            # 如果有processing参数，也需要纳入标识符
            processing = tts_config.get("processing", {})
            if processing:
                return f"edge_{voice}_{hash(str(processing))}"
            return f"edge_{voice}"

    elif tts_type == "fishspeech":
        # 对于鱼声TTS，使用reference_audio和reference_text组合作为标识符
        reference_audio = tts_config.get("reference_audio", [])
        reference_text = tts_config.get("reference_text", [])

        if reference_audio and reference_text:
            # 使用第一个音频和文本作为标识符的一部分
            return f"fishspeech_{reference_audio[0]}_{hash(reference_text[0])}"

    # 其他类型的TTS，可以根据需要添加

    # 如果没有明确的唯一标识，使用整个配置的哈希值
    return f"{tts_type}_{hash(str(tts_config))}"


class StoryRespProcessor:
    """流式文本处理器，将流式返回的大语言模型文本按照标点符号分段并进行TTS处理"""
    
    def __init__(self, connection_handler):
        """
        初始化流式文本处理器
        
        Args:
            connection_handler: 连接处理器实例，包含speak_and_play等方法
        """
        self.conn = connection_handler
        self.prompt_manager = self.conn.story_session.prompt_manager
        self.punctuations = ("。", "？", "！", "；", "：", ",", "、", ":", "：")  # 可用于分割的标点符号
        
        # 获取故事模式配置
        self.story_config = {}
        try:
            if hasattr(self.conn, 'config'):
                self.story_config = get_module_config(self.conn.config, "story-module")
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载故事模式配置失败: {e}")
        
        # 检查是否为JSON响应模式
        self.is_json_mode = False
        self.json_buffer = []
        # 角色音色映射，用于保持角色音色一致性
        self.role_voice_mapping = {}
        # 旁白音色映射，用于保持旁白音色一致性
        self.narrator_voice_config = None
        # 已使用的声音列表，用于避免角色和旁白声音重复
        self.used_voices = set()

        threading.Thread(target=self._run_audio_sequencer, daemon=True).start()
    
    async def _process_text_segment(self, text, text_index):
        """处理单个文本段"""
        if hasattr(self.conn, 'recode_first_last_text'):
            self.conn.recode_first_last_text(text, text_index)
        
        # 获取旁白的语音配置
        tts_config = self._get_narrator_voice()
        
        # 使用特定音色生成语音
        self.custom_speak_and_play(text, text_index, tts_config)


    async def _process_json_content(self, stage, text, stage_index):
        """处理llm的输出内容，根据不同的阶段调用不同的处理方法
        
        Args:
            text: 未解析的内容
            stage_index: 阶段索引
        """

        # 尝试获取当前阶段的模板
        template = self.prompt_manager.get_template(stage)

        if not template:
            logger.bind(tag=TAG).warning(f"未找到阶段 {stage} 的模板，无法解析JSON")
            return False
        try:
            # 直接使用模板的parse方法解析JSON文本，使用validate_required=True验证必填字段
            parsed_data = template.parse(text, validate_required=True)
            # 根据当前阶段调用相应的处理方法
            if stage == "story_continuation":
                await self._process_story_continuation(parsed_data, stage_index)
            elif stage == "outline_generation":
                await self._process_story_outline(parsed_data, stage_index)
            else:
                # 对于其他JSON内容，尝试转换为字符串处理
                # 这是一个后备处理方式
                try:
                    if hasattr(parsed_data, "model_dump"):
                        text = json.dumps(parsed_data.model_dump(), ensure_ascii=False)
                    else:
                        text = json.dumps(parsed_data if isinstance(parsed_data, dict) else vars(parsed_data), ensure_ascii=False)
                    
                    segment_text = get_string_no_punctuation_or_emoji(text)
                    if segment_text:
                        await self._process_text_segment(segment_text, stage_index)
                except:
                    logger.bind(tag=TAG).error("无法将解析的数据转换为文本")
            
            return True
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理JSON内容时出错: {e}")
            return False
    
    async def _process_story_continuation(self, story_data, stage_index):
        """处理故事续写阶段的JSON内容"""
        try:
            # 直接使用已经解析的故事数据
            # 检查是否有story_seg字段
            story_seg = story_data.story_seg
            # 处理故事片段
            for i, segment in enumerate(story_seg):
                if segment.is_role:
                    # 处理角色对话
                    await self._process_role_dialogue(segment, stage_index, i)
                else:
                    # 处理旁白
                    await self._process_narrator(segment.content, stage_index, i)
        
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理故事续写数据时出错: {e}")
            # 记录完整的故事数据以便调试
            try:
                logger.bind(tag=TAG).debug(f"故事数据: {story_data}")
            except:
                pass
    
    async def _process_role_dialogue(self, segment, stage_index, seg_index):
        """处理角色对话"""
        try:
            # 支持两种访问方式
            role_name = segment.role_name
            content = segment.content
            mood = segment.mood
            mood_location = segment.mood_location
            
            # 获取性别，用于选择合适的语音
            role_gender = segment.role_gender
            
            # 根据mood_location组装完整句子
            if mood_location == 1:
                # 情绪描述在前
                full_text = f"{role_name}{mood}说道：{content}"
            elif mood is not None:
                # 情绪描述在后
                full_text = f"{role_name}说道：{content}{mood}"
            else:
                full_text = f"{role_name}说道：{content}"
            
            # 获取角色的语音配置
            tts_config = self._get_role_voice(role_name, role_gender)

            # 使用特定音色生成语音
            self.custom_speak_and_play(full_text, stage_index, seg_index, tts_config)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理角色对话时出错: {e}")
            logger.bind(tag=TAG).error(f"错误详情: {str(e)}")
            # 记录segment内容以便调试
            try:
                logger.bind(tag=TAG).debug(f"段落内容: {segment}")
            except:
                pass
    
    async def _process_narrator(self, content, stage_index, seg_index):
        """处理旁白"""
        try:                        
            # 获取旁白的语音配置
            tts_config = self._get_narrator_voice()
            
            # 使用特定音色生成语音
            self.custom_speak_and_play(content, stage_index, seg_index, tts_config)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理旁白时出错: {e}")
    
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
        
        # 过滤掉已使用的声音
        available_voices = []
        for voice in suitable_voices:
            voice_id = _get_voice_unique_id(voice.get("tts", {}))
            if voice_id and voice_id not in self.used_voices:
                available_voices.append(voice)
        
        # 如果所有合适的声音都已被使用，则不进行过滤
        if not available_voices:
            available_voices = suitable_voices
        
        # 随机选择一个音色
        if available_voices:
            selected_voice = random.choice(available_voices)
            # 确保选择的音色配置至少包含type属性
            tts_config = selected_voice.get("tts", {})
            if tts_config and "type" in tts_config:
                # 将声音标记为已使用
                voice_id = _get_voice_unique_id(tts_config)
                if voice_id:
                    self.used_voices.add(voice_id)
                
                # 存储映射关系，保持一致性
                self.role_voice_mapping[role_name] = tts_config
                return tts_config
        
        return None
    
    def _get_narrator_voice(self):
        """获取旁白的语音配置，保持一致性"""
        # 检查是否已有旁白音色配置
        if self.narrator_voice_config:
            return self.narrator_voice_config
        
        # 从配置中获取可用的旁白音色
        narrators = self.story_config.get("voices", {}).get("narrators", [])
        if narrators:
            # 过滤掉已使用的声音
            available_narrators = []
            for voice in narrators:
                voice_id = _get_voice_unique_id(voice.get("tts", {}))
                if voice_id and voice_id not in self.used_voices:
                    available_narrators.append(voice)
            
            # 如果所有旁白声音都已被使用，则不进行过滤
            if not available_narrators:
                available_narrators = narrators
            
            # 随机选择一个旁白音色
            selected_voice = random.choice(available_narrators)
            # 确保选择的音色配置至少包含type属性
            tts_config = selected_voice.get("tts", {})
            if tts_config and "type" in tts_config:
                # 将声音标记为已使用
                voice_id = _get_voice_unique_id(tts_config)
                if voice_id:
                    self.used_voices.add(voice_id)
                
                # 存储音色配置，保持一致性
                self.narrator_voice_config = tts_config
                return tts_config
        return None

    def _get_merged_tts_config(self, tts_config):
        """合并TTS配置"""
        # 首先获取全局配置中对应TTS类型的配置
        tts_type = tts_config.get("type")
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
        
        # 返回合并后的配置
        return merged_config

    def _run_audio_sequencer(self):
        """后台线程，用于按顺序处理生成的音频文件"""
        try:
            while True:
                # 检查下一个要播放的音频是否已经准备好
                stage_key = str(self.conn.story_session.next_play_index)
                if stage_key in self.conn.story_session.tts_stage_dict:
                    # 获取音频信息
                    seg_list = self.conn.story_session.tts_stage_dict.pop(stage_key)
                    for seg in seg_list:
                        text_index, audio_file, text = seg.get('text_index'), seg.get('audio_file'), seg.get('text')
                        cur_text_index = self.conn.story_session.incr_text_index()

                        # 创建一个已完成的Future对象
                        completed_future = Future()
                        completed_future.set_result((audio_file, text, cur_text_index))

                        self.conn.recode_first_last_text(text, cur_text_index)
                        # 将音频放入播放队列
                        self.conn.tts_queue.put(completed_future)
                        logger.bind(tag=TAG).warning(f"顺序播放器: 已将索引 {cur_text_index} 的音频加入播放队列 : {text}")

                    # 更新下一个要播放的索引
                    self.conn.story_session.next_play_index += 1
                else:
                    # 如果下一个要播放的音频尚未准备好，短暂等待
                    time.sleep(0.1)
        except Exception as e:
            logger.bind(tag=TAG).error(f"顺序播放器线程出错: {e}")

    def custom_speak_and_play(self, text, stage_index, seg_index, tts_config):
        """使用指定的TTS配置生成语音并播放"""
        def _custom_speak_and_play_thread(_text, _stage_index, _seg_index, _tts_config):
            """在独立线程中执行的TTS生成和处理函数"""
            try:
                logger.bind(tag=TAG).debug(f"线程开始处理TTS: {_text[:30]}...")
                
                if not _tts_config:
                    # 使用默认TTS方法
                    _audio_file, _text, _index = self.conn.speak_and_play(_text, _stage_index)
                    self.conn.story_session.update_tts_stage_dict(_stage_index, _seg_index, _audio_file, _text)
                    return
                
                # 获取TTS类型
                tts_type = _tts_config.get("type")
                if not tts_type:
                    # 如果没有指定TTS类型，使用默认方法
                    _audio_file, _text, _index = self.conn.speak_and_play(_text, _stage_index)
                    self.conn.story_session.update_tts_stage_dict(_stage_index, _seg_index, _audio_file, _text)
                    return

                # 合并TTS配置
                merged_config = self._get_merged_tts_config(_tts_config)

                # 创建TTS实例
                tts_instance = tts.create_instance(
                    tts_type,
                    merged_config,
                    self.conn.config["delete_audio"]
                )

                if not tts_instance:
                    # 如果创建失败，使用默认方法
                    logger.bind(tag=TAG).info("创建自定义TTS失败，使用默认方法")
                    _audio_file, _text, _index = self.conn.speak_and_play(_text, _stage_index)
                    self.conn.story_session.update_tts_stage_dict(_stage_index, _seg_index, _audio_file, _text)
                    return

                # 生成语音文件
                _audio_file = tts_instance.to_tts(_text)
                if not _audio_file:
                    # 如果生成失败，使用默认方法
                    _audio_file, _text, _index = self.conn.speak_and_play(_text, _stage_index)
                
                # 更新TTS阶段字典（保证按顺序播放）
                logger.bind(tag=TAG).debug(f"TTS生成成功: {_audio_file}")
                self.conn.story_session.update_tts_stage_dict(_stage_index, _seg_index, _audio_file, _text)
            except Exception as e:
                logger.bind(tag=TAG).error(f"独立线程TTS处理出错: {e}")
        
        # 创建并启动独立线程来处理TTS生成
        # daemon=True 确保主程序退出时线程也会退出
        tts_thread = threading.Thread(
            target=_custom_speak_and_play_thread,
            args=(text, stage_index, seg_index, tts_config),
            daemon=True
        )
        tts_thread.start()

    
    async def _process_story_outline(self, outline_data, stage_index):
        """处理故事大纲阶段的JSON内容"""
        try:
            # 直接使用已经解析的大纲数据
            # 按顺序处理各个部分
            outline_parts = []

            # 1. 标题
            if outline_data.title:
                outline_parts.append(f"故事的标题是：{outline_data.title}")
            
            # 2. 角色
            roles = outline_data.roles
            if roles:
                roles_text = "，".join(roles) if isinstance(roles, list) else str(roles)
                outline_parts.append(f"故事中的角色有：{roles_text}")
                        
            # 依次生成语音
            for i, part in enumerate(outline_parts):
                # 记录文本位置
                await self._process_narrator(part, stage_index, i)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理故事大纲数据时出错: {e}")
    
    async def process_complete_text(self, stage, complete_text, is_json_mode=False, stage_index=0):
        """
        处理完整的非流式响应文本
        
        Args:
            complete_text: 完整的响应文本
            is_json_mode: 是否为JSON响应模式
            stage_index: 每个阶段的下标，用来保证按照阶段顺序发送音频
            
        Returns:
            处理状态，成功为True，失败为False
        """
        logger.bind(tag=TAG).info(f"开始处理完整文本，长度: {len(complete_text)}, 是否JSON模式: {is_json_mode}")
        
        try:
            # 设置当前状态
            self.is_json_mode = is_json_mode
            
            
            # 处理文本
            if is_json_mode:
                # JSON模式处理
                json_processed = await self._process_json_content(stage, complete_text, stage_index)
                
                # 如果JSON处理失败，按普通文本处理
                if not json_processed:
                    logger.bind(tag=TAG).warning("JSON处理失败，按普通文本处理")
                    return await self._process_as_normal_text(complete_text, stage_index)
                
                return json_processed
            else:
                # 普通文本模式处理
                return await self._process_as_normal_text(complete_text, stage_index)
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理完整文本时出错: {e}")
            return False
    
    async def _process_as_normal_text(self, text, start_text_index):
        """将完整文本按标点符号分段并处理"""
        try:
            text_index = start_text_index
            processed_chars = 0
            
            # 查找所有标点符号位置
            punct_positions = []
            for i, char in enumerate(text):
                if char in self.punctuations:
                    punct_positions.append(i)
            
            # 如果没有找到标点，整个文本作为一段处理
            if not punct_positions:
                segment_text = get_string_no_punctuation_or_emoji(text)
                if segment_text:
                    text_index += 1
                    await self._process_text_segment(segment_text, text_index)
                return True
            
            # 按标点分段处理
            last_pos = 0
            for pos in punct_positions:
                segment_text_raw = text[last_pos:pos+1]
                segment_text = get_string_no_punctuation_or_emoji(segment_text_raw)
                if segment_text:
                    text_index += 1
                    await self._process_text_segment(segment_text, text_index)
                last_pos = pos + 1
            
            # 处理最后一段
            if last_pos < len(text):
                segment_text = get_string_no_punctuation_or_emoji(text[last_pos:])
                if segment_text:
                    text_index += 1
                    await self._process_text_segment(segment_text, text_index)
            
            return True
        except Exception as e:
            logger.bind(tag=TAG).error(f"按普通文本处理时出错: {e}")
            return False
