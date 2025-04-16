from config.logger import setup_logging
import json
from core.handle.sendAudioHandle import send_stt_message
from core.storymode.storySession import StorySession
from core.utils.dialogue import Message
from loguru import logger
import asyncio
from core.prompts import PromptManager
import re
from core.storymode.streaming_processor import StreamingTextProcessor

TAG = __name__
logger = setup_logging()


async def handle_story_mode(conn, text):
    """处理故事模式请求"""
    # 检查是否已经在故事模式中
    if hasattr(conn, 'in_story_mode') and conn.in_story_mode:
        # 已经在故事模式，继续故事流程
        return await continue_story_mode(conn, text)
    else:
        # 进入故事模式
        return await enter_story_mode(conn, text)


async def enter_story_mode(conn, text):
    """进入故事模式"""
    logger.bind(tag=TAG).info("用户进入故事模式")
    
    # 标记用户进入故事模式
    conn.in_story_mode = True
    
    # 创建故事会话
    if not hasattr(conn, 'story_session'):
        conn.story_session = StorySession(conn, conn.session_id)
    
    # 发送用户输入的文本
    await send_stt_message(conn, text)
    
    # 记录对话
    conn.dialogue.put(Message(role="user", content=text))
    
    # 初始化提示词管理器
    prompt_manager = PromptManager()
    
    # 获取故事模式的初始提示词
    initial_prompt = prompt_manager.get_template("story_mode_intro")
    
    # 获取当前文本索引
    text_index = conn.tts_last_text_index + 1 if hasattr(conn, 'tts_last_text_index') else 0
    conn.recode_first_last_text(initial_prompt.template, text_index)
    
    # 生成并播放初始反馈
    future = conn.executor.submit(conn.speak_and_play, initial_prompt.template, text_index)
    conn.tts_queue.put(future)
    
    # 记录对话
    if initial_prompt.formatted_prompt is not None or initial_prompt.formatted_prompt != "":
        conn.dialogue.put(Message(role="assistant", content=initial_prompt.formatted_prompt))
    
    # 提取故事主题
    theme_data = await extract_story_theme(conn, text)
    conn.story_session.theme_data = theme_data
    
    # 更新故事阶段
    conn.story_session.update_stage("outline_generation")
    
    # 启动故事大纲生成任务
    asyncio.create_task(generate_story_outline(conn, theme_data))
    
    return True


async def continue_story_mode(conn, text):
    """继续故事模式的对话"""
    # 检查上一次大模型返回中是否仍在故事模式
    last_assistant_message = conn.dialogue.get_last_assistant_message()
    
    if last_assistant_message and is_story_mode_active(last_assistant_message):
        # 发送用户输入文本
        await send_stt_message(conn, text)
        
        # 记录对话
        conn.dialogue.put(Message(role="user", content=text))
        
        # 处理用户输入，继续故事
        asyncio.create_task(process_user_story_input(conn, text))
        
        return True
    else:
        # 故事模式已结束
        conn.in_story_mode = False
        logger.bind(tag=TAG).info("用户退出故事模式")
        return False


def is_story_mode_active(message):
    """检查大模型返回的消息中是否包含故事模式标记"""
    try:
        # 尝试解析消息中的JSON部分
        content = message.content
        # 寻找可能的JSON字符串
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            json_part = content[start_idx:end_idx+1]
            data = json.loads(json_part)
            
            # 如果存在ended字段且为真，表示故事已结束
            if "ended" in data and data["ended"] is True:
                logger.bind(tag=TAG).info("检测到故事已结束标记")
                return False
                
            # 检查story_mode标记
            return data.get("story_mode", False)
    except Exception as e:
        logger.bind(tag=TAG).error(f"解析故事模式状态时出错: {str(e)}")
    
    # 默认假设非故事模式
    return False


async def generate_story_outline(conn, theme_data=None):
    """生成故事大纲并缓存"""
    try:
        # 获取大纲生成阶段的 LLM
        _llm = conn.story_session.get_llm("outline_generation")
        
        # 获取大纲生成阶段的对话历史
        dialogue = conn.story_session.get_dialogue("outline_generation")

        # 使用提示词管理器获取提示词模板
        prompt_manager = conn.story_session.prompt_manager
        prompt_template = prompt_manager.get_template("outline_generation")

        # 更新对话历史：系统提示放在system消息中，主题数据放在user消息中
        dialogue.update_system_message(prompt_template.formatted_prompt)
        # 如果有主题数据，添加到提示词中
        if theme_data:
            dialogue.put(Message(role="user", content=theme_data))
        
        # 创建缓存结构
        conn.story_session.outline_cache = {
            "generating": True,
            "completed": False,
            "content": "",
            "tts_chunks": []
        }
        
        # 导入流式处理器
        
        # 创建处理器实例
        processor = StreamingTextProcessor(conn)
        
        # 调用大模型进行流式生成
        llm_responses = _llm.response(conn.session_id, dialogue.get_llm_dialogue())
        
        # 处理流式响应的回调函数
        def on_completion(complete_response):
            # 更新缓存内容
            conn.story_session.outline_cache["content"] = complete_response
            conn.story_session.outline_cache["generating"] = False
            conn.story_session.outline_cache["completed"] = True
            
            # 记录助手回复
            dialogue.put(Message(role="assistant", content=complete_response))
            conn.dialogue.put(Message(role="assistant", content=complete_response))
            
            # 更新故事阶段为交互阶段
            conn.story_session.update_stage("story_continuation")
            
            # 预先缓存故事续写的第一部分
            asyncio.create_task(prepare_story_continuation(conn))
        
        # 使用流式处理器处理
        result = await processor.process_streaming_text(
            llm_responses,
            dialogue_callback=on_completion
        )
        
        # 缓存TTS索引（从处理器内部的记录获取）
        conn.story_session.outline_cache["tts_chunks"] = list(range(
            conn.tts_first_text_index, 
            conn.tts_last_text_index + 1
        ))
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"生成故事大纲时出错: {str(e)}")
        # 出错时结束故事模式
        conn.in_story_mode = False
        
        # 通知用户出错
        error_message = "很抱歉，故事创作过程中出现了问题。让我们下次再试吧。"
        text_index = conn.tts_last_text_index + 1 if hasattr(conn, 'tts_last_text_index') else 0
        conn.recode_first_last_text(error_message, text_index)
        future = conn.executor.submit(conn.speak_and_play, error_message, text_index)
        conn.tts_queue.put(future)


async def process_user_story_input(conn, text):
    """处理用户在故事模式中的输入"""
    try:
        # 如果存在预缓存的续写内容，标记为中断
        if hasattr(conn.story_session, 'continuation_cache'):
            conn.story_session.continuation_cache["is_interrupted"] = True
            conn.story_session.continuation_cache["is_paused"] = True
        
        # 判断用户输入与故事的相关性
        judgement = await judge_user_input_relevance(conn, text)
        logger.bind(tag=TAG).info(f"用户输入相关性判断: {judgement}")
        
        # 如果判断不应继续故事，退出故事模式
        if not judgement.get("should_continue", True):
            conn.in_story_mode = False
            logger.bind(tag=TAG).info("根据用户输入判断，退出故事模式")
            
            # 发送故事结束提示
            exit_message = "好的，我们暂时结束故事讲述。希望您喜欢这个故事！有任何其他需要，请告诉我。"
            text_index = conn.tts_last_text_index + 1 if hasattr(conn, 'tts_last_text_index') else 0
            conn.recode_first_last_text(exit_message, text_index)
            future = conn.executor.submit(conn.speak_and_play, exit_message, text_index)
            conn.tts_queue.put(future)
            
            # 记录对话
            conn.dialogue.put(Message(role="assistant", content=exit_message))
            return
        
        # 如果需要新的大纲，重新生成故事
        if judgement.get("need_new_outline", False):
            logger.bind(tag=TAG).info("根据用户输入，需要重新生成故事大纲")
            
            # 更新主题数据（如果需要）
            new_theme_data = await extract_story_theme(conn, text)
            conn.story_session.theme_data = new_theme_data
            
            # 更新阶段
            conn.story_session.update_stage("outline_generation")
            
            # 通知用户正在调整故事
            adjust_message = "根据您的建议，我将调整故事情节。请稍等..."
            text_index = conn.tts_last_text_index + 1 if hasattr(conn, 'tts_last_text_index') else 0
            conn.recode_first_last_text(adjust_message, text_index)
            future = conn.executor.submit(conn.speak_and_play, adjust_message, text_index)
            conn.tts_queue.put(future)
            
            # 记录对话
            conn.dialogue.put(Message(role="assistant", content=adjust_message))
            
            # 重新生成大纲
            asyncio.create_task(generate_story_outline(conn, new_theme_data))
            return
        
        # 正常处理用户输入，继续故事
        current_stage = conn.story_session.stage
        _llm = conn.story_session.get_llm(current_stage)
        dialogue = conn.story_session.get_dialogue(current_stage)
        dialogue.put(Message(role="user", content=text))
        
        # 构建提示词
        prompt_template = conn.story_session.prompt_manager.get_template(f"story_mode_{current_stage}")
        
        # 导入流式处理器
        from core.storymode.streaming_processor import StreamingTextProcessor
        
        # 创建处理器实例
        processor = StreamingTextProcessor(conn)
        
        # 调用大模型获取流式响应
        llm_responses = _llm.response(conn.session_id, dialogue.get_llm_dialogue())
        
        # 处理流式响应的回调函数
        def on_completion(complete_response):
            # 记录助手回复
            dialogue.put(Message(role="assistant", content=complete_response))
            conn.dialogue.put(Message(role="assistant", content=complete_response))
            
            # 检查是否退出故事模式
            if not is_story_mode_active(Message(role="assistant", content=complete_response)):
                conn.in_story_mode = False
                logger.bind(tag=TAG).info("故事模式结束")
            else:
                # 检查故事是否已经结束
                try:
                    # 尝试解析JSON以检查ended标志
                    story_data = prompt_template.parse(complete_response)
                    ended = getattr(story_data, "ended", False) if hasattr(story_data, "ended") else story_data.get("ended", False)
                    
                    if ended:
                        logger.bind(tag=TAG).info("故事已经结束，不再准备后续内容")
                        # 修改连接状态，标记故事模式结束
                        conn.in_story_mode = False
                        
                        # 发送故事结束提示
                        exit_message = "故事讲完了。希望您喜欢这个故事！有任何其他需要，请告诉我。"
                        text_index = conn.tts_last_text_index + 1 if hasattr(conn, 'tts_last_text_index') else 0
                        conn.recode_first_last_text(exit_message, text_index)
                        future = conn.executor.submit(conn.speak_and_play, exit_message, text_index)
                        conn.tts_queue.put(future)
                        return
                except Exception as e:
                    logger.bind(tag=TAG).error(f"检查故事结束状态时出错: {e}")
                
                # 只有故事未结束时，才准备下一段内容
                asyncio.create_task(prepare_story_continuation(conn))
        
        # 使用流式处理器处理
        await processor.process_streaming_text(
            llm_responses,
            dialogue_callback=on_completion
        )
            
    except Exception as e:
        logger.bind(tag=TAG).error(f"处理用户故事输入时出错: {str(e)}")
        # 出错时可能需要结束故事模式
        conn.in_story_mode = False


async def check_story_mode_keywords(conn, text):
    """检查文本中是否包含故事模式的触发关键词"""
    # 故事模式的触发关键词
    story_triggers = ["讲个故事", "开始故事", "故事模式", "讲故事", "讲一个故事", 
                     "说个故事", "故事时间", "开启故事模式"]
    
    # 检查文本是否包含触发词
    for trigger in story_triggers:
        if trigger in text:
            return True
    
    return False 


async def prepare_story_continuation(conn):
    """预先准备故事续写内容"""
    try:
        # 获取故事续写阶段的 LLM
        _llm = conn.story_session.get_llm("story_continuation")
        
        # 获取故事续写阶段的对话历史
        dialogue = conn.story_session.get_dialogue("story_continuation")
        
        # 获取大纲内容
        outline = conn.story_session.outline_cache.get("content", "")
        
        # 创建续写缓存结构
        conn.story_session.continuation_cache = {
            "generating": True,
            "completed": False,
            "content": "",
            "tts_chunks": [],
            "current_position": 0,  # 当前播放位置
            "is_paused": False,     # 是否暂停
            "is_interrupted": False # 是否被用户中断
        }
        
        # 使用提示词管理器获取续写提示模板
        prompt_manager = conn.story_session.prompt_manager
        prompt_template = prompt_manager.get_template("story_continuation")
        
        # 更新系统提示
        dialogue.update_system_message(prompt_template.formatted_prompt)
        input_prompt = prompt_template.get_input_prompt({
            "outline": outline,
            "before": conn.story_session.outline_cache.get("content", "")
        })

        # 更新用户消息
        dialogue.put(Message(role="user", content=input_prompt))

        llm_responses = _llm.response(conn.session_id, dialogue.get_llm_dialogue())

        # 创建处理器实例
        processor = StreamingTextProcessor(conn)
        
        def on_completion(complete_response):
            # 更新缓存内容
            conn.story_session.continuation_cache["content"] = complete_response
            conn.story_session.continuation_cache["generating"] = False
            conn.story_session.continuation_cache["completed"] = True
            
            # 记录助手回复
            dialogue.put(Message(role="assistant", content=complete_response))
            conn.dialogue.put(Message(role="assistant", content=complete_response))
            
            # 更新故事阶段为交互阶段
            conn.story_session.update_stage("story_continuation")
            
            # 检查故事是否已经结束
            try:
                # 尝试解析JSON以检查ended标志
                story_data = prompt_template.parse(complete_response)
                ended = getattr(story_data, "ended", False) if hasattr(story_data, "ended") else story_data.get("ended", False)
                
                if ended:
                    logger.bind(tag=TAG).info("故事已经结束，不再准备后续内容")
                    # 修改连接状态，标记故事模式结束
                    conn.in_story_mode = False
                    return
            except Exception as e:
                logger.bind(tag=TAG).error(f"检查故事结束状态时出错: {e}")
            
            # 只有故事未结束时，才准备下一段内容
            asyncio.create_task(prepare_story_continuation(conn))
        
        # 使用流式处理器处理
        result = await processor.process_streaming_text(
            llm_responses,
            dialogue_callback=on_completion
        )
        logger.bind(tag=TAG).info("故事续写内容已预先缓存")
        
    except Exception as e:
        logger.bind(tag=TAG).error(f"预准备故事续写内容时出错: {str(e)}")
        # 标记缓存生成失败
        if hasattr(conn.story_session, 'continuation_cache'):
            conn.story_session.continuation_cache["generating"] = False
            conn.story_session.continuation_cache["completed"] = False

async def extract_story_theme(conn, text):
    """从用户输入中提取故事主题，并结合用户信息"""
    try:
        # 获取内容判断LLM
        _llm = conn.story_session.get_llm("story_theme_extraction")
        
        # 构建提示词
        user_info = {}
        
        # 如果有用户记忆，可以添加用户信息
        if hasattr(conn, 'memory') and conn.memory:
            memory_str = await conn.memory.query_memory(text)
            if memory_str:
                user_info["memory"] = memory_str
        
        # 添加对话历史中的关键信息
        user_info["dialogue_history"] = conn.dialogue.get_last_n_message_without_system(5)
        
        # 获取提示词模板和JSON Schema
        prompt_manager = conn.story_session.prompt_manager
        template_data = prompt_manager.get_template("story_theme_extraction")
        input_prompt = template_data.get_input_prompt({
            "user_info": user_info,
            "text": text
        })
        
        # 调用LLM分析
        theme_result = _llm.response_no_stream(
            system_prompt=template_data.formatted_prompt,
            user_prompt=input_prompt
        )

        default_theme = {
            "theme": "一般故事",
            "age_group": "通用", 
            "style": "温馨",
            "has_explicit_theme": False
        }
        
        # 解析结果
        try:
            # 尝试使用解析器解析结果
            theme_data = template_data.parse(theme_result)
            # 如果是Pydantic模型，转换为字符串
            if hasattr(theme_data, "model_dump"):
                return json.dumps(theme_data.model_dump(), ensure_ascii=False)
            return json.dumps(theme_data, ensure_ascii=False)
        except Exception as parse_error:
            logger.bind(tag=TAG).error(f"解析主题数据时出错: {str(parse_error)}")
            # 如果解析失败，尝试直接返回原始结果
            if isinstance(theme_result, str):
                return theme_result
            
            # 默认返回一般主题
            return json.dumps(default_theme, ensure_ascii=False)
            
    except Exception as e:
        logger.bind(tag=TAG).error(f"提取故事主题时出错: {str(e)}")
        # 返回默认主题
        return json.dumps(default_theme, ensure_ascii=False)

async def judge_user_input_relevance(conn, text):
    """判断用户输入是否与故事的相关性"""
    try:
        # 获取内容判断LLM
        _llm = conn.story_session.get_llm("content_judge")
        
        # 获取当前故事内容
        story_content = ""
        if hasattr(conn.story_session, 'outline_cache') and conn.story_session.outline_cache.get("completed"):
            story_content += conn.story_session.outline_cache.get("content", "")
        
        if hasattr(conn.story_session, 'continuation_cache') and conn.story_session.continuation_cache.get("content"):
            story_content += "\n" + conn.story_session.continuation_cache.get("content", "")
        
        # 根据内容长度智能截取摘要
        if len(story_content) <= 100:
            story_summary = story_content
        elif len(story_content) <= 500:
            story_summary = story_content[-100:] + "..."
        else:
            story_summary = story_content[:100] + "..." + story_content[-100:]
        
        # 获取提示词模板和JSON Schema
        prompt_manager = conn.story_session.prompt_manager
        template_data = prompt_manager.get_template("content_judge")
        # 调用LLM进行判断
        judgement = await _llm.response_no_stream(
            system_prompt=template_data.formatted_prompt,
            user_prompt=template_data.get_input_prompt({
                "text": text,
                "story_content": story_summary
            })
        )
        
        # 解析JSON结果
        try:
            # 尝试使用解析器解析结果
            judgement_data = template_data.parse(judgement)
            # 如果是Pydantic模型，转换为字典
            if hasattr(judgement_data, "model_dump"):
                return judgement_data.model_dump()
            return judgement_data
        except Exception as parse_error:
            logger.bind(tag=TAG).error(f"解析判断结果时出错: {str(parse_error)}")
            # 如果解析失败，尝试直接解析JSON
            try:
                return json.loads(judgement)
            except json.JSONDecodeError:
                # 如果直接解析JSON也失败，尝试提取JSON部分
                match = re.search(r'\{.*\}', judgement, re.DOTALL)
                if match:
                    judgement_json = match.group(0)
                    try:
                        return json.loads(judgement_json)
                    except:
                        pass
            
            # 默认返回继续故事
            return {
                "is_relevant": True,
                "should_continue": True,
                "need_new_outline": False,
                "explanation": "解析结果出错，默认继续故事"
            }
            
    except Exception as e:
        logger.bind(tag=TAG).error(f"判断用户输入与故事的相关性时出错: {str(e)}")
        # 默认继续故事
        return {
            "is_relevant": True,
            "should_continue": True,
            "need_new_outline": False,
            "explanation": "判断过程出错，默认继续故事"
        } 