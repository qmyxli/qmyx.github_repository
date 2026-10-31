"""ReAct范式智能体 - 推理-行动循环调度"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
# 加载当前目录下 .env 文件
load_dotenv()


class ThoughtType(Enum):
    """思考类型"""
    OBSERVE = "observe"       # 观察当前状态
    REASON = "reason"         # 推理下一步
    ACT = "act"              # 执行动作
    REFLECT = "reflect"      # 反思结果
    FINAL = "final"          # 最终答案


@dataclass
class ReActStep:
    """ReAct单步记录"""
    step_num: int
    thought_type: ThoughtType
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    is_final: bool = False


@dataclass
class ReActResult:
    """ReAct执行结果"""
    final_answer: str
    steps: List[ReActStep] = field(default_factory=list)
    total_steps: int = 0
    success: bool = True
    error: Optional[str] = None


class ReActAgent:
    """
    ReAct范式智能体
    
    核心循环:
    Thought → Action → Observation → Thought → ... → Final Answer
    
    工具链:
    - SQL查询工具 (NL2SQL)
    - 数据处理工具 (DataAnalyzer)
    - 可视化工具 (DataVisualizer)
    - 知识库检索工具 (KnowledgeBase)
    - 报告生成工具 (ReportGenerator)
    """
    
    def __init__(self, tools: Dict[str, Any], llm=None,
                 max_steps: int = 10, system_prompt: str = None):
        """
        初始化ReAct Agent
        
        Args:
            tools: 可用工具字典 {tool_name: tool_function}
            llm: LLM模型实例
            max_steps: 最大推理步数
            system_prompt: 系统提示词
        """
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.step_history: List[ReActStep] = []
        
        logger.info(f"✓ ReAct Agent初始化完成 (工具: {list(tools.keys())}, "
                   f"最大步数: {max_steps})")
    
    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        tool_descriptions = "\n".join([
            f"  - {name}: {getattr(tool, 'description', '无描述')}"
            if isinstance(tool, object) else f"  - {name}: 函数工具"
            for name, tool in self.tools.items()
        ])
        
        return f"""你是一个专业的电商数据分析智能体，使用ReAct（推理+行动）范式解决问题。

## 可用工具
{tool_descriptions}

## ReAct循环
每一步你必须按以下格式输出:

Thought: [你的推理过程，分析当前状态和下一步行动]
Action: [工具名称]
Action Input: [工具输入参数，JSON格式]

系统会返回:
Observation: [工具执行结果]

然后继续下一轮Thought...

当你认为已经获得足够信息可以回答时，输出:

Thought: 我已经获得了足够的信息来回答问题。
Final Answer: [你的最终答案]

## 要求
1. 每次只执行一个Action
2. 基于Observation结果决定下一步
3. 如果工具执行失败，分析原因并尝试其他方法
4. 保持推理过程简洁、有逻辑
5. 最终答案必须基于实际数据结果，不要编造数据
"""
    
    def run(self, query: str) -> ReActResult:
        """
        执行ReAct循环
        
        Args:
            query: 用户查询
        
        Returns:
            ReActResult
        """
        logger.info(f"ReAct Agent开始处理: {query}")
        
        self.step_history = []
        step_num = 0
        context = f"用户查询: {query}\n"
        
        while step_num < self.max_steps:
            step_num += 1
            logger.info(f"--- ReAct 步骤 {step_num} ---")
            
            # 生成Thought + Action
            thought, action, action_input, is_final, final_answer = \
                self._think(context, step_num)
            
            # 记录思考步骤
            step = ReActStep(
                step_num=step_num,
                thought_type=ThoughtType.REASON,
                thought=thought,
                action=action,
                action_input=action_input,
                is_final=is_final
            )
            
            if is_final:
                step.thought_type = ThoughtType.FINAL
                step.observation = final_answer
                self.step_history.append(step)
                
                logger.info(f"ReAct完成 (共{step_num}步)")
                return ReActResult(
                    final_answer=final_answer,
                    steps=self.step_history,
                    total_steps=step_num,
                    success=True
                )
            
            # 执行Action
            observation = self._execute_action(action, action_input)
            step.observation = observation
            self.step_history.append(step)
            
            logger.info(f"Action: {action}, Observation: {observation[:100]}...")
            
            # 更新上下文
            context += f"\n步骤{step_num}:\n"
            context += f"Thought: {thought}\n"
            context += f"Action: {action}\n"
            context += f"Observation: {observation}\n"
        
        # 超过最大步数
        logger.warning(f"ReAct超过最大步数 ({self.max_steps})")
        return ReActResult(
            final_answer="抱歉，分析问题需要更多步骤，已达到最大推理步数限制。请尝试简化问题。",
            steps=self.step_history,
            total_steps=step_num,
            success=False,
            error="超过最大推理步数"
        )
    
    def _think(self, context: str, step_num: int) -> Tuple[str, Optional[str], 
                                                            Optional[Dict], bool, str]:
        """
        推理阶段
        
        Returns:
            (thought, action, action_input, is_final, final_answer)
        """
        if self.llm:
            return self._think_with_llm(context, step_num)
        else:
            return self._think_with_rules(context, step_num)
    
    def _think_with_llm(self, context: str, step_num: int) -> Tuple:
        """使用LLM进行推理"""
        prompt = self.system_prompt + "\n\n" + context
        
        if step_num > 1:
            prompt += "\n请继续推理下一步。如果已有足够信息，请给出Final Answer。"
        else:
            prompt += "\n请开始分析问题并执行第一步。"
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # 解析ReAct格式
            return self._parse_react_output(content)
        except Exception as e:
            logger.error(f"LLM推理失败: {e}")
            return f"推理出错: {e}", None, None, True, "推理过程出错，请重试"
    
    def _think_with_rules(self, context: str, step_num: int) -> Tuple:
        """基于规则的推理（无LLM降级方案）"""
        context_lower = context.lower()
        
        # 简单规则引擎
        if step_num == 1:
            # 第一步：判断需要什么数据
            if any(kw in context_lower for kw in ['sql', '查询', '数据库', '统计', '多少']):
                return (
                    "用户需要查询数据，我应该使用NL2SQL工具",
                    "nl2sql_query",
                    {"query": context.split("用户查询:")[1].strip() if "用户查询:" in context else ""},
                    False, ""
                )
            elif any(kw in context_lower for kw in ['分析', 'rfm', '趋势', '客户']):
                return (
                    "用户需要数据分析，我应该使用数据分析工具",
                    "analyze_data",
                    {"analysis_type": "descriptive"},
                    False, ""
                )
            elif any(kw in context_lower for kw in ['图表', '可视化', '画图']):
                return (
                    "用户需要可视化，我应该使用可视化工具",
                    "generate_chart",
                    {"chart_type": "sales_trend"},
                    False, ""
                )
            elif any(kw in context_lower for kw in ['报告', 'report']):
                return (
                    "用户需要报告，我应该使用报告生成工具",
                    "generate_report",
                    {"format": "html"},
                    False, ""
                )
            elif any(kw in context_lower for kw in ['指标', '定义', '口径']):
                return (
                    "用户询问指标定义，我应该使用知识库检索工具",
                    "search_knowledge",
                    {"query": context.split("用户查询:")[1].strip() if "用户查询:" in context else ""},
                    False, ""
                )
        
        # 默认：基于上一步观察决定
        if "observation" in context_lower:
            return (
                "我已经获得了数据结果，可以生成最终答案",
                None, None, True,
                self._summarize_observations(context)
            )
        
        return (
            "继续分析中...",
            None, None, True,
            "基于当前信息，无法完成完整分析。请提供更具体的查询。"
        )
    
    def _parse_react_output(self, content: str) -> Tuple:
        """解析LLM的ReAct格式输出"""
        thought = ""
        action = None
        action_input = None
        is_final = False
        final_answer = ""
        
        # 提取Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|$)', 
                                  content, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 检查是否是Final Answer
        final_match = re.search(r'Final Answer:\s*(.+)', content, re.DOTALL)
        if final_match:
            is_final = True
            final_answer = final_match.group(1).strip()
            return thought, None, None, True, final_answer
        
        # 提取Action
        action_match = re.search(r'Action:\s*(.+)', content)
        if action_match:
            action = action_match.group(1).strip()
        
        # 提取Action Input
        input_match = re.search(r'Action Input:\s*(\{.+?\})', content, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # 尝试修复JSON
                json_str = input_match.group(1).strip()
                json_str = json_str.replace("'", '"')
                try:
                    action_input = json.loads(json_str)
                except:
                    action_input = {"raw_input": json_str}
        
        return thought, action, action_input, False, ""
    
    def _execute_action(self, action: str, action_input: Dict) -> str:
        """执行动作"""
        if not action:
            return "无动作可执行"
        
        if action not in self.tools:
            return f"未知工具: {action}。可用工具: {list(self.tools.keys())}"
        
        try:
            tool = self.tools[action]
            
            # 调用工具
            if hasattr(tool, 'invoke'):
                # LangChain工具
                result = tool.invoke(action_input or {})
            elif callable(tool):
                # 普通函数
                result = tool(**(action_input or {}))
            else:
                return f"工具 {action} 不可调用"
            
            # 格式化观察结果
            if isinstance(result, str):
                return result
            elif hasattr(result, 'to_string'):
                return result.to_string()
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"工具执行失败 ({action}): {e}")
            return f"工具执行失败: {e}"
    
    def _summarize_observations(self, context: str) -> str:
        """总结所有观察结果"""
        observations = re.findall(r'Observation:\s*(.+?)(?=\n步骤|\Z)', 
                                  context, re.DOTALL)
        
        if not observations:
            return "未获取到有效信息"
        
        summary_parts = ["基于分析结果："]
        for i, obs in enumerate(observations, 1):
            obs_clean = obs.strip()[:200]
            summary_parts.append(f"{i}. {obs_clean}")
        
        return "\n".join(summary_parts)
    
    def get_step_history(self) -> List[ReActStep]:
        """获取推理步骤历史"""
        return self.step_history
    
    def format_trace(self) -> str:
        """格式化推理轨迹"""
        lines = ["=" * 60, "ReAct 推理轨迹", "=" * 60]
        
        for step in self.step_history:
            lines.append(f"\n--- 步骤 {step.step_num} ---")
            lines.append(f"Thought: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}")
            if step.action_input:
                lines.append(f"Action Input: {step.action_input}")
            if step.observation:
                lines.append(f"Observation: {step.observation[:200]}...")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试用例：无LLM时ReAct智能体走规则推理并返回答案
    def fake_tool(**kwargs):
        return "工具执行成功"

    agent = ReActAgent(tools={'analyze_data': fake_tool}, llm=None, max_steps=5)
    result = agent.run("帮我分析一下销售数据")
    assert result.final_answer, "ReAct未返回答案"
    print(f"测试通过: 最终答案 -> {result.final_answer[:40]}")
