import yaml
import os
from typing import Dict, Any, Optional


class PromptManager:
    def __init__(self, prompts_dir: str = None):
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__))
        )
        self._index = None
        self._loaded_prompts = {}

    def _load_index(self) -> Dict[str, Any]:
        if self._index is None:
            index_path = os.path.join(self.prompts_dir, "index.yaml")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"索引文件不存在: {index_path}")
            with open(index_path, "r", encoding="utf-8") as f:
                self._index = yaml.safe_load(f)
        return self._index

    def _load_prompt_file(self, file_path: str) -> str:
        if file_path in self._loaded_prompts:
            return self._loaded_prompts[file_path]

        full_path = os.path.join(self.prompts_dir, file_path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                self._loaded_prompts[file_path] = content
                return content
        return ""

    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        index = self._load_index()
        return index.get("agents", {}).get(agent_name)

    def get_agent_prompt(self, agent_name: str) -> str:
        config = self.get_agent_config(agent_name)
        if not config:
            return ""

        main_prompt = self._load_prompt_file(config.get("main_prompt", ""))

        skills_prompt = ""
        for skill_name in config.get("skills", []):
            skill_config = self._load_index().get("skills", {}).get(skill_name)
            if skill_config:
                skill_content = self._load_prompt_file(skill_config.get("file", ""))
                skills_prompt += f"\n\n## {skill_config.get('description', '')}\n{skill_content}"

        template_prompt = ""
        output_format = config.get("output_format", "text")
        if output_format == "json":
            template_path = self._load_index().get("templates", {}).get("json_output", "")
            if template_path:
                template_content = self._load_prompt_file(template_path)
                template_prompt += f"\n\n## 输出格式\n{template_content}"

        return f"{main_prompt}{skills_prompt}{template_prompt}"

    def get_tool_call_template(self) -> str:
        index = self._load_index()
        template_path = index.get("templates", {}).get("tool_call", "")
        return self._load_prompt_file(template_path)

    def get_all_agent_names(self) -> list:
        index = self._load_index()
        return list(index.get("agents", {}).keys())

    def get_skill_description(self, skill_name: str) -> Optional[str]:
        index = self._load_index()
        skill = index.get("skills", {}).get(skill_name)
        return skill.get("description") if skill else None

    def reload(self):
        self._index = None
        self._loaded_prompts = {}


_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager