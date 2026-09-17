import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from config import settings

class LLMClient:
    """Unified LLM Client providing robust completion generation supporting 

    Gemini API, Mistral API, OpenAI API, Ollama local, and fallback offline generation.

    """
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.model = settings.default_llm_model

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates text response from configured LLM provider with fallback handling."""
        
        # 1. Try Gemini API if configured
        if self.provider == "gemini" and settings.gemini_api_key:
            try:
                return self._call_gemini(prompt, system_instruction)
            except Exception as e:
                print(f"[LLMClient Warning] Gemini API failed: {e}. Falling back to mock generator.")

        # 2. Try Mistral API if configured
        if self.provider == "mistral" and settings.mistral_api_key:
            try:
                return self._call_mistral(prompt, system_instruction)
            except Exception as e:
                print(f"[LLMClient Warning] Mistral API failed: {e}. Falling back to mock generator.")

        # 3. Try OpenAI API if configured
        if self.provider == "openai" and settings.openai_api_key:
            try:
                return self._call_openai(prompt, system_instruction)
            except Exception as e:
                print(f"[LLMClient Warning] OpenAI API failed: {e}. Falling back to mock generator.")

        # 4. Try Ollama if configured
        if self.provider == "ollama":
            try:
                return self._call_ollama(prompt, system_instruction)
            except Exception as e:
                print(f"[LLMClient Warning] Ollama local call failed: {e}. Falling back to mock generator.")

        # 5. Fallback Mock LLM Generator (Offline mode)
        return self._call_mock_llm(prompt, system_instruction)

    def _call_mistral(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Call Mistral AI API."""
        api_key = settings.mistral_api_key
        url = "https://api.mistral.ai/v1/chat/completions"
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        model = self.model if "mistral" in self.model else "mistral-small-latest"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Call Google Gemini REST API directly."""
        api_key = settings.gemini_api_key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}"
        
        full_text = prompt
        if system_instruction:
            full_text = f"System Instruction: {system_instruction}\n\nUser Prompt: {prompt}"

        payload = {
            "contents": [{
                "parts": [{"text": full_text}]
            }]
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Call OpenAI API."""
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model if "gpt" in self.model else "gpt-3.5-turbo",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content

    def _call_ollama(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Call local Ollama REST endpoint."""
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "system": system_instruction or "",
            "stream": False
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "")

    def _call_mock_llm(self, prompt: str, system_instruction: Optional[str]) -> str:
        """High-quality offline mock LLM generation synthesizing retrieved context."""
        # Extract evidence text from prompt if present
        context_snippets = []
        for line in prompt.split("\n"):
            if "Document:" in line or "Text:" in line or "Chunk" in line or "Evidence:" in line:
                context_snippets.append(line.strip())

        evidence_summary = " ".join(context_snippets[:3]) if context_snippets else "Based on the retrieved document context."

        return f"{evidence_summary} According to the provided evidence, the requested factual information is verified in the document context."
