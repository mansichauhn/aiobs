"""OpenAI LLM adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import io

from .base import BaseLLM, LLMMessage, LLMResponse


class OpenAILLM(BaseLLM):
    """LLM adapter for OpenAI and OpenAI-compatible APIs.
    
    Works with:
    - OpenAI
    - Azure OpenAI
    - Groq
    - Together AI
    - Any OpenAI-compatible API
    
    Supports multiple OpenAI APIs:
    - Chat Completions (complete, complete_async, complete_messages)
    - Embeddings (create_embeddings, create_embeddings_async)
    - Completions (legacy text completion API)
    - Audio (transcriptions and text-to-speech)
    - Images (DALL-E image generation)
    - Moderations (content moderation)
    
    Example:
        from openai import OpenAI
        from aiobs.llm import LLM
        
        client = OpenAI()
        llm = LLM.from_client(client, model="gpt-4o")
        
        # Chat completions
        response = llm.complete("Hello!")
        
        # Embeddings
        embeddings = llm.create_embeddings("Hello world")
        
        # Audio transcription
        transcription = llm.transcribe_audio("audio.mp3")
        
        # Image generation
        image_url = llm.generate_image("A sunset over mountains")
        
        # Content moderation
        moderation = llm.moderate("Check this text")
    """
    
    provider: str = "openai"
    
    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize OpenAI LLM adapter.
        
        Args:
            client: OpenAI client instance.
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        super().__init__(client, model, temperature, max_tokens)
    
    @classmethod
    def is_compatible(cls, client: Any) -> bool:
        """Check if client is OpenAI-compatible.
        
        Args:
            client: Client instance to check.
            
        Returns:
            True if client has OpenAI-compatible interface.
        """
        return (
            hasattr(client, "chat") 
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        )
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build messages list for API call.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            
        Returns:
            List of message dicts.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse.
        
        Args:
            response: Raw OpenAI response.
            
        Returns:
            Parsed LLMResponse.
        """
        content = response.choices[0].message.content or ""
        
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            raw_response=response,
        )
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion synchronously.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            LLMResponse with generated content.
        """
        messages = self._build_messages(prompt, system_prompt)
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        
        call_kwargs.update(kwargs)
        
        response = self.client.chat.completions.create(**call_kwargs)
        return self._parse_response(response)
    
    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion asynchronously.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            LLMResponse with generated content.
        """
        messages = self._build_messages(prompt, system_prompt)
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        
        call_kwargs.update(kwargs)
        
        # Check if client has async support
        if hasattr(self.client.chat.completions, "acreate"):
            response = await self.client.chat.completions.acreate(**call_kwargs)
        else:
            # Fallback: run sync in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(**call_kwargs)
            )
        
        return self._parse_response(response)
    
    def complete_messages(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from a list of messages.
        
        Args:
            messages: List of conversation messages.
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            LLMResponse with generated content.
        """
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        
        call_kwargs.update(kwargs)
        
        response = self.client.chat.completions.create(**call_kwargs)
        return self._parse_response(response)
    
    # ==================== Embeddings API ====================
    
    def create_embeddings(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create embeddings for text input.
        
        Args:
            input: Text or list of texts to embed.
            model: Embedding model name (default: "text-embedding-3-small").
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing embeddings data, model, and usage.
            
        Example:
            llm = OpenAILLM(client, model="gpt-4o")
            result = llm.create_embeddings("Hello world")
            embeddings = result["data"][0]["embedding"]
        """
        if not hasattr(self.client, "embeddings"):
            raise ValueError("Client does not support embeddings API")
        
        model = model or "text-embedding-3-small"
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "input": input,
        }
        call_kwargs.update(kwargs)
        
        response = self.client.embeddings.create(**call_kwargs)
        
        return {
            "data": [
                {
                    "index": item.index,
                    "embedding": item.embedding,
                    "object": getattr(item, "object", "embedding"),
                }
                for item in response.data
            ],
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "raw_response": response,
        }
    
    async def create_embeddings_async(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create embeddings for text input asynchronously.
        
        Args:
            input: Text or list of texts to embed.
            model: Embedding model name (default: "text-embedding-3-small").
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing embeddings data, model, and usage.
        """
        if not hasattr(self.client, "embeddings"):
            raise ValueError("Client does not support embeddings API")
        
        model = model or "text-embedding-3-small"
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "input": input,
        }
        call_kwargs.update(kwargs)
        
        if hasattr(self.client.embeddings, "acreate"):
            response = await self.client.embeddings.acreate(**call_kwargs)
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embeddings.create(**call_kwargs)
            )
        
        return {
            "data": [
                {
                    "index": item.index,
                    "embedding": item.embedding,
                    "object": getattr(item, "object", "embedding"),
                }
                for item in response.data
            ],
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "raw_response": response,
        }
    
    # ==================== Completions API (Legacy) ====================
    
    def complete_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate text completion using legacy completions API.
        
        Note: This is the legacy API. Prefer using complete() for chat completions.
        
        Args:
            prompt: Text prompt to complete.
            model: Model name (default: uses instance model or "gpt-3.5-turbo-instruct").
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing completion text, model, and usage.
            
        Example:
            llm = OpenAILLM(client, model="gpt-3.5-turbo-instruct")
            result = llm.complete_text("The capital of France is")
            print(result["text"])  # "Paris."
        """
        if not hasattr(self.client, "completions"):
            raise ValueError("Client does not support completions API")
        
        model = model or self.model or "gpt-3.5-turbo-instruct"
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        
        call_kwargs.update(kwargs)
        
        response = self.client.completions.create(**call_kwargs)
        
        return {
            "text": response.choices[0].text if response.choices else "",
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "raw_response": response,
        }
    
    async def complete_text_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate text completion asynchronously using legacy completions API.
        
        Args:
            prompt: Text prompt to complete.
            model: Model name (default: uses instance model or "gpt-3.5-turbo-instruct").
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing completion text, model, and usage.
        """
        if not hasattr(self.client, "completions"):
            raise ValueError("Client does not support completions API")
        
        model = model or self.model or "gpt-3.5-turbo-instruct"
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        
        call_kwargs.update(kwargs)
        
        if hasattr(self.client.completions, "acreate"):
            response = await self.client.completions.acreate(**call_kwargs)
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.completions.create(**call_kwargs)
            )
        
        return {
            "text": response.choices[0].text if response.choices else "",
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "raw_response": response,
        }
    
    # ==================== Audio API ====================
    
    def transcribe_audio(
        self,
        audio: Union[str, Path, bytes, io.BufferedReader],
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe audio to text using Whisper API.
        
        Args:
            audio: Audio file path, bytes, or file-like object.
            model: Model name (default: "whisper-1").
            language: Language code (e.g., "en", "es", "fr").
            prompt: Optional prompt to guide the model's style.
            response_format: Response format: "json", "text", "srt", "verbose_json", "vtt".
            temperature: Sampling temperature (0.0 to 1.0).
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing transcription text and metadata.
            
        Example:
            llm = OpenAILLM(client, model="gpt-4o")
            result = llm.transcribe_audio("audio.mp3")
            print(result["text"])
        """
        if not hasattr(self.client, "audio"):
            raise ValueError("Client does not support audio API")
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "file": audio,
            "response_format": response_format,
        }
        
        if language:
            call_kwargs["language"] = language
        if prompt:
            call_kwargs["prompt"] = prompt
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        
        call_kwargs.update(kwargs)
        
        response = self.client.audio.transcriptions.create(**call_kwargs)
        
        if isinstance(response, str):
            return {"text": response, "raw_response": response}
        else:
            return {
                "text": response.text,
                "language": getattr(response, "language", None),
                "duration": getattr(response, "duration", None),
                "words": getattr(response, "words", None),
                "segments": getattr(response, "segments", None),
                "raw_response": response,
            }
    
    async def transcribe_audio_async(
        self,
        audio: Union[str, Path, bytes, io.BufferedReader],
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe audio to text asynchronously using Whisper API.
        
        Args:
            audio: Audio file path, bytes, or file-like object.
            model: Model name (default: "whisper-1").
            language: Language code (e.g., "en", "es", "fr").
            prompt: Optional prompt to guide the model's style.
            response_format: Response format: "json", "text", "srt", "verbose_json", "vtt".
            temperature: Sampling temperature (0.0 to 1.0).
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing transcription text and metadata.
        """
        if not hasattr(self.client, "audio"):
            raise ValueError("Client does not support audio API")
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "file": audio,
            "response_format": response_format,
        }
        
        if language:
            call_kwargs["language"] = language
        if prompt:
            call_kwargs["prompt"] = prompt
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        
        call_kwargs.update(kwargs)
        
        if hasattr(self.client.audio.transcriptions, "acreate"):
            response = await self.client.audio.transcriptions.acreate(**call_kwargs)
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.audio.transcriptions.create(**call_kwargs)
            )
        
        if isinstance(response, str):
            return {"text": response, "raw_response": response}
        else:
            return {
                "text": response.text,
                "language": getattr(response, "language", None),
                "duration": getattr(response, "duration", None),
                "words": getattr(response, "words", None),
                "segments": getattr(response, "segments", None),
                "raw_response": response,
            }
    
    def text_to_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> bytes:
        """Convert text to speech using TTS API.
        
        Args:
            text: Text to convert to speech.
            voice: Voice to use: "alloy", "echo", "fable", "onyx", "nova", "shimmer".
            model: Model name: "tts-1" or "tts-1-hd".
            speed: Speed multiplier (0.25 to 4.0).
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Audio data as bytes.
            
        Example:
            llm = OpenAILLM(client, model="gpt-4o")
            audio_data = llm.text_to_speech("Hello, world!")
            with open("output.mp3", "wb") as f:
                f.write(audio_data)
        """
        if not hasattr(self.client, "audio"):
            raise ValueError("Client does not support audio API")
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
        }
        call_kwargs.update(kwargs)
        
        response = self.client.audio.speech.create(**call_kwargs)
        
        # Response is a binary stream
        return response.content if hasattr(response, "content") else bytes(response)
    
    async def text_to_speech_async(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> bytes:
        """Convert text to speech asynchronously using TTS API.
        
        Args:
            text: Text to convert to speech.
            voice: Voice to use: "alloy", "echo", "fable", "onyx", "nova", "shimmer".
            model: Model name: "tts-1" or "tts-1-hd".
            speed: Speed multiplier (0.25 to 4.0).
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Audio data as bytes.
        """
        if not hasattr(self.client, "audio"):
            raise ValueError("Client does not support audio API")
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
        }
        call_kwargs.update(kwargs)
        
        if hasattr(self.client.audio.speech, "acreate"):
            response = await self.client.audio.speech.acreate(**call_kwargs)
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.audio.speech.create(**call_kwargs)
            )
        
        return response.content if hasattr(response, "content") else bytes(response)
    
    # ==================== Images API ====================
    
    def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        response_format: str = "url",
        style: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image using DALL-E API.
        
        Args:
            prompt: Text description of the image to generate.
            model: Model name: "dall-e-2" or "dall-e-3".
            n: Number of images to generate (1 for dall-e-3, 1-10 for dall-e-2).
            size: Image size: "256x256", "512x512", "1024x1024" (dall-e-2),
                  "1024x1024", "1792x1024", "1024x1792" (dall-e-3).
            quality: Image quality: "standard" or "hd" (dall-e-3 only).
            response_format: Response format: "url" or "b64_json".
            style: Image style: "vivid" or "natural" (dall-e-3 only).
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing image URLs or base64 data.
            
        Example:
            llm = OpenAILLM(client, model="gpt-4o")
            result = llm.generate_image("A sunset over mountains")
            print(result["data"][0]["url"])
        """
        if not hasattr(self.client, "images"):
            raise ValueError("Client does not support images API")
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
        }
        
        if model == "dall-e-3":
            if quality:
                call_kwargs["quality"] = quality
            if style:
                call_kwargs["style"] = style
        
        call_kwargs.update(kwargs)
        
        response = self.client.images.generate(**call_kwargs)
        
        return {
            "data": [
                {
                    "url": getattr(item, "url", None),
                    "b64_json": getattr(item, "b64_json", None),
                    "revised_prompt": getattr(item, "revised_prompt", None),
                }
                for item in response.data
            ],
            "created": getattr(response, "created", None),
            "raw_response": response,
        }
    
    async def generate_image_async(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        response_format: str = "url",
        style: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image asynchronously using DALL-E API.
        
        Args:
            prompt: Text description of the image to generate.
            model: Model name: "dall-e-2" or "dall-e-3".
            n: Number of images to generate (1 for dall-e-3, 1-10 for dall-e-2).
            size: Image size: "256x256", "512x512", "1024x1024" (dall-e-2),
                  "1024x1024", "1792x1024", "1024x1792" (dall-e-3).
            quality: Image quality: "standard" or "hd" (dall-e-3 only).
            response_format: Response format: "url" or "b64_json".
            style: Image style: "vivid" or "natural" (dall-e-3 only).
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing image URLs or base64 data.
        """
        if not hasattr(self.client, "images"):
            raise ValueError("Client does not support images API")
        
        call_kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
        }
        
        if model == "dall-e-3":
            if quality:
                call_kwargs["quality"] = quality
            if style:
                call_kwargs["style"] = style
        
        call_kwargs.update(kwargs)
        
        if hasattr(self.client.images, "agenerate"):
            response = await self.client.images.agenerate(**call_kwargs)
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.images.generate(**call_kwargs)
            )
        
        return {
            "data": [
                {
                    "url": getattr(item, "url", None),
                    "b64_json": getattr(item, "b64_json", None),
                    "revised_prompt": getattr(item, "revised_prompt", None),
                }
                for item in response.data
            ],
            "created": getattr(response, "created", None),
            "raw_response": response,
        }
    
    # ==================== Moderations API ====================
    
    def moderate(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check if content violates OpenAI's usage policies.
        
        Args:
            input: Text or list of texts to moderate.
            model: Moderation model (default: "text-moderation-latest").
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing moderation results with categories and scores.
            
        Example:
            llm = OpenAILLM(client, model="gpt-4o")
            result = llm.moderate("Check this text for violations")
            if result["results"][0]["flagged"]:
                print("Content flagged:", result["results"][0]["categories"])
        """
        if not hasattr(self.client, "moderations"):
            raise ValueError("Client does not support moderations API")
        
        model = model or "text-moderation-latest"
        
        call_kwargs: Dict[str, Any] = {
            "input": input,
            "model": model,
        }
        call_kwargs.update(kwargs)
        
        response = self.client.moderations.create(**call_kwargs)
        
        # Parse categories and scores safely
        def extract_categories(categories_obj):
            """Extract categories from Pydantic model or dict."""
            if hasattr(categories_obj, "model_dump"):
                return categories_obj.model_dump()
            elif hasattr(categories_obj, "dict"):
                return categories_obj.dict()
            elif hasattr(categories_obj, "__dict__"):
                return {k: v for k, v in categories_obj.__dict__.items() if not k.startswith("_")}
            return {}
        
        return {
            "id": response.id,
            "model": response.model,
            "results": [
                {
                    "flagged": result.flagged,
                    "categories": extract_categories(result.categories),
                    "category_scores": extract_categories(result.category_scores),
                }
                for result in response.results
            ],
            "raw_response": response,
        }
    
    async def moderate_async(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check if content violates OpenAI's usage policies asynchronously.
        
        Args:
            input: Text or list of texts to moderate.
            model: Moderation model (default: "text-moderation-latest").
            **kwargs: Additional arguments passed to the API.
            
        Returns:
            Dict containing moderation results with categories and scores.
        """
        if not hasattr(self.client, "moderations"):
            raise ValueError("Client does not support moderations API")
        
        model = model or "text-moderation-latest"
        
        call_kwargs: Dict[str, Any] = {
            "input": input,
            "model": model,
        }
        call_kwargs.update(kwargs)
        
        if hasattr(self.client.moderations, "acreate"):
            response = await self.client.moderations.acreate(**call_kwargs)
        else:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.moderations.create(**call_kwargs)
            )
        
        # Parse categories and scores safely
        def extract_categories(categories_obj):
            """Extract categories from Pydantic model or dict."""
            if hasattr(categories_obj, "model_dump"):
                return categories_obj.model_dump()
            elif hasattr(categories_obj, "dict"):
                return categories_obj.dict()
            elif hasattr(categories_obj, "__dict__"):
                return {k: v for k, v in categories_obj.__dict__.items() if not k.startswith("_")}
            return {}
        
        return {
            "id": response.id,
            "model": response.model,
            "results": [
                {
                    "flagged": result.flagged,
                    "categories": extract_categories(result.categories),
                    "category_scores": extract_categories(result.category_scores),
                }
                for result in response.results
            ],
            "raw_response": response,
        }

