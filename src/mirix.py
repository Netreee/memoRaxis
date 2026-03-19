from typing import Dict, Any, List, Optional
from .memory_interface import BaseMemorySystem, Evidence

try:
    from MIRIX.remote_client import MirixClient
except ImportError:
    try:
        from mirix.client.remote_client import MirixClient
    except ImportError:
        raise ImportError("Could not import MirixClient. Please ensure MIRIX package is available.")

class Mirix(BaseMemorySystem):
    """
    Adapter for Mirix memory system implementing BaseMemorySystem interface.
    """
    def __init__(self, client: MirixClient, user_id: str = "default_user"):
        self.client = client
        self.user_id = user_id

    def add_memory(self, data: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add memory to Mirix.
        Wraps data in a conversation format (User -> Assistant).
        According to MirixClient.add docstring, messages should end with an assistant turn.
        Supports 'user_id' in metadata to override default user_id.
        Additional metadata keys are passed as filter_tags.
        """
        target_user_id = self.user_id
        filter_tags = {}
        
        if metadata:
            if "user_id" in metadata:
                target_user_id = metadata["user_id"]
            
            # Use metadata as filter_tags (excluding user_id if present)
            for k, v in metadata.items():
                if k != "user_id":
                    filter_tags[k] = v

        messages = [
            {"role": "user", "content": [{"type": "text", "text": data}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Acknowledged."}]}
        ]
        
        # Note: This is an asynchronous operation in Mirix
        self.client.add(
            user_id=target_user_id, 
            messages=messages,
            filter_tags=filter_tags if filter_tags else None
        )

    def retrieve(self, query: str, top_k: int = 5, user_id: Optional[str] = None) -> List[Evidence]:
        """
        Retrieve evidence from Mirix based on a query.
        Uses embedding search (semantic similarity) for better retrieval quality.
        Supports optional user_id argument to override default user_id.
        """
        target_user_id = user_id if user_id is not None else self.user_id

        results = self.client.search(
            user_id=target_user_id,
            query=query,
            memory_type="all",
            search_field="null",
            search_method="embedding",
            limit=top_k,
        )

        evidences = []
        for item in results.get("results", []):
            if not isinstance(item, dict):
                continue
            content = (item.get("summary") or item.get("details") or
                       item.get("content") or item.get("text") or "")
            if item.get("details") and item.get("summary"):
                content = f"{item['summary']}\n{item['details']}"
            if content:
                meta = {k: v for k, v in item.items()
                        if k not in ("summary", "details", "content", "text")}
                meta["source"] = "mirix"
                evidences.append(Evidence(content=content, metadata=meta))

        return evidences

    def reset(self) -> None:
        """
        Reset memory for the current user.
        """
        self.client.clear_memory(user_id=self.user_id)
