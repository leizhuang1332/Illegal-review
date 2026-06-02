import asyncio
import logging
from typing import Dict, List, Optional

from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.illegal_review.data_models import Entity
from src.illegal_review.config.settings import TextAnalysisConfig

logger = logging.getLogger(__name__)

_TAG_TO_TYPE = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
    "TIME": "time",
}


class NERecognizer:
    """命名实体识别 — 使用 ckiplab/bert-base-chinese-ner"""

    def __init__(self, config: TextAnalysisConfig):
        if not config.ner_enabled:
            logger.info("NER disabled by config")
            self._model = None
            self._tokenizer = None
            return

        logger.info("Loading NER model: ckiplab/bert-base-chinese-ner")
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self._model = AutoModelForTokenClassification.from_pretrained(
            "ckiplab/bert-base-chinese-ner"
        )
        self._model.eval()
        self._id2label: Dict[int, str] = self._model.config.id2label
        logger.info("NER model loaded")

    async def recognize(self, text: str) -> List[Entity]:
        """识别命名实体（线程池推理）"""
        if self._model is None:
            return []
        return await asyncio.to_thread(self._recognize_sync, text)

    def _recognize_sync(self, text: str) -> List[Entity]:
        import torch
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]

        tokens = self._tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().tolist()
        )

        entities: List[Entity] = []
        current_entity: Optional[dict] = None

        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            label = self._id2label.get(pred, "O")

            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue

            if token.startswith("##"):
                if current_entity:
                    current_entity["name"] += token[2:]
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities.append(self._make_entity(current_entity, text))
                entity_type = _TAG_TO_TYPE.get(label[2:], "other")
                current_entity = {"name": token, "type": entity_type, "start_pos": None}

            elif label.startswith("I-") and current_entity:
                current_entity["name"] += token

            else:
                if current_entity:
                    entities.append(self._make_entity(current_entity, text))
                    current_entity = None

        if current_entity:
            entities.append(self._make_entity(current_entity, text))

        return entities

    def _make_entity(self, entity_info: dict, text: str) -> Entity:
        name = entity_info["name"]
        start_pos = text.find(name)
        if start_pos == -1:
            start_pos = 0
        return Entity(
            name=name,
            type=entity_info["type"],
            start_pos=start_pos,
            end_pos=start_pos + len(name),
            confidence=1.0,
        )
