# from transformers.models.layoutlm import *

# flake8: noqa
# from .data.funsd import FunsdDataset
from .layoutlm import (
    LayoutlmConfig,
    LayoutlmForSequenceClassification,
    LayoutlmForTokenClassification,
)

from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from .Utils import *