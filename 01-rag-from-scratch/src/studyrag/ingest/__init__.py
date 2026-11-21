"""Document loading: files -> Documents made of located Blocks."""

from .loaders import SUPPORTED, file_hash, load_document
from .models import Block, Document, location_label

__all__ = ["SUPPORTED", "Block", "Document", "file_hash", "load_document", "location_label"]
