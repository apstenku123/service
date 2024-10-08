# file: models.py
# directory: .

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    ForeignKey,
    DateTime,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from pgvector.sqlalchemy import Vector

Base = declarative_base()

# ORM models with indexes defined
class BaseImageUrl(Base):
    __tablename__ = 'base_image_urls'
    id = Column(Integer, primary_key=True)
    base_url = Column(String, nullable=False)
    images = relationship("Image", back_populates="base_image_url")

    __table_args__ = (
        Index('idx_base_image_urls_base_url', 'base_url'),
    )


class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    base_url_id = Column(Integer, ForeignKey('base_image_urls.id'), nullable=False)
    filename = Column(String, nullable=False)
    processed = Column(Boolean, default=False)
    base_image_url = relationship("BaseImageUrl", back_populates="images")
    embeddings = relationship("ImageEmbedding", back_populates="image")
    archived_image = relationship("ArchivedImage", back_populates="image", uselist=False)

    __table_args__ = (
        UniqueConstraint('base_url_id', 'filename', name='_base_image_uc'),
        Index('idx_images_base_url_id', 'base_url_id'),
        Index('idx_images_processed', 'processed'),
        Index('idx_images_filename', 'filename'),
    )


class Batch(Base):
    __tablename__ = 'batches'
    id = Column(Integer, primary_key=True)
    page_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now())
    processed = Column(Boolean, default=False)
    images = relationship("BatchImage", back_populates="batch")
    logs = relationship("BatchLog", back_populates="batch")

    __table_args__ = (
        UniqueConstraint('page_number', name='_page_number_uc'),
        Index('idx_batches_processed', 'processed'),
    )


class BatchImage(Base):
    __tablename__ = 'batch_images'
    batch_id = Column(Integer, ForeignKey('batches.id'), primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), primary_key=True)
    batch = relationship("Batch", back_populates="images")
    image = relationship("Image")

    __table_args__ = (
        Index('idx_batch_images_batch_id', 'batch_id'),
        Index('idx_batch_images_image_id', 'image_id'),
    )


class Checkpoint(Base):
    __tablename__ = 'checkpoints'
    id = Column(Integer, primary_key=True)
    page_url = Column(String, unique=True, nullable=False)

    __table_args__ = (
        Index('idx_checkpoints_page_url', 'page_url'),
    )


class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    filename = Column(String, nullable=False)
    embedding = Column(Vector(512), nullable=True)
    insightface_embedding = Column(Vector(512), nullable=True)  # Новое поле

    image = relationship("Image", back_populates="embeddings")


    __table_args__ = (
        Index('idx_filename', 'filename'),
        Index('idx_image_embeddings_image_id', 'image_id'),
        Index(
            'idx_image_embeddings_embedding',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_with={"lists": 100},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
        Index(
            'idx_image_embeddings_insightface_embedding',
            'insightface_embedding',
            postgresql_using='ivfflat',
            postgresql_with={"lists": 100},
            postgresql_ops={'insightface_embedding': 'vector_cosine_ops'}
        ),
    )


# New models for host and log data
class HostLog(Base):
    __tablename__ = 'host_logs'
    id = Column(Integer, primary_key=True)
    host_name = Column(String, nullable=False)
    function_name = Column(String, nullable=False)
    log_file = Column(String, nullable=False)
    batches = relationship("BatchLog", back_populates="host_log")

    __table_args__ = (
        Index('idx_host_logs_host_name', 'host_name'),
        Index('idx_host_logs_function_name', 'function_name'),
    )


class BatchLog(Base):
    __tablename__ = 'batch_logs'
    id = Column(Integer, primary_key=True)
    batch_id = Column(Integer, ForeignKey('batches.id'), nullable=False)
    host_log_id = Column(Integer, ForeignKey('host_logs.id'), nullable=False)
    host_log = relationship("HostLog", back_populates="batches")
    batch = relationship("Batch", back_populates="logs")

    __table_args__ = (
        Index('idx_batch_logs_batch_id', 'batch_id'),
    )


class ArchivedImage(Base):
    __tablename__ = 'archived_images'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)  # Добавлено поле внешнего ключа
    filename = Column(String, index=True)
    archive_url = Column(String)
    image = relationship("Image", back_populates="archived_image")

    __table_args__ = (
        Index('idx_archived_images_archive_url', 'archive_url'),
    )


class TestEmbedding(Base):
    __tablename__ = 'test_embeddings'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    embedding = Column(Vector(512), nullable=False)

    __table_args__ = (
        Index('idx_test_embeddings_embedding',
              'embedding',
              postgresql_using='ivfflat',
              postgresql_with={"lists": 100},
              postgresql_ops={'embedding': 'vector_cosine_ops'}
              ),
    )