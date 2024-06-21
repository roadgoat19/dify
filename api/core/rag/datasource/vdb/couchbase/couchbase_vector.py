import json
import logging
from datetime import timedelta
from typing import Any

import couchbase.search as search
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.management.collections import CollectionManager

# needed for options -- cluster, timeout, SQL++ (N1QL) query, etc.
from couchbase.options import ClusterOptions, SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch
from flask import current_app
from pydantic import BaseModel, root_validator

from core.rag.datasource.entity.embedding import Embeddings
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from models.dataset import Dataset

logger = logging.getLogger(__name__)


class CouchbaseConfig(BaseModel):
    connection_string: str
    user: str
    password: str
    bucket_name: str
    scope_name: str
    index_name: str

    @root_validator()
    def validate_config(cls, values: dict) -> dict:
        if not values.get('connection_string'):
            raise ValueError("config COUCHBASE_CONNECTION_STRING is required")
        if not values.get('user'):
            raise ValueError("config COUCHBASE_USER is required")
        if not values.get('password'):
            raise ValueError("config COUCHBASE_PASSWORD is required")
        if not values.get('bucket_name'):
            raise ValueError("config COUCHBASE_PASSWORD is required")
        if not values.get('scope_name'):
            raise ValueError("config COUCHBASE_SCOPE_NAME is required")
        if not values.get('index_name'):
            raise ValueError("config COUCHBASE_INDEX_NAME is required")
        return values
    def to_couchbase_params(self):
        return {
            'connection_string': self.connection_string,
            'user': self.user,
            'password': self.password,
            'bucket_name': self.bucket_name,
            'scope_name': self.scope_name,
            'index_name': self.index_name
        }
    
class CouchbaseVector(BaseVector):

    def __init__(self, collection_name: str, config: CouchbaseConfig):
        super().__init__(collection_name)
        self._client_config = config

        """Connect to couchbase"""

        auth = PasswordAuthenticator(config.user, config.password)
        options = ClusterOptions(auth)
        self._cluster = Cluster(config.connection_string, options)
        self._bucket = self._cluster.bucket(config.bucket_name)
        self._scope = self._bucket.scope(config.scope_name)
        self._index_name = config.index_name
        self._bucket_name = config.bucket_name

        # Wait until the cluster is ready for use.
        self._cluster.wait_until_ready(timedelta(seconds=5))

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        metadatas = [d.metadata for d in texts]
        self.create_collection(embeddings, metadatas)
        self.add_texts(texts, embeddings)

    def create_collection(self):
        lock_name = 'vector_indexing_lock_{}'.format(self._collection_name)
        with redis_client.lock(lock_name, timeout=20):
            collection_exist_cache_key = 'vector_indexing_{}'.format(self._collection_name)
            if redis_client.get(collection_exist_cache_key):
                return
            manager = CollectionManager(self._bucket, self._client_config.bucket_name)
            manager.create_collection(self._client_config.scope_name, self._collection_name)
            redis_client.set(collection_exist_cache_key, 1, ex=3600)

    def get_type(self) -> str:
        return VectorType.COUCHBASE

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        uuids = self._get_uuids(documents)
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        doc_ids = []

        documents_to_insert = [
            {
                id: {
                    'text': text,
                    'embedding': vector,
                    'metadata': metadata,
                }
                for id, text, vector, metadata in zip(
                    uuids, texts, embeddings, metadatas
                )
            }
        ]

        result = self._scope.collection(self._collection_name).upsert_multi(documents_to_insert)
        if result.all_ok:
            doc_ids.extend(documents_to_insert.keys())
        
        return doc_ids


    def text_exists(self, id: str) -> bool:
        query = f"SELECT COUNT(1) AS count FROM `{self._client_config.bucket_name}`.`{self._client_config.scope_name}`.`{self._collection_name}` WHERE `id` = {id}"
        result = self._cluster.query(query)
        for row in result:
            return row['count'] > 0

    def delete_by_ids(self, ids: list[str]) -> None:
        for doc_id in ids:
            query = f"""
                DELETE FROM `{self._client_config.bucket_name}`.`{self._client_config.scope_name}`.`{self._collection_name}`
                WHERE id = {doc_id};
                """
            self._cluster.query(query)
            

    def delete_by_document_id(self, document_id: str):
        query = f"""
                DELETE FROM `{self._client_config.bucket_name}`.`{self._client_config.scope_name}`.`{self._collection_name}`
                WHERE id = {document_id};
                """
        self._cluster.query(query)

    def get_ids_by_metadata_field(self, key: str, value: str):
        query = f"""
            SELECT id FROM `{self._client_config.bucket_name}`.`{self._client_config.scope_name}`.`{self._collection_name}`
            WHERE metadata.{key} = '{value}';
            """
        result = self._cluster.query(query)
        return [row[0] for row in result]

    
    def delete_by_metadata_field(self, key: str, value: str) -> None:
        query = f"""
            DELETE FROM `{self._client_config.bucket_name}`.`{self._client_config.scope_name}`.`{self._collection_name}`
            WHERE metadata.{key} = '{value}';
            """
        self._cluster.query(query)
        
    def search_by_vector(
            self,
            query_vector: list[float],
            **kwargs: Any
    ) -> list[Document]:
        top_k = kwargs.get("top_k", 5)
        score_threshold = kwargs.get("score_threshold") if kwargs.get("score_threshold") else 0.0

        search_req = search.SearchRequest.create(
            VectorSearch.from_vector_query(
                VectorQuery(
                    'embedding',
                    query_vector,
                    top_k,
                )
            )
        )
        try:
            search_iter = self._cluster.search(
                    index=self._index_name,
                    request=search_req,
                    options=SearchOptions(limit=top_k),
                )

            docs = []

            # Parse the results
            for row in search_iter.rows():
                text = row.fields.pop('text', "")
                metadata = row.fields.pop('metadata', "")
                score = row.score
                doc = Document(page_content=text, metadata=metadata)
                if score >= score_threshold:
                    docs.append(doc)

        except Exception as e:
            raise ValueError(f"Search failed with error: {e}")

        return docs

    def search_by_full_text(
            self, query: str,
            **kwargs: Any
    ) -> list[Document]:
        top_k=kwargs.get('top_k', 2)
        try:
            search_iter = self._cluster.search_query(self._bucket_name,
                                        search.MatchQuery(query),
                                        options=SearchOptions(limit=top_k))
            docs = []
            for row in search_iter.rows():
                text = row.fields.pop('text', "")
                metadata = row.fields.pop('metadata', "")
                docs.append(Document(page_content = text, metadata=metadata))

        except Exception as e:
            raise ValueError(f"Search failed with error: {e}")
        
        return docs

class CouchbaseVectorFactory(AbstractVectorFactory):
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> CouchbaseVector:
        if dataset.index_struct_dict:
            class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
            collection_name = class_prefix
        else:
            dataset_id = dataset.id
            collection_name = Dataset.gen_collection_name_by_id(dataset_id)
            dataset.index_struct = json.dumps(
                self.gen_index_struct_dict(VectorType.COUCHBASE, collection_name))

        config = current_app.config
        return CouchbaseVector(
            collection_name=collection_name,
            config=CouchbaseConfig(
                connection_string=config.get('COUCHBASE_CONNECTION_STRING'),
                user=config.get('COUCHBASE_USER'),
                password=config.get('COUCHBASE_PASSWORD'),
                bucket_name=config.get('COUCHBASE_BUCKET_NAME'),
                scope_name=config.get('COUCHBASE_SCOPE_NAME'),
                index_name=config.get('COUCHBASE_INDEX_NAME')
            )
        )