"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 mem0ai/mem0
Original code from https://github.com/mem0ai/mem0
Licensed under the Apache License, Version 2.0
"""

import logging
import uuid
from typing import Any, Dict, List, Mapping, Optional

try:
    import weaviate
except ImportError:
    raise ImportError(
        "The 'weaviate' library is required. Please install it using 'pip install weaviate-client weaviate'."
    )

import weaviate.classes.config as wvcc
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import get_valid_uuid

from .base import BaseVectorStore, OutputData

logger = logging.getLogger(__name__)


class WeaviateVectorStore(BaseVectorStore):
    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int = 768,
        cluster_url: str = None,
        auth_client_secret: str = None,
        additional_headers: dict = None,
    ):
        """
        Initialize the Weaviate vector store.

        Args:
            collection_name (str): Name of the collection/class in Weaviate.
            embedding_model_dims (int, optional): Dimensions of the embedding model.
            client (WeaviateClient, optional): Existing Weaviate client instance. Defaults to None.
            cluster_url (str, optional): URL for Weaviate server. Defaults to None.
            auth_config (dict, optional): Authentication configuration for Weaviate. Defaults to None.
            additional_headers (dict, optional): Additional headers for requests. Defaults to None.
        """
        super().__init__(embedding_model_dims)

        if "localhost" in cluster_url:
            self.client = weaviate.connect_to_local(headers=additional_headers)
        else:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=Auth.api_key(auth_client_secret),
                headers=additional_headers,
            )

        # Weaviate capitalizes the first letter of collection names, so we need to handle this
        self.collection_name = collection_name
        self.weaviate_collection_name = collection_name.capitalize()
        self.create_collection(embedding_model_dims)

    def _parse_output(self, data: Dict) -> List[OutputData]:
        """
        Parse the output data.

        Args:
            data (Dict): Output data.

        Returns:
            List[OutputData]: Parsed output data.
        """
        keys = ["ids", "distances", "metadatas"]
        values = []

        for key in keys:
            value = data.get(key, [])
            if isinstance(value, list) and value and isinstance(value[0], list):
                value = value[0]
            values.append(value)

        ids, distances, metadatas = values
        max_length = max(len(v) for v in values if isinstance(v, list) and v is not None)

        result = []
        for i in range(max_length):
            entry = OutputData(
                id=ids[i] if isinstance(ids, list) and ids and i < len(ids) else None,
                score=(distances[i] if isinstance(distances, list) and distances and i < len(distances) else None),
                payload=(metadatas[i] if isinstance(metadatas, list) and metadatas and i < len(metadatas) else None),
            )
            result.append(entry)

        return result

    def create_collection(self, vector_size: int, distance: str = "cosine") -> None:
        """
        Create a new collection with the specified schema.

        Args:
            vector_size (int): Size of the vectors to be stored.
            distance (str, optional): Distance metric for vector similarity. Defaults to "cosine".
        """
        if self.client.collections.exists(self.weaviate_collection_name):
            logging.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
            return

        properties = [
            wvcc.Property(name="ids", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="hash", data_type=wvcc.DataType.TEXT),
            wvcc.Property(
                name="metadata",
                data_type=wvcc.DataType.TEXT,
                description="Additional metadata",
            ),
            wvcc.Property(name="data", data_type=wvcc.DataType.TEXT),
            wvcc.Property(name="created_at", data_type=wvcc.DataType.TEXT),
            wvcc.Property(
                name="category",
                data_type=wvcc.DataType.TEXT,
                index_filterable=True,
                index_searchable=False,  # Disable text search for exact matching
                tokenization=wvcc.Tokenization.FIELD,  # Use field tokenization for exact matching
            ),
            wvcc.Property(name="updated_at", data_type=wvcc.DataType.TEXT),
        ]

        vector_config = wvcc.Configure.Vectors.self_provided()

        self.client.collections.create(
            self.weaviate_collection_name,
            vector_config=vector_config,
            properties=properties,
        )

    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        # Validate vector dimensions
        self._validate_vector_dimensions(vectors)

        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        with self.client.batch.fixed_size(batch_size=100) as batch:
            for idx, vector in enumerate(vectors):
                object_id = ids[idx] if ids and idx < len(ids) else str(uuid.uuid4())
                object_id = get_valid_uuid(object_id)

                data_object = payloads[idx] if payloads and idx < len(payloads) else {}

                # Ensure 'id' is not included in properties (it's used as the Weaviate object ID)
                if "ids" in data_object:
                    del data_object["ids"]

                batch.add_object(
                    collection=self.weaviate_collection_name,
                    properties=data_object,
                    uuid=object_id,
                    vector=vector,
                )

    def search(
        self,
        query: str,
        vectors: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[OutputData]:
        """
        Search for similar vectors.
        """
        logger.info(f"Searching in collection {self.weaviate_collection_name} with query: {query}")
        logger.info(f"Vector dimensions: {len(vectors)}, limit: {limit}")

        collection = self.client.collections.get(str(self.weaviate_collection_name))
        filter_conditions = []
        if filters:
            for key, value in filters.items():
                if value:
                    filter_conditions.append(Filter.by_property(key).equal(value))
        combined_filter = Filter.all_of(filter_conditions) if filter_conditions else None

        try:
            # Try vector similarity search first
            response = collection.query.near_vector(
                near_vector=vectors,
                limit=limit,
                filters=combined_filter,
                distance=0.99,  # Use a very permissive distance threshold
                # Return all properties to support custom fields
                return_metadata=MetadataQuery(score=True),
            )

            # If no results from vector search, fall back to filtered object retrieval
            if len(response.objects) == 0:
                response = collection.query.fetch_objects(limit=limit, filters=combined_filter)

        except Exception as e:
            logger.error(f"Search failed with error: {e}")
            raise

        results = []
        for obj in response.objects:
            payload = obj.properties.copy()
            payload["id"] = str(obj.uuid).split("'")[0]  # Include the id in the payload
            results.append(
                OutputData(
                    id=str(obj.uuid),
                    score=(
                        1 if obj.metadata.distance is None else 1 - obj.metadata.distance
                    ),  # Convert distance to score
                    payload=payload,
                )
            )
        return results

    def delete(self, vector_id: str) -> None:
        """
        Delete a vector by ID.

        Args:
            vector_id: ID of the vector to delete.
        """
        collection = self.client.collections.get(str(self.weaviate_collection_name))
        collection.data.delete_by_id(vector_id)

    def update(
        self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a vector and its payload.

        Args:
            vector_id: ID of the vector to update.
            vector (list, optional): Updated vector. Defaults to None.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        collection = self.client.collections.get(str(self.weaviate_collection_name))

        if payload:
            collection.data.update(uuid=vector_id, properties=payload)

        if vector:
            existing_data = self.get(vector_id)
            if existing_data:
                existing_data = dict(existing_data)
                if "id" in existing_data:
                    del existing_data["id"]
                existing_payload: Mapping[str, str] = existing_data
                collection.data.update(uuid=vector_id, properties=existing_payload, vector=vector)

    def close(self) -> None:
        """
        Close the Weaviate client connection.
        """
        if hasattr(self, "client") and self.client:
            self.client.close()

    def get(self, vector_id: str) -> Optional[OutputData]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id: ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector and metadata.
        """
        vector_id = get_valid_uuid(vector_id)
        collection = self.client.collections.get(str(self.weaviate_collection_name))

        response = collection.query.fetch_object_by_id(
            uuid=vector_id,
        )
        if response is None:
            return None

        payload = response.properties.copy()
        payload["id"] = str(response.uuid).split("'")[0]
        results = OutputData(
            id=str(response.uuid).split("'")[0],
            score=1.0,
            payload=payload,
        )
        return results

    def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        collections = self.client.collections.list_all()
        logger.debug(f"collections: {collections}")
        print(f"collections: {collections}")

        # collections.list_all() returns a dict where keys are collection names
        if isinstance(collections, dict):
            collection_names = list(collections.keys())
        else:
            # Handle case where it returns a list of objects with .name
            collection_names = [col.name if hasattr(col, "name") else str(col) for col in collections]

        # Return simple list of collection names for test compatibility
        # Convert back to original naming convention (uncapitalized)
        original_names = []
        for name in collection_names:
            # If this matches our weaviate collection name, return the original collection name
            if name == self.weaviate_collection_name:
                original_names.append(self.collection_name)
            else:
                # For other collections, convert first letter back to lowercase
                original_names.append(name[0].lower() + name[1:] if name else name)
        return original_names

    def delete_collection(self) -> None:
        """Delete a collection."""
        self.client.collections.delete(self.weaviate_collection_name)

    def collection_info(self) -> Dict[str, Any]:
        """
        Get information about a collection.

        Returns:
            dict: Collection information.
        """
        try:
            collection = self.client.collections.get(self.weaviate_collection_name)
            # Get collection config/schema information
            config = collection.config.get()
            return {"name": self.collection_name, "config": str(config)}
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"name": self.collection_name, "error": str(e)}

    def list(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[OutputData]:
        """
        List all vectors in a collection.
        """
        collection = self.client.collections.get(self.weaviate_collection_name)
        filter_conditions = []
        if filters:
            for key, value in filters.items():
                if value:
                    filter_conditions.append(Filter.by_property(key).equal(value))
        combined_filter = Filter.all_of(filter_conditions) if filter_conditions else None

        try:
            # Use fetch_objects with filters when filters are applied
            if combined_filter:
                response = collection.query.fetch_objects(
                    limit=limit,
                    filters=combined_filter,
                    # Return all properties to support custom fields
                )
            else:
                # No filters, just fetch all objects
                response = collection.query.fetch_objects(
                    limit=limit,
                    # Return all properties to support custom fields
                )
        except Exception as e:
            logger.error(f"List failed with error: {e}")
            raise

        results = []
        for obj in response.objects:
            payload = obj.properties.copy()
            payload["id"] = str(obj.uuid).split("'")[0]
            results.append(OutputData(id=str(obj.uuid).split("'")[0], score=1.0, payload=payload))
        return results

    def reset(self) -> None:
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_collection()
        self.create_collection(self.embedding_model_dims)
