import asyncio
import json
import os
from hypha_rpc import connect_to_server

async def main():
    SERVER_URL = "https://hypha.aicell.io"
    USER_TOKEN = ""
    JSON_DIR = "" 

    vector_collection_manifest = {
        "name": "20 images Vector Collection",
        "description": "20 images Vector Collection",
    }

    vector_collection_config = {
        "vector_fields": [
            {
                "type": "VECTOR",
                "name": "embedding",
                "algorithm": "FLAT",
                "attributes": {
                    "TYPE": "FLOAT32",
                    "DIM": 256,  
                    "DISTANCE_METRIC": "COSINE",
                },
            },
            {"type": "TEXT", "name": "image_name"},
            {"type": "TAG", "name": "mask_id"},
            {"type": "TEXT", "name": "bounding_box"},
            {"type": "TEXT", "name": "mask_coord"},
            {"type": "TAG", "name": "object"},
            {"type": "TAG", "name": "cell_structure"},
            {"type": "TAG", "name": "cell_cycle"},
            {"type": "TAG", "name": "cell_morphology"},
        ],
        "embedding_models": {
            "embedding": "SAM encoder", 
        },
    }

    vectors = []
    for filename in os.listdir(JSON_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(JSON_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict):
                        required_keys = ["image_id", "mask_id", "bounding_box", "embedding", "coordinates"]
                        if not all(key in item for key in required_keys):
                            print(f"Missing keys in {filename}, skipping.")
                            continue
                        if not isinstance(item["embedding"], list) or len(item["embedding"]) != 256:
                            print(f"Invalid embedding in {filename}, skipping.")
                            continue
                        vector = {
                            "embedding": item["embedding"],
                            "image_name": f"{item['image_id']}.png",
                            "mask_id": item["mask_id"],
                            "bounding_box": json.dumps(item["bounding_box"]),
                            "mask_coord": json.dumps(item["coordinates"]),
                            "object": item.get("object", "null"),
                            "cell_structure": item.get("cell_structure", "null"),
                            "cell_cycle": item.get("cell_cycle", "null"),
                            "cell_morphology": item.get("cell_morphology", "null"),
                        }
                        vectors.append(vector)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if not vectors:
        print("No valid vectors to upload.")
        return

    async with connect_to_server(
        {
            "name": "test-deploy-client",
            "server_url": SERVER_URL,
            "token": USER_TOKEN,
            "workspace": "fine-tune-gpt-workspace",
        }
    ) as api:
        artifact_manager = await api.get_service("public/artifact-manager")

        try:
            vector_collection = await artifact_manager.create(
                type="vector-collection",
                manifest=vector_collection_manifest,
                config=vector_collection_config,
            )
            print("\nVector Collection created with ID:", vector_collection.id)
        except Exception as e:
            print(f"Error creating vector collection: {e}")
            return

        try:
            upload_response = await artifact_manager.add_vectors(
                artifact_id=vector_collection.id,
                vectors=vectors,
            )
            print(f"\nUploaded {len(vectors)} vectors successfully.")
            # 打印上传响应，确认上传成功
            print("Upload response:", json.dumps(upload_response, indent=4))
        except Exception as e:
            print(f"Error uploading vectors: {e}")
            return

        try:
            vector_list = await artifact_manager.list_vectors(
                artifact_id=vector_collection.id,
                offset=0,
                limit=10,
            )
            assert len(vector_list) > 0, "No vectors found in the collection."

            first_vector = vector_list[0]
            print("\nFirst vector details:")
            print(json.dumps(first_vector, indent=4))

            if 'embedding' in first_vector:
                print("\n'Embedding' field is present in the first vector.")
            else:
                print("\n'Embedding' field is NOT present in the first vector.")
        except AssertionError as ae:
            print(ae)
        except Exception as e:
            print(f"Error listing vectors: {e}")

        try:
            collection = await artifact_manager.read(artifact_id=vector_collection.id)
            print("\nCollection Details:")
            print(json.dumps(collection, indent=4))


            if 'vectors' in collection and len(collection['vectors']) > 0:
                first_vector_in_collection = collection['vectors'][0]
                print("\nFirst vector's embedding:")
                print(json.dumps(first_vector_in_collection.get('embedding', 'No embedding found'), indent=4))
            else:
                print("No vectors found in the collection details.")
        except Exception as e:
            print(f"Error reading collection details: {e}")

if __name__ == "__main__":
    asyncio.run(main())
