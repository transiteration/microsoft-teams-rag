import os
import re
import json
import time
import uuid
import msal
import torch
import logging
import requests
import tempfile
import argparse
import warnings
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone

from google import genai
from google.genai.types import EmbedContentConfig

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from langchain_docling import DoclingLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

logging.info(
    f"Docling will use device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
)

TARGET_TEAM_NAME = os.getenv("MS_TEAM_NAME")
MS_CHANNEL_WHITELIST_STR = os.getenv("MS_CHANNEL_WHITELIST")
TARGET_CHANNELS = (
    {name.strip() for name in MS_CHANNEL_WHITELIST_STR.split(",")}
    if MS_CHANNEL_WHITELIST_STR
    else None
)
TENANT_ID = os.getenv("MS_TENANT_ID")
CLIENT_ID = os.getenv("MS_CLIENT_ID")
CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE"))

FILE_STATE_STORE = "./state/file_ingest_state.json"
THREAD_DELTA_STORE = "./state/thread_delta_links.json"
DEFAULT_START_TIME_STR = os.getenv("DEFAULT_START_TIME_STR")
SUPPORTED_FILE_EXTENSIONS = tuple(
    ext.strip() for ext in os.getenv("SUPPORTED_FILE_EXTENSIONS").split(",")
)


def get_graph_api_token() -> str | None:
    """Get Microsoft Graph API token"""
    logging.info("Acquiring Microsoft Graph API token...")
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        client_credential=CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )
    if "access_token" in result:
        logging.info("Successfully acquired API token.")
        return result.get("access_token")
    else:
        logging.error(f"Failed to acquire token: {result.get('error_description')}")
        return None


def initialize_qdrant_collection(client: QdrantClient):
    """Ensure Qdrant collection exists"""
    try:
        client.get_collection(collection_name=QDRANT_COLLECTION)
        logging.info(f"Qdrant collection '{QDRANT_COLLECTION}' already exists.")
    except (UnexpectedResponse, Exception):
        logging.info(f"Collection '{QDRANT_COLLECTION}' not found. Creating it...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION, distance=models.Distance.COSINE
            ),
        )
        logging.info(f"Collection '{QDRANT_COLLECTION}' created successfully.")


def embed_and_upsert_chunks(
    qdrant_client: QdrantClient,
    genai_client: genai.Client,
    chunks: list[Document],
    task_type: str,
    qdrant_batch_size: int = 100,
):
    """
    Embeds document chunks in batches and upserts them into Qdrant.
    This is much more efficient than embedding one-by-one.
    """
    logging.info(
        f"Embedding {len(chunks)} chunks in batches of {EMBEDDING_BATCH_SIZE} and upserting to Qdrant..."
    )

    points_to_upsert = []
    total_chunks = len(chunks)

    for i in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
        batch_chunks = chunks[i : i + EMBEDDING_BATCH_SIZE]
        texts_to_embed = [chunk.page_content for chunk in batch_chunks]
        logging.info(
            f"  Processing batch {i // EMBEDDING_BATCH_SIZE + 1}/{-(-total_chunks // EMBEDDING_BATCH_SIZE)} (chunks {i + 1}-{min(i + EMBEDDING_BATCH_SIZE, total_chunks)})..."
        )
        try:
            response = genai_client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=texts_to_embed,
                config=EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSION,
                ),
            )
            embeddings = response.embeddings
            if len(embeddings) != len(batch_chunks):
                logging.error(
                    f"  Mismatch in embedding count for batch. Expected {len(batch_chunks)}, got {len(embeddings)}. Skipping batch."
                )
                continue
        except Exception as e:
            logging.error(f"  Failed to embed batch starting at chunk {i}: {e}")
            continue

        for chunk, embedding in zip(batch_chunks, embeddings):
            payload = chunk.metadata.copy()
            payload["page_content"] = chunk.page_content
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.values,
                payload=payload,
            )
            points_to_upsert.append(point)

        if len(points_to_upsert) >= qdrant_batch_size:
            try:
                qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points_to_upsert,
                    wait=True,
                )
                logging.info(
                    f"    Successfully upserted a batch of {len(points_to_upsert)} points to Qdrant."
                )
                points_to_upsert = []
            except Exception as e:
                logging.error(f"    Failed to upsert batch to Qdrant: {e}")
                points_to_upsert = []

    if points_to_upsert:
        try:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points_to_upsert,
                wait=True,
            )
            logging.info(
                f"  Successfully upserted final batch of {len(points_to_upsert)} points to Qdrant."
            )
        except Exception as e:
            logging.error(f"  Failed to upsert final batch to Qdrant: {e}")


def load_file_ingest_timestamp() -> str:
    """Load last file ingest timestamp"""
    if os.path.exists(FILE_STATE_STORE):
        with open(FILE_STATE_STORE, "r", encoding="utf-8") as f:
            state = json.load(f)
            return state.get("last_successful_run_utc", DEFAULT_START_TIME_STR)
    return DEFAULT_START_TIME_STR


def save_file_ingest_timestamp(timestamp_utc_str: str) -> None:
    """Save current file ingest timestamp"""
    with open(FILE_STATE_STORE, "w", encoding="utf-8") as f:
        json.dump({"last_successful_run_utc": timestamp_utc_str}, f, indent=2)


def find_new_or_updated_files(
    token: str, team_id: str, channel_id: str, last_run_iso: str
) -> list[dict]:
    """Find new or updated files in a channel"""
    headers = {"Authorization": f"Bearer {token}"}
    files_to_process = []
    try:
        folder_info_url = (
            f"{GRAPH_API_ENDPOINT}/teams/{team_id}/channels/{channel_id}/filesFolder"
        )
        folder_resp = requests.get(folder_info_url, headers=headers)
        folder_resp.raise_for_status()
        folder_data = folder_resp.json()
        drive_id = folder_data["parentReference"]["driveId"]
        root_item_id = folder_data["id"]
    except requests.HTTPError as e:
        logging.warning(
            f"Could not access file folder for channel {channel_id}. Skipping. Error: {e.response.text}"
        )
        return []

    folders_to_visit = [(root_item_id, "")]
    while folders_to_visit:
        current_folder_id, current_path = folders_to_visit.pop()
        children_url = (
            f"{GRAPH_API_ENDPOINT}/drives/{drive_id}/items/{current_folder_id}/children"
        )
        while children_url:
            try:
                items_resp = requests.get(children_url, headers=headers)
                items_resp.raise_for_status()
                items_data = items_resp.json()
            except requests.HTTPError as e:
                logging.error(
                    f"Failed to get items in folder {current_folder_id}. Error: {e}"
                )
                break

            for item in items_data.get("value", []):
                item_name = item.get("name")
                if "folder" in item:
                    folders_to_visit.append((item["id"], f"{current_path}/{item_name}"))
                    continue
                if "file" in item and item_name.lower().endswith(
                    SUPPORTED_FILE_EXTENSIONS
                ):
                    last_modified_str = item.get("lastModifiedDateTime")
                    if last_modified_str > last_run_iso:
                        files_to_process.append(
                            {
                                "id": item["id"],
                                "name": item_name,
                                "webUrl": item["webUrl"],
                                "downloadUrl": item["@microsoft.graph.downloadUrl"],
                                "last_modified": last_modified_str,
                            }
                        )
            children_url = items_data.get("@odata.nextLink")
    return files_to_process


def ingest_files(client: QdrantClient, genai_client: genai.Client):
    """
    Ingest files from all channels in a Team, processing and uploading on a
    per-channel basis.
    """
    logging.info("=" * 50)
    logging.info("STARTING: Microsoft Teams File Ingestion")
    logging.info("=" * 50)

    initial_token = get_graph_api_token()
    if not initial_token:
        logging.error(
            "Failed to acquire initial Graph API token. Aborting file ingestion."
        )
        return

    headers = {"Authorization": f"Bearer {initial_token}"}
    try:
        teams_resp = requests.get(
            f"{GRAPH_API_ENDPOINT}/teams?$filter=displayName eq '{TARGET_TEAM_NAME}'",
            headers=headers,
        )
        teams_resp.raise_for_status()
        teams = teams_resp.json().get("value", [])
        if not teams:
            logging.error(
                f"Team '{TARGET_TEAM_NAME}' not found. Aborting file ingestion."
            )
            return
        team_id = teams[0]["id"]
        logging.info(f"Targeting Team '{TARGET_TEAM_NAME}' (ID: {team_id})")

        channels_url = f"{GRAPH_API_ENDPOINT}/teams/{team_id}/channels"
        all_channels = []
        while channels_url:
            channels_resp = requests.get(channels_url, headers=headers)
            channels_resp.raise_for_status()
            data = channels_resp.json()
            all_channels.extend(data.get("value", []))
            channels_url = data.get("@odata.nextLink")
    except requests.HTTPError as e:
        logging.error(
            f"Failed to get Team or Channel info. Error: {e.response.text}. Aborting file ingestion."
        )
        return

    if not all_channels:
        logging.warning(
            f"No channels found in team '{TARGET_TEAM_NAME}'. Exiting file ingestion."
        )
        return

    if TARGET_CHANNELS:
        logging.info(
            f"Channel whitelist is active. Only processing: {list(TARGET_CHANNELS)}"
        )
        channels_to_process = [
            ch for ch in all_channels if ch["displayName"] in TARGET_CHANNELS
        ]
    else:
        channels_to_process = all_channels

    num_channels_to_process = len(channels_to_process)
    logging.info(f"Found {num_channels_to_process} channels to scan for files.")

    last_run_timestamp = load_file_ingest_timestamp()
    start_time_utc = datetime.now(timezone.utc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    total_files_processed = 0
    total_chunks_ingested = 0

    logging.info(f"Checking for files modified after: {last_run_timestamp}")

    for i, channel in enumerate(channels_to_process, 1):
        channel_id, channel_name = channel["id"], channel["displayName"]

        logging.info("-" * 40)
        logging.info(
            f"[{i}/{num_channels_to_process}] Scanning Channel: '{channel_name}'"
        )

        token_for_channel = get_graph_api_token()
        if not token_for_channel:
            logging.warning(
                f"Could not get a token for channel '{channel_name}'. Skipping."
            )
            continue

        files_in_channel = find_new_or_updated_files(
            token_for_channel, team_id, channel_id, last_run_timestamp
        )

        if not files_in_channel:
            logging.info("  No new or updated files found in this channel.")
            continue

        logging.info(
            f"  Found {len(files_in_channel)} new/updated file(s) to process in this channel."
        )
        channel_chunks_to_process = []

        for file_idx, file_info in enumerate(files_in_channel, 1):
            file_id, file_name = file_info["id"], file_info["name"]
            logging.info(
                f"    Processing file {file_idx}/{len(files_in_channel)}: '{file_name}'"
            )
            logging.info(f"    Deleting old versions of '{file_name}' from Qdrant...")
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.file_id",
                                match=models.MatchValue(value=file_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            tmp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(file_name).suffix
                ) as tmp:
                    response = requests.get(file_info["downloadUrl"])
                    response.raise_for_status()
                    tmp.write(response.content)
                    tmp_file_path = tmp.name
                loader = DoclingLoader(file_path=tmp_file_path)
                docs = loader.load()
            except Exception as e:
                logging.error(f"    Error loading file {file_name}: {e}. Skipping.")
                continue
            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

            for doc in docs:
                doc.page_content = f"File Name: {file_name}\n\n{doc.page_content}"
                doc.metadata = {
                    "source": file_info["webUrl"],
                    "file_name": file_name,
                    "file_id": file_id,
                    "channel_name": channel_name,
                    "team_name": TARGET_TEAM_NAME,
                    "last_modified": file_info["last_modified"],
                }
            chunks = splitter.split_documents(docs)
            channel_chunks_to_process.extend(chunks)
            logging.info(f"    Split into {len(chunks)} chunks. Queued for ingestion.")

        if channel_chunks_to_process:
            logging.info(
                f"  Starting ingestion for {len(channel_chunks_to_process)} chunks from channel '{channel_name}'..."
            )
            embed_and_upsert_chunks(
                qdrant_client=client,
                genai_client=genai_client,
                chunks=channel_chunks_to_process,
                task_type="RETRIEVAL_DOCUMENT",
            )
            total_files_processed += len(files_in_channel)
            total_chunks_ingested += len(channel_chunks_to_process)
            logging.info(f"  Completed ingestion for channel '{channel_name}'.")

    final_timestamp = start_time_utc.isoformat().replace("+00:00", "Z")
    save_file_ingest_timestamp(final_timestamp)

    logging.info("=" * 50)
    logging.info("File Ingestion Summary")
    if total_files_processed > 0:
        logging.info(
            f"Total files processed across all channels: {total_files_processed}"
        )
        logging.info(f"Total chunks ingested into Qdrant: {total_chunks_ingested}")
    else:
        logging.info("No new or updated files were found across all targeted channels.")
    logging.info(f"Next run will check for files modified after {final_timestamp}")
    logging.info("File ingestion complete!")


def load_thread_delta_links() -> dict[str, str]:
    """Load thread delta links"""
    if os.path.exists(THREAD_DELTA_STORE):
        with open(THREAD_DELTA_STORE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_thread_delta_links(delta_dict: dict[str, str]) -> None:
    """Save thread delta links"""
    with open(THREAD_DELTA_STORE, "w", encoding="utf-8") as f:
        json.dump(delta_dict, f, indent=2)


def clean_html(raw_html: str) -> str:
    """Remove HTML tags from string"""
    return re.sub("<.*?>", "", raw_html)


def get_message_author_name(msg: dict) -> str:
    """Get author name from message"""
    from_details = msg.get("from")
    if not from_details:
        return "System Message"
    if user := from_details.get("user"):
        return user.get("displayName", "Unknown User")
    if app := from_details.get("application"):
        return app.get("displayName", "Unknown Application")
    return "Unknown Sender"


def fetch_messages_for_channel(
    token: str,
    team_id: str,
    channel_id: str,
    channel_name: str,
    start_or_delta_url: str,
) -> tuple[list[Document], list[str], str | None]:
    """
    Fetch new or updated Teams messages for a single channel using its delta link.
    Returns documents, a list of parent message IDs to replace, and the new delta link.
    """
    headers = {"Authorization": f"Bearer {token}"}
    docs, thread_ids_to_replace = [], []
    new_delta_link = None
    next_url = start_or_delta_url

    while next_url:
        try:
            r = requests.get(next_url, headers=headers)
            r.raise_for_status()
            data = r.json()
        except requests.HTTPError as e:
            logging.error(
                f"  Failed to fetch messages for channel {channel_name}: {e.response.text}"
            )
            return [], [], None
        for msg in data.get("value", []):
            parent_msg_id = msg["id"]
            thread_ids_to_replace.append(parent_msg_id)
            body_content = clean_html(
                (msg.get("body") or {}).get("content") or ""
            ).strip()
            author = get_message_author_name(msg)
            ts = msg["createdDateTime"]
            subject = (msg.get("subject") or "").strip()
            thread_web_url = msg.get("webUrl")
            if not body_content and author == "System Message":
                continue
            conversation_parts = [
                f"[{ts} | {channel_name}] {author}{f' (Post Subject: {subject})' if subject else ''}: {body_content}"
            ]
            last_modified_thread = msg.get("lastModifiedDateTime", ts)
            replies_url = f"{GRAPH_API_ENDPOINT}/teams/{team_id}/channels/{channel_id}/messages/{parent_msg_id}/replies"
            while replies_url:
                try:
                    rep = requests.get(replies_url, headers=headers)
                    rep.raise_for_status()
                    rep_data = rep.json()
                    for reply in rep_data.get("value", []):
                        reply_content = clean_html(
                            (reply.get("body") or {}).get("content") or ""
                        ).strip()
                        if not reply_content:
                            continue
                        reply_author = get_message_author_name(reply)
                        reply_ts = reply.get("createdDateTime", "")
                        if reply.get("lastModifiedDateTime", "") > last_modified_thread:
                            last_modified_thread = reply["lastModifiedDateTime"]
                        conversation_parts.append(
                            f"[{reply_ts} | {channel_name}] {reply_author} (reply): {reply_content}"
                        )
                    replies_url = rep_data.get("@odata.nextLink")
                except requests.HTTPError as e:
                    logging.warning(
                        f"    Could not fetch replies for message {parent_msg_id}: {e.response.text}"
                    )
                    replies_url = None
            docs.append(
                Document(
                    page_content="\n---\n".join(conversation_parts),
                    metadata={
                        "source": thread_web_url,
                        "team_name": TARGET_TEAM_NAME,
                        "channel_name": channel_name,
                        "author": author,
                        "message_id": parent_msg_id,
                        "created_datetime": ts,
                        "last_modified_thread": last_modified_thread,
                        "subject": subject,
                    },
                )
            )
        next_url = data.get("@odata.nextLink")
        if not next_url and "@odata.deltaLink" in data:
            new_delta_link = data["@odata.deltaLink"]
    return docs, list(set(thread_ids_to_replace)), new_delta_link


def ingest_threads(client: QdrantClient, genai_client: genai.Client):
    """
    Ingest message threads from a Team, processing and uploading on a
    per-channel basis.
    """
    logging.info("=" * 50)
    logging.info("STARTING: Microsoft Teams Thread Ingestion")
    logging.info("=" * 50)

    initial_token = get_graph_api_token()
    if not initial_token:
        logging.error(
            "Failed to acquire initial Graph API token. Aborting thread ingestion."
        )
        return

    headers = {"Authorization": f"Bearer {initial_token}"}
    try:
        teams_resp = requests.get(
            f"{GRAPH_API_ENDPOINT}/teams?$filter=displayName eq '{TARGET_TEAM_NAME}'",
            headers=headers,
        )
        teams_resp.raise_for_status()
        teams = teams_resp.json().get("value", [])
        if not teams:
            logging.error(
                f"Team '{TARGET_TEAM_NAME}' not found. Aborting thread ingestion."
            )
            return
        team_id = teams[0]["id"]
        logging.info(f"Targeting Team '{TARGET_TEAM_NAME}' (ID: {team_id})")

        channels_url = f"{GRAPH_API_ENDPOINT}/teams/{team_id}/channels"
        all_channels = []
        while channels_url:
            channels_resp = requests.get(channels_url, headers=headers)
            channels_resp.raise_for_status()
            data = channels_resp.json()
            all_channels.extend(data.get("value", []))
            channels_url = data.get("@odata.nextLink")
    except requests.HTTPError as e:
        logging.error(
            f"Failed to get Team or Channel info. Error: {e.response.text}. Aborting thread ingestion."
        )
        return

    if not all_channels:
        logging.warning(
            f"No channels found in team '{TARGET_TEAM_NAME}'. Exiting thread ingestion."
        )
        return

    if TARGET_CHANNELS:
        logging.info(
            f"Channel whitelist is active. Only processing: {list(TARGET_CHANNELS)}"
        )
        channels_to_process = [
            ch for ch in all_channels if ch["displayName"] in TARGET_CHANNELS
        ]
    else:
        channels_to_process = all_channels

    num_channels_to_process = len(channels_to_process)
    logging.info(f"Found {num_channels_to_process} channels to scan for threads.")

    delta_cache = load_thread_delta_links()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    total_threads_processed = 0
    total_chunks_ingested = 0

    for i, channel in enumerate(channels_to_process, 1):
        channel_id, channel_name = channel["id"], channel["displayName"]

        logging.info("-" * 40)
        logging.info(
            f"[{i}/{num_channels_to_process}] Scanning Channel for threads: '{channel_name}'"
        )

        token_for_channel = get_graph_api_token()
        if not token_for_channel:
            logging.warning(
                f"Could not get a token for channel '{channel_name}'. Skipping."
            )
            continue

        start_or_delta_url = delta_cache.get(channel_id) or (
            f"{GRAPH_API_ENDPOINT}/teams/{team_id}/channels/{channel_id}/messages/delta"
            f"?$filter=lastModifiedDateTime gt {DEFAULT_START_TIME_STR}"
        )
        docs_for_channel, ids_to_replace, new_delta_link = fetch_messages_for_channel(
            token_for_channel, team_id, channel_id, channel_name, start_or_delta_url
        )

        if not docs_for_channel and not ids_to_replace:
            logging.info("  No new or updated messages found in this channel.")
            if new_delta_link:
                delta_cache[channel_id] = new_delta_link
            continue
        logging.info(
            f"  Found {len(docs_for_channel)} new/updated threads to process in this channel."
        )

        if ids_to_replace:
            logging.info(
                f"    Deleting {len(ids_to_replace)} old thread versions from Qdrant..."
            )
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.message_id",
                                match=models.MatchAny(any=ids_to_replace),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logging.info("    Deletion complete.")

        if docs_for_channel:
            chunks = splitter.split_documents(docs_for_channel)
            logging.info(f"    Split into {len(chunks)} chunks. Starting ingestion...")
            embed_and_upsert_chunks(
                qdrant_client=client,
                genai_client=genai_client,
                chunks=chunks,
                task_type="RETRIEVAL_DOCUMENT",
            )
            total_threads_processed += len(docs_for_channel)
            total_chunks_ingested += len(chunks)
            logging.info(f"  Completed ingestion for channel '{channel_name}'.")

        if new_delta_link:
            delta_cache[channel_id] = new_delta_link
            logging.info(f"  Updated delta link for channel '{channel_name}'.")

    save_thread_delta_links(delta_cache)

    logging.info("=" * 50)
    logging.info("Thread Ingestion Summary")
    if total_threads_processed > 0:
        logging.info(
            f"Total threads processed across all channels: {total_threads_processed}"
        )
        logging.info(f"Total chunks ingested into Qdrant: {total_chunks_ingested}")
    else:
        logging.info(
            "No new or updated messages were found across all targeted channels."
        )
    logging.info("Thread ingestion complete!")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Ingest data from Microsoft Teams into a Qdrant vectordb.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--type",
        choices=["files", "threads"],
        default=None,
        help="Specify the type of data to ingest.\n"
        "  'files': Ingests new or updated files from Team channels.\n"
        "  'threads': Ingests new or updated message threads.\n"
        "  (default): Ingests both files and threads.",
    )
    args = parser.parse_args()

    logging.info("--- Starting Ingestion Pipeline ---")

    qdrant_client = QdrantClient(url=QDRANT_URL)
    initialize_qdrant_collection(qdrant_client)
    os.makedirs("state", exist_ok=True)

    logging.info("Initializing Google GenAI client...")
    genai_client = genai.Client(
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        vertexai=True,
    )
    logging.info("Google GenAI client initialized.")

    run_files = args.type is None or args.type == "files"
    run_threads = args.type is None or args.type == "threads"

    if run_files:
        ingest_files(client=qdrant_client, genai_client=genai_client)

    if run_threads:
        ingest_threads(client=qdrant_client, genai_client=genai_client)

    elapsed = time.time() - start_time
    logging.info(
        f"--- Ingestion Pipeline Finished --- (Elapsed: {elapsed:.2f} seconds)"
    )
