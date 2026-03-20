import base64
from typing import Any, cast

import httpx
from fastapi import HTTPException, status
from supabase import AsyncClient

from src.lib.logging import get_logger
from src.api.v1.admin.messaging._models import MessagingThread
from src.lib.settings import get_settings
from src.lib.supabase import get_public_client, get_tensor_client

logger = get_logger(__name__)

TWILIO_BASE_URL = "https://api.twilio.com/2010-04-01"


# --------------------------------
# Private: Twilio API calls
# --------------------------------


def _twilio_auth_header() -> dict:
    settings = get_settings()
    credentials = base64.b64encode(
        f"{settings.twilio_account_sid}:{settings.twilio_auth_token}".encode()
    ).decode()
    return {"Authorization": f"Basic {credentials}"}


def _twilio_send(to: str, text: str, channel: str, media_url: str | None = None) -> str:
    settings = get_settings()
    # 1. Build From/To based on channel
    if channel == "whatsapp":
        from_addr = f"whatsapp:{settings.twilio_whatsapp_from}"
        to_addr = f"whatsapp:+{to}"
    else:
        from_addr = settings.twilio_whatsapp_from
        to_addr = f"+{to}"

    # 2. Build request payload
    payload: dict = {"From": from_addr, "To": to_addr, "Body": text}
    if media_url:
        payload["MediaUrl"] = media_url

    # 3. Send via Twilio REST API
    try:
        response = httpx.post(
            f"{TWILIO_BASE_URL}/Accounts/{settings.twilio_account_sid}/Messages.json",
            headers=_twilio_auth_header(),
            data=payload,
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        logger.error(f"Twilio send {channel} request failed: {exc}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Twilio API error") from exc

    # 4. Validate success status
    if response.status_code not in (200, 201):
        logger.error(f"Twilio send {channel} failed: status={response.status_code}, body={response.text[:500]}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Twilio API error")

    # 5. Return message SID
    return str(response.json()['sid'])


# --------------------------------
# Private: DB helpers
# --------------------------------

async def _get_thread(client: AsyncClient, person_id: str, channel: str) -> MessagingThread | None:
    result = await (
        client.table("messaging_threads")
        .select("*")
        .eq("person_id", person_id)
        .eq("channel", channel)
        .limit(1)
        .execute()
    )
    if result.data:
        return MessagingThread.model_validate(result.data[0])
    return None


async def _get_or_create_thread(person_id: str, phone_number: str, channel: str) -> str:
    tensor = await get_tensor_client()

    # 1. Look for existing thread
    thread = await _get_thread(tensor, person_id, channel)
    if thread:
        return str(thread.id)

    # 2. Create new thread
    insert_result = await (
        tensor.table("messaging_threads")
        .insert({"person_id": person_id, "channel": channel, "phone_number": phone_number})
        .execute()
    )
    row = MessagingThread.model_validate(insert_result.data[0])
    return str(row.id)


async def _store_message(
    thread_id: str,
    direction: str,
    sender_handle: str,
    body: str,
    external_message_id: str,
    sent_by_person_id: str | None,
) -> None:
    # Insert message; skip silently if external_message_id already exists
    tensor = await get_tensor_client()
    await tensor.table("messaging_messages").upsert(
        {
            "thread_id": thread_id,
            "direction": direction,
            "sender_handle": sender_handle,
            "body": body,
            "external_message_id": external_message_id,
            "sent_by_person_id": sent_by_person_id,
        },
        on_conflict="external_message_id",
        ignore_duplicates=True,
    ).execute()


# --------------------------------
# Public: reusable functions
# --------------------------------


async def send_message(person_id: str, text: str, sent_by_person_id: str | None = None) -> dict:
    """Send a message to a user, routing to Twilio SMS (CONCIERGE) or Twilio WhatsApp (all others)."""
    settings = get_settings()

    # 1. Look up user's membership and phone number
    public = await get_public_client()
    user_result = await (
        public.table("user_information")
        .select("phone_number,membership")
        .eq("person_id", person_id)
        .limit(1)
        .execute()
    )
    if not user_result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user = cast(dict[str, Any], user_result.data[0])
    phone = str(user["phone_number"])
    membership = str(user["membership"])

    # 2. Choose channel and optional media
    channel = "sms" if membership == "CONCIERGE" else "whatsapp"
    media_url = None
    if channel == "sms":
        tensor = await get_tensor_client()
        existing_thread = await _get_thread(tensor, person_id, "sms")
        media_url = settings.vcf_url if not existing_thread and settings.vcf_url else None

    # 3. Send via Twilio
    message_sid = _twilio_send(phone, text, channel=channel, media_url=media_url)

    # 4. Get or create thread
    thread_id = await _get_or_create_thread(person_id, phone, channel=channel)

    # 5. Store outbound message
    await _store_message(
        thread_id=thread_id,
        direction="outbound",
        sender_handle=settings.twilio_whatsapp_from,
        body=text,
        external_message_id=message_sid,
        sent_by_person_id=sent_by_person_id,
    )

    logger.info(f"Sent {channel} to person {person_id}, sid={message_sid}")
    return {"thread_id": thread_id, "message_id": message_sid}


async def store_inbound_message(
    sender_handle: str,
    body: str,
    twilio_message_sid: str,
    channel: str,
) -> dict:
    """Store an inbound Twilio message. Returns stored=False if sender unknown."""
    # 1. Validate channel and look up person_id by normalized phone number
    if channel not in {"sms", "whatsapp"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported channel")

    phone = sender_handle
    public = await get_public_client()
    user_result = await (
        public.table("user_information")
        .select("person_id")
        .eq("phone_number", phone)
        .limit(1)
        .execute()
    )

    person_id: str | None = None
    if user_result.data:
        person_id = str(cast(dict[str, Any], user_result.data[0])["person_id"])

    if not person_id:
        logger.info(f"Inbound {channel} from unknown number {phone}")
        return {"stored": False}

    # 2. Get or create thread
    thread_id = await _get_or_create_thread(person_id, phone, channel=channel)

    # 3. Store inbound message
    await _store_message(
        thread_id=thread_id,
        direction="inbound",
        sender_handle=phone,
        body=body,
        external_message_id=twilio_message_sid,
        sent_by_person_id=None,
    )

    logger.info(f"Stored inbound {channel} from {phone} (person {person_id}), sid={twilio_message_sid}")
    return {"stored": True}
